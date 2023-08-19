# Environment ------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
from collections import Counter
from functools import reduce

# customize modules
from src.Python import viz

import warnings
warnings.filterwarnings('ignore')


# Data I/O ---------------------------------------------------------------------
df = pd.read_csv(
    Path('data', 'processed', 'cohort_trs-n_1379-desc_24m.csv')
).drop(columns='Hosp_cluster')

fi_array = np.load(
    Path('outputs', 'cache', 'automl-data_24m-desc_featureimportance.npy')
)

fi_df = viz.featureimportance(
    fi_array, df.drop(columns='is_cloz').columns.values.tolist()
).torank().to_frame()

fi_df.columns = ['importance']
fi_df.reset_index(inplace=True)
sorted_df = fi_df.sort_values(by='importance', ascending=False)
top20_features = sorted_df['index'].tolist()[0:20]

# create numpy array for features and target
y = df.pop('is_cloz').to_numpy()
x = df[top20_features].to_numpy()

# Model Development ------------------------------------------------------------
feature_importance = list()
tr_cv_auroc = list()
tpot_te_y = list()
tpot_te_pred = list()
tpot_te_probas = list()
tpot_bag_te_pred = list()
tpot_bag_te_probas = list()
tpot_bag_calib_te_pred = list()
tpot_bag_calib_te_probas = list()
choosen_pipeline = list()

for iteration in range(100):

    print(f'\nNow running iteration {iteration+1}\n')

    # make sure won't get the same random split from others
    iteration = iteration + 2600

    # randomly split the data to train and test
    tr_x, te_x, tr_y, te_y = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=iteration
    )

    tpot_te_y.append(te_y)

    # median impute
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_median.fit(tr_x)
    tr_x = imp_median.transform(tr_x)
    te_x = imp_median.transform(te_x)

    # select best pipeline using TPOT
    pipeline_optimizer = TPOTClassifier(
        generations=10, population_size=50, cv=5, random_state=1234,
        template='StandardScaler-Classifier',
        mutation_rate=0.8, crossover_rate=0.2,
        scoring='roc_auc', verbosity=2, n_jobs=4
    )
    pipeline_optimizer.fit(tr_x, tr_y)
    te_y_pred = pipeline_optimizer.predict(te_x)
    try:
        te_y_hat = pipeline_optimizer.predict_proba(te_x)[:, 1]
        print(f'TPOT AUROC: {roc_auc_score(te_y, te_y_hat)}')
    except:
        te_y_hat = [np.nan] * te_y_pred.size

    print(f'TPOT Brier: {brier_score_loss(te_y, te_y_pred)}')

    # append
    tpot_te_pred.append(te_y_pred)
    tpot_te_probas.append(te_y_hat)

    # save pipeline name
    choosen_pipeline.append(pipeline_optimizer.fitted_pipeline_.steps[1][0])

    # apply bagging to optimized pipeline and save feature importance
    best_model_bagging = BaggingClassifier(
        base_estimator=pipeline_optimizer.fitted_pipeline_,
        n_estimators=100,
        random_state=1234,
        n_jobs=4
    )

    tr_cv_auroc.append(
        np.nanmean(
            cross_val_score(
                best_model_bagging, tr_x, tr_y, cv=5, scoring='roc_auc',
                n_jobs=4
            )
        )
    )

    best_model_bagging.fit(tr_x, tr_y)

    te_y_bag_pred = best_model_bagging.predict(te_x)
    try:
        te_y_bag_hat = best_model_bagging.predict_proba(te_x)[:, 1]
        print(f'TPOT Bag AUROC: {roc_auc_score(te_y, te_y_bag_hat)}')
    except:
        te_y_bag_hat = [np.nan] * te_y_bag_pred.size

    print(f'TPOT Bag Brier: {brier_score_loss(te_y, te_y_bag_pred)}')

    # append
    tpot_bag_te_pred.append(te_y_bag_pred)
    tpot_bag_te_probas.append(te_y_bag_hat)

    try:
        estimators_flatten_ = [[model for model in pipeline]
                               for pipeline in best_model_bagging.estimators_]
        _, bagged_models_ = zip(*estimators_flatten_)
        feature_importance.append(
            np.mean([model.feature_importances_ for model in bagged_models_], axis=0)
        )
    except AttributeError:
        pass

    # refit the optimized pipeline with bagging and calibration
    calib_model = CalibratedClassifierCV(
        best_model_bagging,
        cv=5,
        method='sigmoid',
        n_jobs=4
    )
    calib_model.fit(tr_x, tr_y)

    te_y_calib_bag_pred = calib_model.predict(te_x)

    # some pipeline may not have predict_proba function
    try:
        te_y_calib_bag_hat = calib_model.predict_proba(te_x)[:, 1]
        print(f'Leaveout AUROC: {roc_auc_score(te_y, te_y_calib_bag_hat)}')
    except:
        te_y_hat = [np.nan] * te_y_calib_bag_pred.size

    print(f'Leaveout Brier: {brier_score_loss(te_y, te_y_calib_bag_pred)}')

    # append
    tpot_bag_calib_te_pred.append(te_y_calib_bag_pred)
    tpot_bag_calib_te_probas.append(te_y_calib_bag_hat)


# caching the outputs for future use
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_tr_cv_auroc.npy'),
    np.asarray(tr_cv_auroc)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_featureimportance.npy'),
    np.asarray(feature_importance)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_y.npy'),
    np.asarray(tpot_te_y)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_tpot_pred.npy'),
    np.asarray(tpot_te_pred)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_tpot_probas.npy'),
    np.asarray(tpot_te_probas)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_bag_pred.npy'),
    np.asarray(tpot_bag_te_pred)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_bag_probas.npy'),
    np.asarray(tpot_bag_te_probas)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_final_pred.npy'),
    np.asarray(tpot_bag_calib_te_pred)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_final_probas.npy'),
    np.asarray(tpot_bag_calib_te_probas)
)
np.save(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_choosen_pipeline.npy'),
    np.asarray(choosen_pipeline)
)

# tr_cv_auroc = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_tr_cv_auroc.npy'))
# feature_importance = np.load(Path('outputs', 'cache', 'automl-data_top20_24m-desc_featureimportance.npy'))
# tpot_te_y = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_y.npy'))
# tpot_te_probas = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_tpot_probas.npy'))
# tpot_te_pred = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_tpot_pred.npy'))
# tpot_bag_te_probas = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_bag_probas.npy'))
# tpot_bag_te_pred = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_bag_pred.npy'))
# tpot_bag_calib_te_probas = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_final_probas.npy'))
# tpot_bag_calib_te_pred = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_final_pred.npy'))
# choosen_pipeline = np.load(Path('outputs', 'cache', 'automl-data_24m_top20-desc_choosen_pipeline.npy'))
