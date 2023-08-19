# Environment ------------------------------------------------------------------
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
    Path('data', 'processed', 'cohort_trs-n_1387-desc_12m.csv')
)

# Model Development ------------------------------------------------------------
auroc_list = list()
brier_list = list()
tr_cv_auroc = list()

for site in [1, 2, 3, 4]:

    # leave one site out
    train_df = df.query("Hosp_cluster != @site").drop(columns='Hosp_cluster')
    test_df = df.query("Hosp_cluster == @site").drop(columns='Hosp_cluster')

    tr_y = train_df.pop('is_cloz').to_numpy()
    te_y = test_df.pop('is_cloz').to_numpy()
    tr_x = train_df.to_numpy()
    te_x = test_df.to_numpy()

    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_median.fit(tr_x)
    tr_x = imp_median.transform(tr_x)
    te_x = imp_median.transform(te_x)

    # select pipeline using TPOT
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

    auroc_list.append(roc_auc_score(te_y, te_y_calib_bag_hat))
    brier_list.append(brier_score_loss(te_y, te_y_calib_bag_hat))

losocv_performacne_df = pd.DataFrame({
    'Site': [1, 2, 3, 4],
    'AUROC': auroc_list,
    'Brier Score': brier_list
})
losocv_performacne_df.to_csv(
    Path('outputs', 'tables', 'losocv_performacne_12m.csv'),
    index=False
)
