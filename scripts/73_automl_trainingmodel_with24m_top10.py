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
import pickle

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
top10_features = sorted_df['index'].tolist()[0:10]

sorted_df['index'][0:10].to_csv(
    Path('outputs', 'tables', 'data_24m-decs_top10_features.csv'),
    index=False
)

# create numpy array for features and target
tr_y = df.pop('is_cloz').to_numpy()
tr_x = df[top10_features].to_numpy()

# Model Development ------------------------------------------------------------

# median impute
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(tr_x)
tr_x = imp_median.transform(tr_x)


# select best pipeline using TPOT
pipeline_optimizer = TPOTClassifier(
    generations=10, population_size=50, cv=5, random_state=1234,
    template='StandardScaler-Classifier',
    mutation_rate=0.8, crossover_rate=0.2,
    scoring='roc_auc', verbosity=2, n_jobs=8
)
pipeline_optimizer.fit(tr_x, tr_y)

# apply bagging to optimized pipeline and save feature importance
best_model_bagging = BaggingClassifier(
    base_estimator=pipeline_optimizer.fitted_pipeline_,
    n_estimators=100,
    random_state=1234,
    n_jobs=4
)
best_model_bagging.fit(tr_x, tr_y)

# refit the optimized pipeline with bagging and calibration
calib_model = CalibratedClassifierCV(
    best_model_bagging,
    cv=5,
    method='sigmoid',
    n_jobs=4
)
calib_model.fit(tr_x, tr_y)

filename = Path('outputs', 'cache', '24m_allsubjs_top10_finalized_model.sav')
pickle.dump(calib_model, open(filename, 'wb'))

# calib_model = pickle.load(open(filename, 'rb'))

# testing
input = np.array(
    [[0, 16.358904, 1, 0.541667, 1, 1.458333, 16, 0.956522, 65.833333, 17]])
calib_model.predict_proba(input)[:, 1]
