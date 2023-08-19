# Evironment -------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

from src.Python import viz


# Data I/O ---------------------------------------------------------------------
trs_baseline_df = pd.read_csv(
    Path('data', 'processed', 'cohort_trs-n_1398-desc_baseline.csv')
).drop(columns=['is_cloz', 'Hosp_cluster'])

trs_12m_df = pd.read_csv(
    Path('data', 'processed', 'cohort_trs-n_1387-desc_12m.csv')
).drop(columns=['is_cloz', 'Hosp_cluster'])

trs_24m_df = pd.read_csv(
    Path('data', 'processed', 'cohort_trs-n_1379-desc_24m.csv')
).drop(columns=['is_cloz', 'Hosp_cluster'])

trs_36m_df = pd.read_csv(
    Path('data', 'processed', 'cohort_trs-n_1363-desc_36m.csv')
).drop(columns=['is_cloz', 'Hosp_cluster'])

fi_baseline_100repeats_array = np.load(
    Path('outputs', 'cache', 'automl-data_baseline-desc_featureimportance.npy')
)

fi_12m_100repeats_array = np.load(
    Path('outputs', 'cache', 'automl-data_12m-desc_featureimportance.npy')
)

fi_24m_100repeats_array = np.load(
    Path('outputs', 'cache', 'automl-data_24m-desc_featureimportance.npy')
)

fi_36m_100repeats_array = np.load(
    Path('outputs', 'cache', 'automl-data_36m-desc_featureimportance.npy')
)


# Data Preparation -------------------------------------------------------------
fi_baseline_df = viz.featureimportance(
    fi_baseline_100repeats_array, trs_baseline_df.columns.values.tolist()
).torank().to_frame()

fi_12m_df = viz.featureimportance(
    fi_12m_100repeats_array, trs_12m_df.columns.values.tolist()
).torank().to_frame()

fi_24m_df = viz.featureimportance(
    fi_24m_100repeats_array, trs_24m_df.columns.values.tolist()
).torank().to_frame()

fi_36m_df = viz.featureimportance(
    fi_36m_100repeats_array, trs_36m_df.columns.values.tolist()
).torank().to_frame()

# reset index and add column names
fi_baseline_df.reset_index(inplace=True)
fi_baseline_df.columns = ['Variable', 'Baseline']
fi_12m_df.reset_index(inplace=True)
fi_12m_df.columns = ['Variable', '12-month']
fi_24m_df.reset_index(inplace=True)
fi_24m_df.columns = ['Variable', '24-month']
fi_36m_df.reset_index(inplace=True)
fi_36m_df.columns = ['Variable', '36-month']

labels = [
    'Age at first service contact', 'Age of illness onset', 'Sex',
    'Years of education', 'First episode duation', 'DUP SA history', 'DUP NSSI history',
    'First episode hosp days', 'Is schizophrneia Dx', 'Is affective type',
    'Admitted to EIS', 'DUP days', 'Any life events', 'Is smoker', 'Comorbidity OCD',
    'Comorbidity anxiety', 'Comorbidity depression',
    'Comorbidity substance abuse', 'Comorbidity others',
    'Positive symptoms (mean)', 'Negative symptoms (mean)',
    'Depressive symptoms (mean)', 'Medication adherence (mean)', 'SOFAS (mean)',
    'Positive symptoms (MSSD)', 'Negative symptoms (MSSD)',
    'Depressive symptoms (MSSD)', 'Medication adherence (MSSD)', 'SOFAS (MSSD)',
    'SA (sum)', 'NSSI (sum)', 'Substance abuse (sum)', 'A&E visit (sum)',
    'OPD visit (sum)', 'Hospitalization (sum)', 'Default (sum)', 'Relapse (sum)',
    'DDD (mean)', 'Anticholinergic (sum)', 'Antidepressant (sum)',
    'Benzodiazepine (sum)', 'Mood stabilizer (sum)', 'Polypharmacy (sum)',
    'ECT (sum)'
]

fi_df = reduce(
    lambda left, right: pd.merge(left, right, on=['Variable'], how='right'),
    [fi_baseline_df, fi_12m_df, fi_24m_df, fi_36m_df]
)
fi_df['Variable'] = labels

reordered_labels = [
    'Age at first service contact', 'Sex', 'Years of education', 'Any life events',
    'Is smoker', 'Is schizophrneia Dx', 'Is affective type', 'Age of illness onset',
    'Admitted to EIS', 'First episode duation', 'First episode hosp days', 'DUP days',
    'DUP SA history', 'DUP NSSI history', 'Comorbidity anxiety',
    'Comorbidity depression', 'Comorbidity OCD', 'Comorbidity substance abuse',
    'Comorbidity others', 'Positive symptoms (mean)', 'Negative symptoms (mean)',
    'Depressive symptoms (mean)', 'Medication adherence (mean)', 'SOFAS (mean)',
    'Positive symptoms (MSSD)', 'Negative symptoms (MSSD)',
    'Depressive symptoms (MSSD)', 'Medication adherence (MSSD)', 'SOFAS (MSSD)',
    'SA (sum)', 'NSSI (sum)', 'Substance abuse (sum)', 'A&E visit (sum)',
    'OPD visit (sum)', 'Hospitalization (sum)', 'Default (sum)', 'Relapse (sum)',
    'DDD (mean)', 'Anticholinergic (sum)', 'Antidepressant (sum)',
    'Benzodiazepine (sum)', 'Mood stabilizer (sum)', 'Polypharmacy (sum)',
    'ECT (sum)'
]
fi_reordered_df = fi_df.set_index('Variable').loc[reordered_labels]

fig = plt.figure(figsize=(8, 10))
ax1 = plt.subplot2grid((4, 49), (0, 4), colspan=44, rowspan=4)
ax2 = plt.subplot2grid((4, 49), (0, 0), colspan=4, rowspan=44)

sns.heatmap(fi_reordered_df, ax=ax1, cmap="viridis",
            yticklabels=reordered_labels, cbar_kws={'shrink': .4})
ax1.set_ylabel('')
ax1.tick_params(axis='y', which='major', pad=25, length=0)
sns.heatmap(pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                          2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4,
                          4, 5, 5, 5, 5, 5, 5, 5]),
            ax=ax2, cbar=False, xticklabels=False, yticklabels=False,
            cmap=['lightgrey', 'moccasin', 'lavenderblush', 'cadetblue', 'khaki'])
plt.subplots_adjust(left=0.5)
ax1.set_xticklabels(
    ax1.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

fig.savefig(Path('outputs', 'figs', 'feature_importance.pdf'))
