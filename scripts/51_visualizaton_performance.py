# Environment ------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.Python import metrics, viz


# Data I/O ---------------------------------------------------------------------
te_baseline_y = np.load(
    Path('outputs', 'cache', 'automl-data_baseline-desc_y.npy'))
te_baseline_probas = np.load(
    Path('outputs', 'cache', 'automl-data_baseline-desc_final_probas.npy'))
te_baseline_top10_y = np.load(
    Path('outputs', 'cache', 'automl-data_baseline_top10-desc_y.npy'))
te_baseline_top10_probas = np.load(
    Path('outputs', 'cache', 'automl-data_baseline_top10-desc_final_probas.npy'))
te_baseline_top15_y = np.load(
    Path('outputs', 'cache', 'automl-data_baseline_top15-desc_y.npy'))
te_baseline_top15_probas = np.load(
    Path('outputs', 'cache', 'automl-data_baseline_top15-desc_final_probas.npy'))
te_baseline_top20_y = np.load(
    Path('outputs', 'cache', 'automl-data_baseline_top20-desc_y.npy'))
te_baseline_top20_probas = np.load(
    Path('outputs', 'cache', 'automl-data_baseline_top20-desc_final_probas.npy'))

te_12m_y = np.load(Path('outputs', 'cache', 'automl-data_12m-desc_y.npy'))
te_12m_probas = np.load(
    Path('outputs', 'cache', 'automl-data_12m-desc_final_probas.npy'))
te_12m_top10_y = np.load(
    Path('outputs', 'cache', 'automl-data_12m_top10-desc_y.npy'))
te_12m_top10_probas = np.load(
    Path('outputs', 'cache', 'automl-data_12m_top10-desc_final_probas.npy'))
te_12m_top15_y = np.load(
    Path('outputs', 'cache', 'automl-data_12m_top15-desc_y.npy'))
te_12m_top15_probas = np.load(
    Path('outputs', 'cache', 'automl-data_12m_top15-desc_final_probas.npy'))
te_12m_top20_y = np.load(
    Path('outputs', 'cache', 'automl-data_12m_top20-desc_y.npy'))
te_12m_top20_probas = np.load(
    Path('outputs', 'cache', 'automl-data_12m_top20-desc_final_probas.npy'))

te_24m_y = np.load(Path('outputs', 'cache', 'automl-data_24m-desc_y.npy'))
te_24m_probas = np.load(
    Path('outputs', 'cache', 'automl-data_24m-desc_final_probas.npy'))
te_24m_top10_y = np.load(
    Path('outputs', 'cache', 'automl-data_24m_top10-desc_y.npy'))
te_24m_top10_probas = np.load(
    Path('outputs', 'cache', 'automl-data_24m_top10-desc_final_probas.npy'))
te_24m_top15_y = np.load(
    Path('outputs', 'cache', 'automl-data_24m_top15-desc_y.npy'))
te_24m_top15_probas = np.load(
    Path('outputs', 'cache', 'automl-data_24m_top15-desc_final_probas.npy'))
te_24m_top20_y = np.load(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_y.npy'))
te_24m_top20_probas = np.load(
    Path('outputs', 'cache', 'automl-data_24m_top20-desc_final_probas.npy'))

te_36m_y = np.load(Path('outputs', 'cache', 'automl-data_36m-desc_y.npy'))
te_36m_probas = np.load(
    Path('outputs', 'cache', 'automl-data_36m-desc_final_probas.npy'))
te_36m_top10_y = np.load(
    Path('outputs', 'cache', 'automl-data_36m_top10-desc_y.npy'))
te_36m_top10_probas = np.load(
    Path('outputs', 'cache', 'automl-data_36m_top10-desc_final_probas.npy'))
te_36m_top15_y = np.load(
    Path('outputs', 'cache', 'automl-data_36m_top15-desc_y.npy'))
te_36m_top15_probas = np.load(
    Path('outputs', 'cache', 'automl-data_36m_top15-desc_final_probas.npy'))
te_36m_top20_y = np.load(
    Path('outputs', 'cache', 'automl-data_36m_top20-desc_y.npy'))
te_36m_top20_probas = np.load(
    Path('outputs', 'cache', 'automl-data_36m_top20-desc_final_probas.npy'))


# AUROC ------------------------------------------------------------------------
auroc_baseline = metrics.AUC(
    te_baseline_y, te_baseline_probas).calculate_mean()
auroc_12m = metrics.AUC(te_12m_y, te_12m_probas).calculate_mean()
auroc_24m = metrics.AUC(te_24m_y, te_24m_probas).calculate_mean()
auroc_36m = metrics.AUC(te_36m_y, te_36m_probas).calculate_mean()

auroc_df = pd.DataFrame({
    'Baseline': auroc_baseline,
    '12-month': auroc_12m,
    '24-month': auroc_24m,
    '36-month': auroc_36m,
})

# print out the full data results
auroc_df.mean(axis=0)
auroc_df.std(axis=0)
st.t.interval(
    alpha=0.95,
    df=99,
    loc=auroc_df.mean(axis=0),
    scale=auroc_df.sem(axis=0)
)

# save csv for R to plot the violin plots
auroc_df.to_csv(
    Path('outputs', 'tables', 'performance-type_auroc.csv'), index=False
)

# top 10 features
auroc_top10_baseline = metrics.AUC(
    te_baseline_top10_y, te_baseline_top10_probas).calculate_mean()
auroc_top10_12m = metrics.AUC(
    te_12m_top10_y, te_12m_top10_probas).calculate_mean()
auroc_top10_24m = metrics.AUC(
    te_24m_top10_y, te_24m_top10_probas).calculate_mean()
auroc_top10_36m = metrics.AUC(
    te_36m_top10_y, te_36m_top10_probas).calculate_mean()

auroc_top10_df = pd.DataFrame({
    'Baseline': auroc_top10_baseline,
    '12-month': auroc_top10_12m,
    '24-month': auroc_top10_24m,
    '36-month': auroc_top10_36m,
})

# save csv for R to plot the violin plots
auroc_top10_df.to_csv(
    Path('outputs', 'tables', 'performance-type_auroc-desc_top10.csv'), index=False
)

# top 15 features
auroc_top15_baseline = metrics.AUC(
    te_baseline_top15_y, te_baseline_top15_probas).calculate_mean()
auroc_top15_12m = metrics.AUC(
    te_12m_top15_y, te_12m_top15_probas).calculate_mean()
auroc_top15_24m = metrics.AUC(
    te_24m_top15_y, te_24m_top15_probas).calculate_mean()
auroc_top15_36m = metrics.AUC(
    te_36m_top15_y, te_36m_top15_probas).calculate_mean()

auroc_top15_df = pd.DataFrame({
    'Baseline': auroc_top15_baseline,
    '12-month': auroc_top15_12m,
    '24-month': auroc_top15_24m,
    '36-month': auroc_top15_36m,
})

# save csv for R to plot the violin plots
auroc_top15_df.to_csv(
    Path('outputs', 'tables', 'performance-type_auroc-desc_top15.csv'), index=False
)

# top 20 features
auroc_top20_baseline = metrics.AUC(
    te_baseline_top20_y, te_baseline_top20_probas).calculate_mean()
auroc_top20_12m = metrics.AUC(
    te_12m_top20_y, te_12m_top20_probas).calculate_mean()
auroc_top20_24m = metrics.AUC(
    te_24m_top20_y, te_24m_top20_probas).calculate_mean()
auroc_top20_36m = metrics.AUC(
    te_36m_top20_y, te_36m_top20_probas).calculate_mean()

auroc_top20_df = pd.DataFrame({
    'Baseline': auroc_top20_baseline,
    '12-month': auroc_top20_12m,
    '24-month': auroc_top20_24m,
    '36-month': auroc_top20_36m,
})

# save csv for R to plot the violin plots
auroc_top20_df.to_csv(
    Path('outputs', 'tables', 'performance-type_auroc-desc_top20.csv'), index=False
)

# Brier Score ------------------------------------------------------------------
bs_baseline = metrics.BrierScore(
    te_baseline_y, te_baseline_probas).calculate_mean()
bs_12m = metrics.BrierScore(te_12m_y, te_12m_probas).calculate_mean()
bs_24m = metrics.BrierScore(te_24m_y, te_24m_probas).calculate_mean()
bs_36m = metrics.BrierScore(te_36m_y, te_36m_probas).calculate_mean()

bs_df = pd.DataFrame({
    'Baseline': bs_baseline,
    '12-month': bs_12m,
    '24-month': bs_24m,
    '36-month': bs_36m,
})
# save csv for R to plot the violin plots
bs_df.to_csv(
    Path('outputs', 'tables', 'performance-type_brierscore.csv'), index=False
)

# print out the full data results
bs_df.mean(axis=0)
bs_df.std(axis=0)
st.t.interval(
    alpha=0.95,
    df=99,
    loc=bs_df.mean(axis=0),
    scale=bs_df.sem(axis=0)
)

# top 10 features
bs_top10_baseline = metrics.BrierScore(
    te_baseline_top10_y, te_baseline_top10_probas).calculate_mean()
bs_top10_12m = metrics.BrierScore(
    te_12m_top10_y, te_12m_top10_probas).calculate_mean()
bs_top10_24m = metrics.BrierScore(
    te_24m_top10_y, te_24m_top10_probas).calculate_mean()
bs_top10_36m = metrics.BrierScore(
    te_36m_top10_y, te_36m_top10_probas).calculate_mean()

bs_top10_df = pd.DataFrame({
    'Baseline': bs_top10_baseline,
    '12-month': bs_top10_12m,
    '24-month': bs_top10_24m,
    '36-month': bs_top10_36m,
})
# save csv for R to plot the violin plots
bs_top10_df.to_csv(
    Path('outputs', 'tables', 'performance-type_brierscore-desc_top10.csv'), index=False
)

# top 15 features
bs_top15_baseline = metrics.BrierScore(
    te_baseline_top15_y, te_baseline_top15_probas).calculate_mean()
bs_top15_12m = metrics.BrierScore(
    te_12m_top15_y, te_12m_top15_probas).calculate_mean()
bs_top15_24m = metrics.BrierScore(
    te_24m_top15_y, te_24m_top15_probas).calculate_mean()
bs_top15_36m = metrics.BrierScore(
    te_36m_top15_y, te_36m_top15_probas).calculate_mean()

bs_top15_df = pd.DataFrame({
    'Baseline': bs_top15_baseline,
    '12-month': bs_top15_12m,
    '24-month': bs_top15_24m,
    '36-month': bs_top15_36m,
})
# save csv for R to plot the violin plots
bs_top15_df.to_csv(
    Path('outputs', 'tables', 'performance-type_brierscore-desc_top15.csv'), index=False
)

# top 20 features
bs_top20_baseline = metrics.BrierScore(
    te_baseline_top20_y, te_baseline_top20_probas).calculate_mean()
bs_top20_12m = metrics.BrierScore(
    te_12m_top20_y, te_12m_top20_probas).calculate_mean()
bs_top20_24m = metrics.BrierScore(
    te_24m_top20_y, te_24m_top20_probas).calculate_mean()
bs_top20_36m = metrics.BrierScore(
    te_36m_top20_y, te_36m_top20_probas).calculate_mean()

bs_top20_df = pd.DataFrame({
    'Baseline': bs_top20_baseline,
    '12-month': bs_top20_12m,
    '24-month': bs_top20_24m,
    '36-month': bs_top20_36m,
})
# save csv for R to plot the violin plots
bs_top20_df.to_csv(
    Path('outputs', 'tables', 'performance-type_brierscore-desc_top20.csv'), index=False
)


# Decision Curve ---------------------------------------------------------------
thresh_group = np.arange(0, 1, 0.05)

net_benefit_baseline_all = viz.decisioncurve(
    thresh_group, te_baseline_y.tolist(), te_baseline_probas.tolist()
)._calculate_net_benefit_all()
net_benefit_baseline_model = viz.decisioncurve(
    thresh_group, te_baseline_y.tolist(), te_baseline_probas.tolist()
)._calculate_net_benefit_model()
net_benefit_12m_all = viz.decisioncurve(
    thresh_group, te_12m_y.tolist(), te_12m_probas.tolist()
)._calculate_net_benefit_all()
net_benefit_12m_model = viz.decisioncurve(
    thresh_group, te_12m_y.tolist(), te_12m_probas.tolist()
)._calculate_net_benefit_model()
net_benefit_24m_all = viz.decisioncurve(
    thresh_group, te_24m_y.tolist(), te_24m_probas.tolist()
)._calculate_net_benefit_all()
net_benefit_24m_model = viz.decisioncurve(
    thresh_group, te_24m_y.tolist(), te_24m_probas.tolist()
)._calculate_net_benefit_model()
net_benefit_36m_all = viz.decisioncurve(
    thresh_group, te_36m_y.tolist(), te_36m_probas.tolist()
)._calculate_net_benefit_all()
net_benefit_36m_model = viz.decisioncurve(
    thresh_group, te_36m_y.tolist(), te_36m_probas.tolist()
)._calculate_net_benefit_model()

average_net_benefit_all = np.mean(
    [
        net_benefit_baseline_all, net_benefit_12m_all,
        net_benefit_24m_all, net_benefit_36m_all
    ], axis=0
)

fig, ax = plt.subplots(figsize=(8, 6))
# plot
ax.plot(thresh_group, net_benefit_baseline_model, linewidth=2,
        color='cadetblue', label='Baseline')
ax.plot(thresh_group, net_benefit_12m_model, linewidth=2,
        color='goldenrod', label='12-month')
ax.plot(thresh_group, net_benefit_24m_model, linewidth=2,
        color='indigo', label='24-month')
ax.plot(thresh_group, net_benefit_36m_model, linewidth=2,
        color='crimson', label='36-month')
ax.plot(thresh_group, net_benefit_baseline_all, linewidth=2,
        color='cadetblue', linestyle=':', label='All (Baseline)')
ax.plot(thresh_group, net_benefit_12m_all, linewidth=2,
        color='goldenrod', linestyle=':', label='All (12-month)')
ax.plot(thresh_group, net_benefit_24m_all, linewidth=2,
        color='indigo', linestyle=':', label='All (24-month)')
ax.plot(thresh_group, net_benefit_36m_all, linewidth=2,
        color='crimson', linestyle=':', label='All (36-month)')
ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='None')
# Fill, Shows that the model is better than treat all and treat none The good part
#y2 = np.maximum(net_benefit_all, 0)
#y1 = np.maximum(net_benefit_model, y2)
#ax.fill_between(self.thresh_group, y1, y2, color='crimson', alpha=0.2)
# Figure Configuration, Beautify the details
ax.set_xlim(0, 0.4)
# adjustify the y axis limitation
ax.set_ylim(-0.01, average_net_benefit_all.max() + 0.05)
ax.set_xlabel(
    xlabel='Threshold Probability',
    fontdict={
        'family': 'Times New Roman', 'fontsize': 20
    }
)
ax.set_ylabel(
    ylabel='Net Benefit',
    fontdict={
        'family': 'Times New Roman', 'fontsize': 20
    }
)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.grid('major')
ax.spines['right'].set_color((0.8, 0.8, 0.8))
ax.spines['top'].set_color((0.8, 0.8, 0.8))
ax.legend(loc='upper right')
ax.legend(prop={'size': 16})
fig.savefig(Path('outputs', 'figs', 'decisioncurve.pdf'))

# Performance at Risk Cutoff ----------------------------------------------------
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Baseline
performance_riskcutoff_baseline = metrics.perf_riskcut(
    te_baseline_y, te_baseline_probas, thresholds
).to_df()
performance_riskcutoff_baseline.to_csv(
    Path('outputs', 'tables', 'performance_riskcutoff_baseline.csv'),
    index=False
)

performance_riskcutoff_12m = metrics.perf_riskcut(
    te_12m_y, te_12m_probas, thresholds
).to_df()
performance_riskcutoff_12m.to_csv(
    Path('outputs', 'tables', 'performance_riskcutoff_12m.csv'),
    index=False
)

performance_riskcutoff_24m = metrics.perf_riskcut(
    te_24m_y, te_24m_probas, thresholds
).to_df()
performance_riskcutoff_24m.to_csv(
    Path('outputs', 'tables', 'performance_riskcutoff_24m.csv'),
    index=False
)

performance_riskcutoff_36m = metrics.perf_riskcut(
    te_36m_y, te_36m_probas, thresholds
).to_df()
performance_riskcutoff_36m.to_csv(
    Path('outputs', 'tables', 'performance_riskcutoff_36m.csv'),
    index=False
)
