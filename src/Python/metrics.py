from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
import numpy as np
import pandas as pd


class common_evaluation:

    def __init__(self, true, pred, probas):
        self.true = true
        self.pred = pred
        self.probas = probas


class AUC:

    def __init__(self, true, probas):
        self.true = true
        self.probas = probas

    def calculate_mean(self):
        auroc_list = list()
        for i in range(len(self.true)):
            auroc_list.append(roc_auc_score(self.true[i], self.probas[i]))

        return auroc_list


class BrierScore:

    def __init__(self, true, probas):
        self.true = true
        self.probas = probas

    def calculate_mean(self):
        breierscore_list = list()
        for i in range(len(self.true)):
            breierscore_list.append(
                brier_score_loss(self.true[i], self.probas[i])
            )

        return breierscore_list


class perf_riskcut:

    def __init__(self, true, probas, thresholds):
        self.true = true
        self.probas = probas
        self.thresholds = thresholds

    def to_df(self):
        # empty list
        sens_mean = list()
        sens_sd = list()
        spec_mean = list()
        spec_sd = list()
        ppv_mean = list()
        ppv_sd = list()
        npv_mean = list()
        npv_sd = list()
        nb_mean = list()
        nb_sd = list()
        snb_mean = list()
        snb_sd = list()

        for thresh in self.thresholds:

            sens_list = list()
            spec_list = list()
            ppv_list = list()
            npv_list = list()
            nb_list = list()
            snb_list = list()

            for i in range(100):
                pred_label = self.probas[i] > thresh
                tn, fp, fn, tp = confusion_matrix(
                    self.true[i], pred_label).ravel()

                sens_list.append(tp / (tp + fn))
                spec_list.append(tn / (tn + fp))
                ppv_list.append(tp / (tp + fp))
                npv_list.append(tn / (tn + fn))

                n = len(self.true[i])
                prevalence = 0.22  # based on meta-analysis
                net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
                nb_list.append(net_benefit)
                snb = net_benefit / prevalence
                snb_list.append(snb)

            sens_mean.append(np.nanmean(sens_list))
            sens_sd.append(np.nanstd(sens_list))
            spec_mean.append(np.nanmean(spec_list))
            spec_sd.append(np.nanstd(spec_list))
            ppv_mean.append(np.nanmean(ppv_list))
            ppv_sd.append(np.nanstd(ppv_list))
            npv_mean.append(np.nanmean(npv_list))
            npv_sd.append(np.nanstd(npv_list))
            nb_mean.append(np.nanmean(nb_list))
            nb_sd.append(np.nanstd(nb_list))
            snb_mean.append(np.nanmean(snb_list))
            snb_sd.append(np.nanstd(snb_list))

        # creature pdf
        df = pd.DataFrame({
            'Risk Cutoff': self.thresholds,
            'Sensitivity (Mean)': sens_mean,
            'Sensitivity (SD)': sens_sd,
            'Specificity (Mean)': spec_mean,
            'Specificity (SD)': spec_sd,
            'PPV (Mean)': ppv_mean,
            'PPV (SD)': ppv_sd,
            'NPV (Mean)': npv_mean,
            'NPV (SD)': npv_sd,
            'NB (Mean)': nb_mean,
            'NB (SD)': nb_sd,
            'sNB (Mean)': snb_mean,
            'sNB (SD)': snb_sd,
        })

        return df
