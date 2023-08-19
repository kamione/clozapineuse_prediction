from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class decisioncurve:

    def __init__(self, thresh_group, y_label, y_probas):
        self.thresh_group = thresh_group
        self.y_probas = y_probas
        self.y_label = y_label

    def _calculate_net_benefit_model(self):

        if isinstance(self.y_probas, list):
            net_benefit_model_list = list()

            for i in range(len(self.y_probas)):
                net_benefit_model = list()
                for thresh in self.thresh_group:
                    y_pred_label = self.y_probas[i] > thresh
                    y_label = self.y_label[i]
                    tn, fp, fn, tp = confusion_matrix(
                        y_label, y_pred_label).ravel()
                    n = len(y_label)
                    net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
                    net_benefit_model = np.append(
                        net_benefit_model, net_benefit)
                net_benefit_model_list.append(net_benefit_model)
            return np.mean(net_benefit_model_list, axis=0)

        else:
            net_benefit_model = np.array([])
            for thresh in self.thresh_group:
                y_pred_label = self.y_probas > thresh
                tn, fp, fn, tp = confusion_matrix(
                    self.y_label, y_pred_label).ravel()
                n = len(self.y_label)
                net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
                net_benefit_model = np.append(net_benefit_model, net_benefit)
            return net_benefit_model

    def _calculate_net_benefit_all(self):

        if isinstance(self.y_label, list):
            net_benefit_all_list = list()
            for i in range(len(self.y_label)):
                net_benefit_all = np.array([])
                tn, fp, fn, tp = confusion_matrix(
                    self.y_label[i], self.y_label[i]).ravel()
                total = tp + tn
                for thresh in self.thresh_group:
                    net_benefit = (tp / total) - (tn / total) * \
                        (thresh / (1 - thresh))
                    net_benefit_all = np.append(net_benefit_all, net_benefit)
                net_benefit_all_list.append(net_benefit_all)
            return np.mean(net_benefit_all_list, axis=0)

        else:
            net_benefit_all = np.array([])
            tn, fp, fn, tp = confusion_matrix(
                self.y_label, self.y_label).ravel()
            total = tp + tn
            for thresh in self.thresh_group:
                net_benefit = (tp / total) - (tn / total) * \
                    (thresh / (1 - thresh))
                net_benefit_all = np.append(net_benefit_all, net_benefit)
            return net_benefit_all

    def plot(self):
        net_benefit_model = self._calculate_net_benefit_model()
        net_benefit_all = self._calculate_net_benefit_all()

        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot
        ax.plot(self.thresh_group, net_benefit_model,
                color='crimson', label='Model')
        ax.plot(self.thresh_group, net_benefit_all, color='black', label='All')
        ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='None')
        # Fill, Shows that the model is better than treat all and treat none The good part
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        ax.fill_between(self.thresh_group, y1, y2, color='crimson', alpha=0.2)
        # Figure Configuration, Beautify the details
        ax.set_xlim(0, 0.4)
        # adjustify the y axis limitation
        ax.set_ylim(-0.01, net_benefit_model.max() + 0.05)
        ax.set_xlabel(
            xlabel='Threshold Probability',
            fontdict={
                'family': 'Times New Roman', 'fontsize': 15
            }
        )
        ax.set_ylabel(
            ylabel='Net Benefit',
            fontdict={
                'family': 'Times New Roman', 'fontsize': 15
            }
        )
        ax.grid('major')
        ax.spines['right'].set_color((0.8, 0.8, 0.8))
        ax.spines['top'].set_color((0.8, 0.8, 0.8))
        ax.legend(loc='upper right')
        return fig


class calibrationcurve:

    def __init__(self, label, probas):
        self.target = label
        self.probas = probas

    def __abc(self):
        print(1)

    def plot(self):
        print(1)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.target, self.probas, n_bins=10, normalize=True
        )

        fig, ax = plt.subplots(1, figsize=(12, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-')
        plt.plot([0, 1], [0, 1], '--', color='gray')

        sns.despine(left=True, bottom=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.title("Calibration Curve", fontsize=20)

        return fig


class featureimportance:

    def __init__(self, importance, label):
        self.importance = importance
        self.label = label

    def tomedian(self):

        # get median
        median_feature_importance = np.median(self.importance, axis=0)

        # create a pandas dataframe
        df = pd.Series(
            median_feature_importance,
            index=self.label
        )

        return df

    def torank(self):

        from scipy.stats import rankdata
        no_columns = self.importance.shape[0]
        no_features = self.importance.shape[1]

        tmp = np.zeros(shape=(no_columns, no_features))

        for i in range(no_columns):
            tmp[[i]] = rankdata(self.importance[[i]], method='ordinal')

        mean_feature_rank = np.mean(tmp, axis=0)

        # create a pandas dataframe
        df = pd.Series(
            mean_feature_rank,
            index=self.label
        )

        return df

    def plot(self):

        df = self.todf()
        fig, ax = plt.subplots(figsize=(6, 8))
        df.plot.barh(ax=ax, color='#1C1C1C')  # SUMI
        ax.invert_yaxis()
        ax.grid(axis='x', color='0.95')
        ax.set_xlabel("Feature Importance")
        ax.spines['right'].set_color((0.8, 0.8, 0.8))
        ax.spines['top'].set_color((0.8, 0.8, 0.8))
        fig.tight_layout()

        return fig
