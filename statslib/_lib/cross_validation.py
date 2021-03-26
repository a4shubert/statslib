import math
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


class CrossValidation:
    def __init__(self, gm, splits, metric):
        self.splits = splits
        self.metric = metric
        self.gm = gm

    def run(self, to_plot=True):
        metric_list = []
        L = 2
        K = math.ceil(len(self.splits) / L)
        if K==1:
            K+=1
        i = j = 0
        if to_plot:
            fig, axs = plt.subplots(K, L, figsize=(15, 15))
        for train_index, test_index in self.splits:
            self.gm.fit(train_index)
            self.gm.forecast(test_index)
            y_hat = self.gm.y_hat
            y0 = self.gm.y0
            y_train = self.gm.DM.dm.y.iloc[train_index]
            y = self.gm.DM.dm.y.iloc[test_index]
            metric_val = self.metric(y, y_hat)
            metric_list.append(metric_val)
            y_hat.at[y0.index.max()] = y0[y0.index.max()]
            y_hat.sort_index(inplace=True)
            y.at[y0.index.max()] = y0[y0.index.max()]
            y.sort_index(inplace=True)
            if to_plot:
                axs[i, j].set_title(r'$\varepsilon_{CV}=$' + f'{metric_val:.4f}')
                y_train.plot(ax=axs[i, j], c='#988ED5')
                y_hat.plot(ax=axs[i, j], c='#8EBA42')
                y.plot(ax=axs[i, j], c='#E24A33')
                axs[i, j].legend(['$y_t$', '$\hat{y}$', '$y_{true}$'])
            j += 1
            if j % L == 0:
                i += 1
                j = 0
        CV_error = np.mean(metric_list)
        CV_std = np.std(metric_list)
        print(f'CV_error: {CV_error:.2f}, CV_std: {CV_std:.2f} | ({CV_error-CV_std:.2f}  {CV_error+CV_std:.2f})')
        if to_plot:
            plt.tight_layout()
            plt.show()
        return CV_error, CV_std