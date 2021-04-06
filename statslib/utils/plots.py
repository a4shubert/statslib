
import matplotlib.pyplot as _plt
import math as _math
import itertools
def plot_to_grid(df, columns=None, title='', plots_per_row=3, legend_on_a_side=False):
    if columns is None:
        columns = df.columns
    L = plots_per_row
    K = _math.ceil(len(columns) / L)
    if K==1:
        K+=1
    i = j = 0
    fig, axs = _plt.subplots(K, L, figsize=(15, 15))
    my_colors = itertools.cycle(get_standard_colors())
    for curve in columns:
        df[curve].plot(ax=axs[i, j], color=next(my_colors))
        axs[i, j].legend([curve])
        j += 1
        if j % L == 0:
            i += 1
            j = 0
    _plt.suptitle(title)
    if legend_on_a_side:
        _plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _plt.tight_layout(h_pad=-2)
    _plt.show()



def get_standard_colors():
    return _plt.rcParams['axes.prop_cycle'].by_key()['color']



def color():
    import itertools
    itr = itertools.cycle(get_standard_colors())
    return itr
