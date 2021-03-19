"""Tools for creating publication figures."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


def plot_swarm_error(
    data, x=None, y=None, hue=None, dark=None, light=None, ax=None, dodge=False,
    capsize=.425
):
    """Make a bar plot with individual points and error bars."""
    if dark is None:
        dark = 'ch:rot=-.5, light=.7, dark=.3, gamma=.6'
    if light is None:
        light = 'ch:rot=-.5, light=.7, dark=.3, gamma=.2'

    if ax is None:
        ax = plt.gca()

    # plot individual points
    sns.swarmplot(
        data=data.reset_index(), x=x, y=y, hue=hue, palette=dark, size=3,
        linewidth=0.1, edgecolor='k', ax=ax, zorder=3, dodge=dodge
    )

    # plot error bars for the mean
    sns.barplot(
        data=data.reset_index(), x=x, y=y, hue=hue, ax=ax, dodge=dodge, color='k',
        palette=light, errwidth=.8, capsize=capsize, edgecolor='k', linewidth=.75,
        errcolor='k'
    )

    # remove overall xlabel and increase size of x-tick labels
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize='large')

    # fix ordering of plot elements
    plt.setp(ax.lines, zorder=100, linewidth=1.25, label=None)
    plt.setp(ax.collections, zorder=100, label=None)

    # delete legend (redundant with the x-tick labels)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
        if hue is not None:
            # refresh the legend to remove the swarm points
            ax.legend()


def plot_group_mat(
    data, x='row', y='col', hue=None, palette=None, color=None, ax=None
):
    """Plot grouping task object locations."""
    if ax is None:
        ax = plt.gca()

    data['label'] = data['community'] * 2 - data['node_type']
    mat_shape = (11, 19)
    mat = np.zeros(mat_shape, dtype=int)
    row = data[x].to_numpy()
    col = data[y].to_numpy()
    mat[row, col] = data['label'].to_numpy()

    node_colors = {
        "d_purple": '#7e1e9c',
        "l_purple": '#bf77f6',
        "d_green": '#15b01a',
        "l_green": '#96f97b',
        "d_red": '#e50000',
        "l_red": '#ff474c',
        "grey": (0.95, 0.95, 0.95),
    }
    color_order = [
        'grey', 'd_purple', 'l_purple', 'd_red', 'l_red', 'd_green', 'l_green'
    ]
    cmap = colors.ListedColormap([node_colors[color] for color in color_order])
    ax.imshow(mat, cmap=cmap)
    ax.set(
        xticks=np.arange(0.5, 18.5), yticks=np.arange(0.5, 10.5), xticklabels=[],
        yticklabels=[]
    )
    ax.grid(True)
    ax.tick_params('both', bottom=False, left=False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
