"""Tools for creating publication figures."""

import matplotlib.pyplot as plt
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

    # delete legend (redundant with the x-tick labels)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    # fix ordering of plot elements
    plt.setp(ax.lines, zorder=100, linewidth=1.25)
    plt.setp(ax.collections, zorder=100)
