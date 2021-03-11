"""Tools for creating publication figures."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_swarm_error(data, x=None, y=None, hue=None, dark=None, light=None, ax=None):
    """Make a bar plot with individual points and error bars."""
    if dark is None:
        dark = 'ch:rot=-.5, light=.7, dark=.3, gamma=.6'
    if light is None:
        light = 'ch:rot=-.5, light=.7, dark=.3, gamma=.2'

    if ax is None:
        ax = plt.gca()

    # plot means as bars with light colors (Seaborn's bar plot gives bars that don't
    # line up with the other plots)
    n_bin = len(data.groupby(x))
    light_pal = sns.color_palette(light, n_colors=n_bin)
    for i, (name, bins) in enumerate(data.groupby(x)):
        ax.bar(
            i, bins.mean(), color=light_pal[i], edgecolor='k', linewidth=.75, width=.6,
            zorder=0
        )

    # plot individual points
    sns.swarmplot(
        data=data.reset_index(), x=x, y=y, hue=hue, palette=dark, size=3,
        linewidth=0.1, edgecolor='k', ax=ax, zorder=3
    )

    # plot error bars for the mean
    sns.pointplot(
        data=data.reset_index(), x=x, y=y, hue=hue, ax=ax, join=False, color='k',
        markers='.', errwidth=1, capsize=.425, scale=.1
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
