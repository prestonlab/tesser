"""Tools for creating publication figures."""

from pkg_resources import resource_filename
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


def set_style(style_path=None):
    """Set default plot style."""
    if style_path is None:
        style_path = resource_filename('tesser', 'data/figures.mplstyle')
    plt.style.use(style_path)


def get_induct_colors():
    """Induction task colors."""
    dark = sns.blend_palette(
        [
            np.array([20, 125, 201]) / 256,
            np.array([0, 166, 161]) / 256,
            np.array([15, 175, 75]) / 256,
        ],
        n_colors=3,
    )
    light = sns.blend_palette(
        [
            np.array([171, 212, 237]) / 256,
            np.array([172, 235, 242]) / 256,
            np.array([165, 232, 177]) / 256,
        ],
        n_colors=3,
    )
    return {'dark': dark, 'light': light}


def plot_group_mat(
    data, x='dim1', y='dim2', hue=None, palette=None, color=None, ax=None
):
    """Plot grouping task object locations."""
    if ax is None:
        ax = plt.gca()

    data = data.copy()
    node_type = data['object_type'].map({'central': 1, 'boundary': 2})
    data['label'] = (data['community'] - 1) * 2 + node_type
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
        'grey',
        'd_purple',
        'l_purple',
        'd_red',
        'l_red',
        'd_green',
        'l_green',
    ]
    cmap = colors.ListedColormap([node_colors[color] for color in color_order])
    ax.imshow(mat, cmap=cmap)
    ax.set(
        xticks=np.arange(0.5, 18.5),
        yticks=np.arange(0.5, 10.5),
        xticklabels=[],
        yticklabels=[],
    )
    ax.grid(True)
    ax.tick_params('both', bottom=False, left=False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
