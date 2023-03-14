# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# %%
import itertools
from collections import OrderedDict
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def load_raw_stats(file_a: str, file_b: str) -> List[Dict[str, np.ndarray]]:
    stats_a = np.load(file_a, allow_pickle=True)
    stats_b = np.load(file_b, allow_pickle=True)
    return [
        {**step_dict_a, **step_dict_b}
        for step_dict_a, step_dict_b
        in zip(stats_a, stats_b)
    ]


def clean_up_name(name: str) -> str:
    return name.split(')|')[1] \
               .replace('|', '/') \
               .replace('_', ' ') \
               .replace('attention/softmax', 'attention/mask,softmax') \
               .replace('mask,softmax dropout', 'softmax dropout') \
               .replace('/attention/', '\nattention/') \
               .replace('/ffn/', '\nFFN/')


def clean_up_stats(stats: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns:
        Dict[str, List[Tuple[str, np.ndarray]]]: A dictionary
        with keys: 'acts', 'grad_xs', 'grad_ws', 'weights', where each contains a
        {var name: histogram} dict, in order of occurrence in the network.
    """
    clean_stats: Dict[str, Dict[str, np.ndarray]] = {
        'acts': OrderedDict(),
        'grad_xs': OrderedDict(),
        'grad_ws': OrderedDict(),
        'weights': OrderedDict()
    }
    for stat_name, histogram in sorted(stats.items()):
        stat_name = stat_name.replace('layer_8|mlm_head', 'mlm_head')  # fix weird logging issue
        # For the sake of clarity, drop some layers
        if "layer" in stat_name \
                and not ("layer_0" in stat_name
                         or "layer_7" in stat_name
                         or "layer_8" in stat_name
                         or "embeddings|layer_norm" in stat_name):
            continue
        if stat_name[-2:] in ["|q", "|k", "|v"]:
            continue
        if ("|mean" in stat_name or "softmax_in" in stat_name
                or "|logits" in stat_name or ("|word" in stat_name and "head" in stat_name)
                or "|mlm_input_tensor" in stat_name or "|nsp_input_tensor" in stat_name):
            continue

        clean_name = clean_up_name(stat_name)
        if 'act' in stat_name:
        #     if 'mlm_head' in stat_name:
        #         if 'log_softmax' in stat_name or 'idx' in stat_name:
        #             histogram = np.roll(histogram, -4)  # correct for mis-reporting
            clean_stats['acts'][clean_name] = histogram
        elif 'grad_x' in stat_name:
            clean_stats['grad_xs'][clean_name] = histogram
        elif 'grad_w' in stat_name:
            clean_stats['grad_ws'][clean_name] = histogram
        elif 'weight' in stat_name:
            clean_stats['weights'][clean_name] = histogram
        else:
            assert False, stat_name
    return clean_stats


def to_dataframe(stats: List[Dict[str, Dict[str, np.ndarray]]]) -> pd.DataFrame:
    df_dict = {
        (step, tensor_type, variable): histogram
        for step, tensor_type_dict in enumerate(stats)
        for tensor_type, variable_dict in tensor_type_dict.items()
        for variable, histogram in variable_dict.items()
    }
    bin_edges = [float('-inf'), -24] + list(range(-14, 15)) + [float('inf')]
    return pd.DataFrame(df_dict, index=bin_edges)


def load_stats(file_a: str, file_b: str) -> pd.DataFrame:
    """
    Returns:
        pd.DataFrame: multi-index of (step, tensor_type, var_name), with columns=bins
    """
    raw_stats = load_raw_stats(file_a, file_b)
    clean_stats = [clean_up_stats(step_stats) for step_stats in raw_stats]
    return to_dataframe(clean_stats)


def heatmap_2_level(
        data: pd.DataFrame,
        axis: Axes,
        x_labels: List[str],
        color: str,
        stat_type: str,
        show_labels: bool) -> None:
    data = data.applymap(lambda x: float("nan") if x == 0.0 else x)
    data = data.transpose()
    data.columns = data.columns.astype(str)

    cdict = {
        col: np.vstack([
            np.multiply(np.array([0.5, 1, 1]), data),
            np.array([1.0, 0 if col != 'alpha' else 1, 0 if col != 'alpha' else 1])])
                for col, data in plt.colormaps[color]._segmentdata.items()
    }
    cmap = matplotlib.colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

    cbar_kws = {'aspect': 80, 'location': 'bottom', 'pad': 0.05}
    heatmap = sns.heatmap(data, cmap=cmap, cbar_kws=cbar_kws, ax=axis, vmin=0.001, vmax=1)
    print("len(x_labels)", len(x_labels))
    heatmap.set_xticks(np.arange(0, len(x_labels)), x_labels, rotation=0)  # type: ignore
    axis.tick_params('x', width=0.5, bottom=True)

    def transpose(xs):
        """Transposes a list of lists (see https://stackoverflow.com/a/6473724)"""
        return list(map(list, itertools.zip_longest(*xs, fillvalue=None)))

    def measure_labels(xs):
        """Given a list of labels where some repeat consecutively,
        returns the fractional positions at which the labels change.
        Note that for matplotlib we require the list to be initially reversed.
        """
        d = []
        latest = None
        for i, x in enumerate(reversed(xs)):
            if latest != x:
                latest = x
                d.append((x, float(i / len(xs))))
        return transpose(d)

    def mean_pairs(xs):
        """Given a list of floats, returns a list of pairwise means."""
        ys = []
        for i in range(0, len(xs) - 1):
            ys.append(np.mean([xs[i], xs[i + 1]]))
        return ys

    y_labels = data.transpose().columns
    split_y_labels = [label.split('/', 1) for label in y_labels]
    outer_y_labels, inner_y_labels = transpose(split_y_labels)
    outer_y_unique, outer_y_positions = measure_labels(outer_y_labels)
    outer_y_positions.append(1.0)
    outer_y_centres = mean_pairs(outer_y_positions)
    inner_y_labels = [label.replace('/', ':') for label in inner_y_labels]

    axis.title.set_text(stat_type if stat_type != "acts" else "activations")

    # Insert elipses
    # i = 0 if stat_type not in ['acts', 'grad_xs'] else 1
    # elipsis_1_pos = outer_y_positions[2]
    elipsis_2_pos = outer_y_positions[6]
    # outer_y_centres.insert(2, elipsis_1_pos)
    outer_y_centres.insert(7, elipsis_2_pos)
    outer_y_centres = [o * 1.0001 for o in outer_y_centres]  # hack required, unsure why
    # outer_y_unique.insert(2, '• • •')
    outer_y_unique.insert(7, '• • • ')

    if show_labels:
        x_label_text = 'log2(|value|):\n\n% of values per bin:'
        axis.set_xlabel(x_label_text, loc='left', fontsize=11)
        axis.xaxis.set_label_coords(-0.28, -0.015)

        # y-axis inner labels
        axis.set_yticks(np.arange(len(inner_y_labels)) + 0.5)
        axis.set_yticklabels(inner_y_labels, fontsize=10)
        axis.yaxis.set_tick_params(length=0)

        axis2 = axis.twinx()

        # vertical line
        axis2.spines["left"].set_position(("axes", -0.32))
        axis2.spines["left"].set_color("black")
        axis2.spines["left"].set_linewidth(0.5)
        axis2.tick_params('both', length=0, width=0, which='minor')
        axis2.tick_params('both', direction='in', which='major', width=0.5)
        axis2.yaxis.set_ticks_position("left")
        axis2.yaxis.set_label_position("left")

        # text
        axis2.set_yticks(outer_y_positions)
        axis2.yaxis.set_major_formatter(ticker.NullFormatter())
        axis2.yaxis.set_minor_locator(ticker.FixedLocator(outer_y_centres))
        axis2.yaxis.set_minor_formatter(ticker.FixedFormatter(outer_y_unique))
    else:
        axis.set_yticks([])
        axis.set_yticklabels([])
        axis.yaxis.set_tick_params(length=0)
        axis2 = axis.twinx()
        axis2.set_yticks(outer_y_positions)
        axis2.set_yticklabels([])
        axis2.yaxis.set_tick_params(length=0)


def cross_section_heatmap(init_stats: pd.DataFrame) -> None:
    sns.set_theme()
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams["font.family"] = "serif"
    _, axs = plt.subplots(2, 2, figsize=(15, 24))
    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.07, hspace=-0.06)
    bin_edges = ['-inf   ', '\n-24'] + \
        [str(edge) if edge % 2 == 0 else "" for edge in range(-14, 15 + 1)] + \
        ['  inf']
    color_map = [['Oranges', 'Blues'], ['Greens', 'Reds']]
    stat_types = [['acts', 'grad_xs'], ['grad_ws', 'weights']]

    for y in range(2):
        for x in range(2):
            heatmap_2_level(
                init_stats[stat_types[y][x]],  # type: ignore
                axs[y, x],  # type: ignore
                bin_edges,
                color_map[y][x],
                stat_type=stat_types[y][x],
                show_labels=(x == 0)
            )


# %%
if __name__ == "__main__":
    act_w_filename: str = "data/paper/init_test/reg_a_L.npy"
    grad_x_grad_w_filename: str = "data/paper/init_test/reg_b_L.npy"

    stats = load_stats(act_w_filename, grad_x_grad_w_filename)
    init_stats: pd.DataFrame = stats[703]  # type: ignore  (warmup=140, final=703)
    cross_section_heatmap(init_stats)
    plt.savefig('reg_end_hist.pdf', pad_inches=1.0, bbox_inches='tight')

# How to get a "time slice": `x.loc[:, (slice(None), 'acts', 'embeddings/segment')]`

# DONT FORGET ROLL

# %%
