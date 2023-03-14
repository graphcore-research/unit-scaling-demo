# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# %%
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import pandas as pd


def load_raw_stats(base_filename: str, n_seeds=3):
    raw_stats = {}
    for scaling_type in ['us', 'reg']:
        raw_stats[scaling_type] = {}
        for seed in range(1, n_seeds + 1):
            stats_a = np.load(f"{base_filename}/{scaling_type}_{seed}_a.npy", allow_pickle=True)
            stats_b = np.load(f"{base_filename}/{scaling_type}_{seed}_b.npy", allow_pickle=True)
            raw_stats[scaling_type][seed] = [
                {**step_dict_a, **step_dict_b}
                for step_dict_a, step_dict_b
                in zip(stats_a, stats_b)
            ]
    return raw_stats


def clean_up_name(name: str) -> str:
    return name.split(')|')[1]


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
        stat_name = stat_name.replace('layer_11|mlm_head', 'mlm_head')  # fix weird logging issue

        clean_name = clean_up_name(stat_name)
        if 'act' in stat_name:
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


def to_series(stats) -> pd.Series:
    df_dict = {
        (scaling_type, seed, step, tensor_type, variable): std
        for scaling_type, seed_dict in stats.items()
        for seed, step_dict in seed_dict.items()
        for step, tensor_type_dict in enumerate(step_dict)
        for tensor_type, variable_dict in tensor_type_dict.items()
        for variable, std in variable_dict.items()
    }
    return pd.Series(df_dict)


def load_stats(base_filename: str, n_seeds) -> pd.Series:
    """
    Returns:
        pd.Series: multi-index of (scaling_type, seed, step, tensor_type, var_name), with columns=bins
    """
    raw_stats = load_raw_stats(base_filename, n_seeds)
    clean_stats = {
        scaling_type: {
            seed: [
                clean_up_stats(step_stats)
                for step_stats in seed_data
            ]
            for seed, seed_data in scaling_type_data.items()
        }
        for scaling_type, scaling_type_data in raw_stats.items()
    }
    return to_series(clean_stats)


def to_dataframe(ratio_types: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    df_dict = {
        ratio_type: layer_stds
        for ratio_type, layer_stds in ratio_types.items()
    }
    return pd.DataFrame(df_dict, index=list(range(9)))


def get_var(data, var_name, scale_us=1.0, accum_df=None):
    var = data[data['var'] == var_name]['value']
    var['us'] = var['us'] * scale_us
    var_df = var.groupby(var.index).agg(['mean', 'std']).apply(lambda x: f"{x[0]:.3} ({x[1]:.3})", axis=1).rename(var_name).to_frame().transpose()
    return pd.concat([accum_df, var_df])


def get_var_mean_layers(data, var_name, accum_df=None):
    layer_df = None
    for layer_idx in range(9):
        layer_var_name = f"layer_{layer_idx}|{var_name}"
        var = data[data['var'] == layer_var_name]['value']
        var = var.rename(layer_var_name).to_frame().transpose()
        layer_df = pd.concat([layer_df, var])
    layer_df_T = layer_df.transpose()
    out_df = layer_df_T.groupby(layer_df_T.index).agg(['mean', 'std']).apply(lambda x: f"{x[0]:.3} ({x[1]:.3})", axis=1).rename(var_name).to_frame().transpose()
    return pd.concat([accum_df, out_df])


def get_var_ratio(data, a_name, b_name, out_name, a_us_scale=1.0, accum_df=None):
    a = data[data['var'] == a_name]['value']
    a['us'] = a['us'] * a_us_scale
    a_df = a.rename(a_name).to_frame().transpose().reset_index().drop('index', axis=1)
    b = data[data['var'] == b_name]['value']
    b_df = b.rename(b_name).to_frame().transpose().reset_index().drop('index', axis=1)
    ratio_df = (a_df / b_df).transpose()
    ratio_df.columns = ['tmp']
    out_df = ratio_df.groupby(ratio_df.index).agg(['mean', 'std']).apply(lambda x: f"{x[0]:.3} ({x[1]:.3})", axis=1).rename(out_name).to_frame().transpose()
    return pd.concat([accum_df, out_df])



def analyse_ratios(data):
    data = data.to_frame().reset_index()
    data.columns = ['scaling', 'seed', 'type', 'var', 'value']
    data = data.set_index('scaling')

    acts = data[data['type'] == 'acts'].drop('type', axis=1)
    grads = data[data['type'] == 'grad_xs'].drop('type', axis=1)

    analysis_df = get_var_mean_layers(acts, "attention|pre-residual_ratio")
    analysis_df = get_var_mean_layers(acts, "ffn|pre-residual_ratio", accum_df=analysis_df)
    analysis_df = get_var_mean_layers(grads, "attention|pre-residual_ratio_bwd", accum_df=analysis_df)
    analysis_df = get_var_mean_layers(grads, "ffn|pre-residual_ratio_bwd", accum_df=analysis_df)
    analysis_df = get_var_mean_layers(acts, "attention|softmax_in", accum_df=analysis_df)
    analysis_df = get_var(acts, "mlm_head|logits", accum_df=analysis_df)
    analysis_df = get_var(acts, "nsp_head|logits", accum_df=analysis_df)
    analysis_df = get_var_ratio(grads, "embeddings|word", "mlm_head|word", "mlm enc/dec", accum_df=analysis_df)
    analysis_df = get_var_ratio(grads, "nsp_head|gather", "layer_8|mlm_head|gather", "nsp/mlm", a_us_scale=4.688, accum_df=analysis_df)
    analysis_df = get_var(acts, "nsp_head|gather", accum_df=analysis_df)

    return analysis_df


# %%
if __name__ == "__main__":
    base_filename: str = "data/base/unit_scaling/ratios/triple/deg_5"

    stats = load_stats(base_filename, n_seeds=1)
    init_stats = stats.loc[(slice(None), slice(None), 0, slice(None), slice(None))]
    out = analyse_ratios(init_stats)
