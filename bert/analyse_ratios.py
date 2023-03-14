# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# %%
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import pandas as pd


def load_raw_stats(file_a: str, file_b: str) -> List[Dict[str, np.ndarray]]:
    stats_a = np.load(file_a, allow_pickle=True)
    stats_b = np.load(file_b, allow_pickle=True)
    return [
        {**step_dict_a, **step_dict_b}
        for step_dict_a, step_dict_b
        in zip(stats_a, stats_b)
    ]


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


def to_series(stats: List[Dict[str, Dict[str, np.ndarray]]]) -> pd.Series:
    df_dict = {
        (step, tensor_type, variable): std
        for step, tensor_type_dict in enumerate(stats)
        for tensor_type, variable_dict in tensor_type_dict.items()
        for variable, std in variable_dict.items()
    }
    return pd.Series(df_dict)


def load_stats(file_a: str, file_b: str) -> pd.Series:
    """
    Returns:
        pd.DataFrame: multi-index of (step, tensor_type, var_name), with columns=bins
    """
    raw_stats = load_raw_stats(file_a, file_b)
    clean_stats = [clean_up_stats(step_stats) for step_stats in raw_stats]
    return to_series(clean_stats)


def to_dataframe(ratio_types: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    df_dict = {
        ratio_type: layer_stds
        for ratio_type, layer_stds in ratio_types.items()
    }
    return pd.DataFrame(df_dict, index=list(range(9)))
    # return pd.DataFrame(df_dict, index=list(range(3)))


def analyse_ratios(data: pd.Series) -> pd.DataFrame:
    acts, grads = data['acts'], data['grad_xs']
    ratios = {
        'sa_fwd':  [],
        'ffn_fwd': [],
        'sa_bwd':  [],
        'ffn_bwd': [],
    }
    for layer_idx in range(9):
    # for layer_idx in range(3):
        # --- SA fwd ---

        sa_fwd_ratio = acts[f"layer_{layer_idx}|attention|pre-residual_ratio"]
        ratios['sa_fwd'].append(sa_fwd_ratio)

        # --- FFN fwd ---

        ffn_fwd_ratio = acts[f"layer_{layer_idx}|ffn|pre-residual_ratio"]
        ratios['ffn_fwd'].append(ffn_fwd_ratio)

        # --- SA bwd ---

        sa_bwd_ratio = grads[f"layer_{layer_idx}|attention|pre-residual_ratio_bwd"]
        ratios['sa_bwd'].append(sa_bwd_ratio)

        # --- FFN bwd ---

        ffn_bwd_ratio = grads[f"layer_{layer_idx}|ffn|pre-residual_ratio_bwd"]
        ratios['ffn_bwd'].append(ffn_bwd_ratio)

    return to_dataframe(ratios)


# %%
if __name__ == "__main__":
    act_w_filename: str = "data/base/unit_scaling/ratios/fix_grad_ws/tensor_stats_a.npy"
    grad_x_grad_w_filename: str = "data/base/unit_scaling/ratios/fix_grad_ws/tensor_stats_b.npy"
    # dev_filename: str = "data/dev/unit_scaling/ratios/both_scaled/tensor_stats_seed_2.npy"

    stats = load_stats(act_w_filename, grad_x_grad_w_filename)
    # stats = load_stats(dev_filename, dev_filename)
    init_stats: pd.Series = stats[0]  # type: ignore  (warmup=140, final=703)
    ratios = analyse_ratios(init_stats)
    print(ratios)
    print(ratios.mean(axis=0))
