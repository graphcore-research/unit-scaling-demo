# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pandas as pd
from random import random
import wandb

api = wandb.Api(overrides={'base_url': 'https://wandb.sourcevertex.net'})


def get_runs():
    tags = ['final', f'{random()}']
    return api.runs(f"research/unit-scaling", filters={"tags": {"$in": tags}})


def extract_details(run):
    tags = run.tags
    details = {
        t.split(':')[0]: t.split(':')[1] for t in tags if ":" in t
    }
    details['init_checkpoint'] = run.config['init_checkpoint']
    summary = run.summary
    details['em'] = summary['exact'] if 'exact' in summary else summary['exact_match']
    details['f1'] = summary['f1']
    return details


def get_run_details():
    runs = get_runs()
    return pd.DataFrame([pd.Series(extract_details(run)) for run in runs])


def get_run_scores():
    details = get_run_details()
    group_keys = ['format', 'size', 'version', 'init_checkpoint']
    ckpt_means = details.groupby(group_keys).mean()

    means = ckpt_means.groupby(group_keys[:-1]).mean()
    stds = ckpt_means.groupby(group_keys[:-1]).std()
    stds.columns = [f'{c}_std' for c in stds.columns]

    means['em'] = means['em'].apply(lambda v: f'{v:.2f}')
    means['f1'] = means['f1'].apply(lambda v: f'{v:.2f}')
    stds['em_std'] = stds['em_std'].apply(lambda v: f'(±{v:.2f})')
    stds['f1_std'] = stds['f1_std'].apply(lambda v: f'(±{v:.2f})')
    joint = means.join(stds)

    em_scores = joint['em'] + ' ' + joint['em_std']
    f1_scores = joint['f1'] + ' ' + joint['f1_std']
    return em_scores.to_frame('em').join(f1_scores.to_frame('f1'))


if __name__ == "__main__":
    print(get_run_scores())
