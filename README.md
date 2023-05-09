# Unit Scaling demo

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/unit-scaling-demo?container=graphcore%2Fpytorch-jupyter%3A3.1.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fpytorch-notebook%2Funit-scaling-notebook.ipynb)

Code for the paper: [Unit Scaling: Out-of-the-Box Low-Precision Training](https://arxiv.org/abs/2303.11257).

We'd like weights, activations & gradients all to be unit-variance at initialisation. To achieve this, we will introduce separate scaling factors for activations in the forwards pass and for gradients in the backwards pass.

This repository contains our experimentation code for experiments on character-level language modelling, and a demo notebook.

**Overview:**
 - Technique - Unit Scaling
 - Task - Character Language Modelling
 - Dataset - [WikiText-103 (raw)](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
 - Framework - [TF2/Keras](https://www.tensorflow.org/), [Poplar SDK](https://www.graphcore.ai/products/poplar)
 - Logging - [WandB](https://wandb.ai)

**Structure:**
 - [run_experiment.py](run_experiment.py) - configuration & entry point for a single experiment
 - [run_sweep.py](run_sweep.py) - sweep logic & configuration
 - [scmm/](scmm) - core Python package and baseline implementation
   - [scmm/uscale/](scmm/uscale) - unit scaling implementation
   - [scmm/pedal/](scmm/pedal) - platform-specific adapters
 - [dev](dev) - development task launch script (tests, lint, etc)
 - [Dataset.ipynb](Dataset.ipynb) - script used to generate the vocabulary from WikiText-103 (raw)
 - [pytorch-notebook/unit-scaling-notebook.ipynb](pytorch-notebook/unit-scaling-notebook.ipynb)

**See also:**
 - [pytorch-notebook/unit-scaling-notebook.ipynb](pytorch-notebook/unit-scaling-notebook.ipynb) - standalone PyTorch demo
 - [branch:2023-01-paper](https://github.com/graphcore-research/unit-scaling-demo/tree/2023-01-paper) - additional supporting materials for the paper

## Usage

This code has been tested on [Poplar SDK](https://www.graphcore.ai/downloads) 3.1.0+1205.

```bash
python3 -m venv .venv
# Append to .venv/bin/activate:
# source PATH/TO/POPLAR_SDK/enable
source .venv/bin/activate
pip install wheel
pip install $POPLAR_SDK_ENABLED/../tensorflow-2.6.3+gc3.1.0+246224+2b7af067dae+amd_znver1-cp38-cp38-linux_x86_64.whl
pip install $POPLAR_SDK_ENABLED/../keras-2.6.0+gc3.1.0+246230+88e2debf-py2.py3-none-any.whl
pip install -r requirements.txt

python run_experiment.py
```

## To reproduce

Our test result sweeps are described by `run_sweep.py`. By default this assumes the data is under /home/research-datasets/wikitext103_raw (`train.txt`, `valid.txt`, `test.txt`) and that the user is logged into WandB.

```bash
python run_sweep.py
```

## References & license

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under a MIT license (see [LICENSE](LICENSE)).

Our dependencies are:

| Component | About | License |
| --- | --- | --- |
| WandB | Weights and Biases client library ([website](https://wandb.ai/)), for optional logging to wandb servers | MIT |

We also use additional Python dependencies for development/testing (see [requirements-dev.txt](requirements-dev.txt)).

The WikiText-103 dataset is licenced under the Creative Commons Attribution-ShareAlike License.
