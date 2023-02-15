# Unit scaling (CharLM)

We'd like weights, activations & gradients all to be unit-normal at initialisation. To achieve this, we will introduce separate scaling factors for activations in the forwards pass and for gradients in the backwards pass.

## Usage

```bash
python3 -m venv .venv
# Append to .venv/bin/activate:
# source PATH/TO/POPLAR_SDK/enable.sh
source .venv/bin/activate
pip install -r requirements.txt

python run_experiment.py
```

## To reproduce

Our test result sweeps are described by `run_sweep.py`. By default this assumes the data is under /home/research-datasets/wikitext103_raw (train.txt, valid.txt, test.txt) and that the user is logged into WandB.

```bash
python3 run_sweep.py
```
