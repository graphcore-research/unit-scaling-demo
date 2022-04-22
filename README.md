# Scale matmuls not initialisation

We'd like weights, activations & gradients all to be unit-normal at initialisation. To achieve this, we will relax the requirement to calculate correct gradients by introducing separate scaling factors for activations in the forwards pass and for gradients in the backwards pass.

## Usage

```bash
popenv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
./dev

python run_experiment.py
```
