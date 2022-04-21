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

## Literature review

| Paper | Notes |
| --- | --- |
| Understanding the difficulty of training deep feedforward neural networks [[pdf](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)] | Introduced "Glorot" or "Xavier" initialisation, very widespread. |
| Tensor Programs V [[arxiv](https://arxiv.org/abs/2203.03466)] | A hyperparameter reparametrisation for initialisation and learning rate based on network width, helping hyperparameters to transfer across scale. |
| Weight Normalization [[arxiv](https://arxiv.org/abs/1602.07868)] | A differentiable reparameterisation for network parameters based on explicit normalisation and scaling. |
| Centered Weight Normalization [[pdf](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Centered_Weight_Normalization_ICCV_2017_paper.pdf)] | Weight normalization with centering (zero mean). |
| Weight Standardization [[arxiv](https://arxiv.org/abs/1903.10520)] | Centered Weight Normalization without a learnable length. |
| Attention is all you need (Transformer) [[arxiv](https://arxiv.org/abs/1706.03762)] | Scaled dot product attention. |
| Block-Normalized Gradient [[arxiv](https://arxiv.org/abs/1707.04822)] | Normalised weight update gradients (like LAMB). |
| On the difficulty of training Recurrent Neural Networks [[arxiv](https://arxiv.org/abs/1211.5063)] | Gradient clipping and gradient norm regularisation. |
| Path-SGD [[arxiv](https://arxiv.org/abs/1506.02617)] | Path regularization and a rescaling-invariant learning rule. |
