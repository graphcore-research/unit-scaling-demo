"""Code listings source and quick tests."""

import numpy as np
from torch import autograd, tensor, randn, nn, matmul
from torch.nn import Parameter, LayerNorm
from torch.nn.functional import gelu


# appendix.py

class ScaledGrad(autograd.Function):
  @staticmethod
  def forward(ctx, X, alpha, beta):
    ctx.save_for_backward(
      tensor(beta, dtype=X.dtype))
    return alpha * X

  @staticmethod
  def backward(ctx, grad_Y):
    beta, = ctx.saved_tensors
    return beta * grad_Y, None, None

def scaled(X, alpha=1, beta=1):
  # Forward: Y = X * alpha
  # Backward: grad_X = grad_Y * beta
  return ScaledGrad.apply(X, alpha, beta)

def scaled_matmul(
  A, B, constrain_A=True, constrain_B=True,
):
  (m, k), (_, n) = A.shape, B.shape
  alpha = k ** -(1/2)
  beta_A = n ** -(1/2)
  beta_B = m ** -(1/2)

  if constrain_A and constrain_B:
    alpha = beta_A = beta_B = \
      (alpha * beta_A * beta_B) ** (1/3)
  elif constrain_A:
    alpha = beta_A = (alpha * beta_A) ** (1/2)
  elif constrain_B:
    alpha = beta_B = (alpha * beta_B) ** (1/2)

  A = scaled(A, beta=beta_A)
  B = scaled(B, beta=beta_B)
  return scaled(matmul(A, B), alpha)

def scaled_gelu(X):
  return 1.5876 * gelu(X)

class ScaledLayerNorm(nn.LayerNorm):
  def forward(self, x):
    beta = (
      np.prod(self.normalized_shape)
      / x.nelement()
    ) ** 0.5
    return nn.functional.layer_norm(
        x,
        self.normalized_shape,
        scaled(self.weight, beta=beta),
        scaled(self.bias, beta=beta),
        self.eps,
    )


# projection.py

# +first 3 lines of `def scaled``

def scaled_projection(X, W):
  (b, _), (m, n) = X.shape, W.shape
  alpha = beta_X = (m * n) ** -(1/4)
  beta_W = b ** -(1/2)
  X = scaled(X, beta=beta_X)
  W = scaled(W, beta=beta_W)
  return scaled(matmul(X, W), alpha)


# ffn.py

class FFN(nn.Module):
  def __init__(self, d, h):
    super().__init__()
    self.norm = LayerNorm(d)
    sigma = (d * h) ** -(1/4)
    self.W_1 = Parameter(
      randn(d, h) * sigma)
    self.W_2 = Parameter(
      randn(h, d) * sigma)

  def forward(self, X):
    Z = self.norm(X)
    Z = matmul(Z, self.W_1)
    Z = gelu(Z)
    Z = matmul(Z, self.W_2)
    return X + Z


# ffn_scaled.py

class ScaledFFN(nn.Module):
  def __init__(self, d, h, tau):
    super().__init__()
    self.norm = ScaledLayerNorm(d)
    self.W1 = Parameter(randn(d, h))
    self.W2 = Parameter(randn(h, d))
    self.tau = tau

  def forward(self, X):
    a = (1 - self.tau) ** (1/2)
    b = self.tau ** (1/2)
    Z = self.norm(scaled(X, beta=b))
    Z = scaled_projection(Z, self.W1)
    Z = scaled_gelu(Z)
    Z = scaled_projection(Z, self.W2)
    return X * a + scaled(Z, b)


# Tests

import torch as T
import pytest

def test_scaled_projection():
  T.manual_seed(1000)
  a = T.randn(100, 300, requires_grad=True)
  b = T.randn(300, 800, requires_grad=True)
  dy = T.randn(a.shape[0], b.shape[1])

  y = scaled_projection(a, b)
  y.backward(dy)
  T.testing.assert_close(float(T.std(y) * T.std(a.grad)), 1.0, atol=1e-2, rtol=0)
  T.testing.assert_close(float(T.std(b.grad)), 1.0, atol=1e-2, rtol=0)

  a.grad = b.grad = None
  y = scaled_matmul(a, b)
  y.backward(dy)
  T.testing.assert_close(float(T.std(y) * T.std(a.grad) * T.std(b.grad)), 1.0, atol=1e-2, rtol=0)

def test_scaled_gelu():
  T.manual_seed(2000)
  x = T.randn(20000, requires_grad=True)
  y = scaled_gelu(x)
  y.backward(T.randn_like(y))
  T.testing.assert_close(float(T.std(y) * T.std(x.grad)), 1.0, atol=1e-2, rtol=0)

@pytest.mark.parametrize("scaled", [False, True])
def test_ffn(scaled):
  T.manual_seed(3000)
  x = T.randn(1024, 256, requires_grad=True)
  dy = T.randn_like(x)
  layer = (
    ScaledFFN(x.shape[1], 4*x.shape[1], tau=0.4)
    if scaled else
    FFN(x.shape[1], 4*x.shape[1])
  )
  y = layer(x)
  assert y.shape == x.shape
  assert T.std(y) < 2

  if scaled:
    y.backward(dy)
    T.testing.assert_close(float(T.std(x.grad)), 1.0, atol=0.1, rtol=0)
    T.testing.assert_close(float(T.std(y)), 1.0, atol=0.1, rtol=0)
    for k, v in layer.named_parameters():
      std = float(T.std(v.grad))
      T.testing.assert_close(std, 1.0, atol=0.5, rtol=0,
        msg=f"std({k}.grad) is {std:.1f}, not close to unit")
