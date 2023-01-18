"""Code listings source and quick tests."""

import numpy as np
import torch
from torch.nn import Parameter, LayerNorm
from torch import tensor, pi, relu, randn, nn


# Appendix - scaling

class ScaledGrad(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, sy, sg):
    ctx.save_for_backward(
      tensor(sg, dtype=x.dtype))
    return sy * x

  @staticmethod
  def backward(ctx, gy):
    sg, = ctx.saved_tensors
    return sg * gy, None, None

def scaled(x, sy=1, sg=1):
  "Compute `y = x * sy` forward,\
  and `gx = gy * sg` backward."
  return ScaledGrad.apply(x, sy, sg)


# Appendix - ops

def scaled_matmul(
  a, b,
  constrain_a = True,
  constrain_b = True,
):
  (m, k), (_, n) = a.shape, b.shape
  sy = k ** -(1/2)
  sa = n ** -(1/2)
  sb = m ** -(1/2)

  if constrain_a and constrain_b:
    sy = sa = sb = (sy*sa*sb) ** (1/3)
  elif constrain_a:
    sy = sa = (sy*sa) ** (1/2)
  elif constrain_b:
    sy = sb = (sy*sb) ** (1/2)

  a = scaled(a, sg=sa)
  b = scaled(b, sg=sb)
  return scaled(a @ b, sy=sy)

def scaled_relu(x):
  s = ((1 - 1/pi) / 4) ** -(1/4)
  return scaled(relu(x), sy=s, sg=s)

class ScaledLayerNorm(nn.LayerNorm):
  def forward(self, x):
    sg = (
      np.prod(self.normalized_shape)
      / x.nelement()
    ) ** 0.5
    return nn.functional.layer_norm(
        x,
        self.normalized_shape,
        scaled(self.weight, sg=sg),
        scaled(self.bias, sg=sg),
        self.eps,
    )


# Body - op

def scaled_projection(x, W):
  (B, _), (dx, dy) = x.shape, W.shape
  sy = (dx * dy) ** -(1/4)
  sW = B ** -(1/2)
  x = scaled(x, sg=sy)
  W = scaled(W, sg=sW)
  return scaled(x @ W, sy=sy)


# Body - layer comparison

class FFN(nn.Module):
  def __init__(self, dx, dFFN):
    super().__init__()
    s_init = (dx * dFFN)**-(1/4)
    self.up = Parameter(
      randn(dx, dFFN) * s_init)
    self.down = Parameter(
      randn(dFFN, dx) * s_init)
    self.norm = LayerNorm(dx)

  def forward(self, x):
    z = self.norm(x)
    z = z @ self.up
    z = relu(z)
    z = z @ self.down
    return x + z

class ScaledFFN(nn.Module):
  def __init__(self, dx, dFFN, tau):
    super().__init__()
    self.up = Parameter(randn(dx, dFFN))
    self.down = Parameter(randn(dFFN, dx))
    self.norm = ScaledLayerNorm(dx)
    self.tau = tau

  def forward(self, x):
    sx = (1-self.tau)**(1/2)
    sz = (self.tau)**(1/2)
    z = self.norm(scaled(x, sg=sz))
    z = scaled_projection(z, self.up)
    z = scaled_relu(z)
    z = scaled_projection(z, self.down)
    return x * sx + scaled(z, sy=sz)


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

def test_scaled_relu():
  T.manual_seed(2000)
  x = T.randn(20000, requires_grad=True)
  y = scaled_relu(x)
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
