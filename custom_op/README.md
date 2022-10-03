# Using the 'fake cast' FP8 op

```bash
ninja
env PYTHONPATH=$PYTHONPATH:build pytest quantisefp8.py
```

In code:

```python
from quantisefp8 import quantise_fp8

x: tf.Tensor = ...

y = quantise_fp8(x, bias=0, format="1.4.3")

assert y.dtype == x.dtype, "dtype has not changed, but data has been rounded"
```
