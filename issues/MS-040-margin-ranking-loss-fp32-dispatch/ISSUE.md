# `ops.margin_ranking_loss` computes in float32 for float64 inputs

**Component:** MindSpore 2.9.0 (CPU), `mindspore.ops.margin_ranking_loss` (the `MarginRankingLoss`
kernel, all reductions).
**Severity:** silent precision loss - with float64 inputs the returned tensor reports `Float64` but
its value is bit-for-bit the **float32** computation widened back:

```python
ops.margin_ranking_loss(x_f64, y_f64, t_f64, margin=0.5, reduction='mean')
  == ops.margin_ranking_loss(x_f32, y_f32, t_f32, margin=0.5, reduction='mean').astype(float64)
```

Against a float64 NumPy reference (`mean(max(0, -t*(x-y) + margin))`) the relative error is
`~2e-8 .. 5e-8` at scales 1, 1e3 and 1e6, while the same-family `ops.smooth_l1_loss` and
`ops.hinge_embedding_loss` are **float64-exact** on identical data (not bit-identical to their own
float32 calls).
**Status:** present in 2.9.0 on CPU, in PyNative and `ms.jit` (both lanes return the same
fp32-quality float64 value).
**Found with:** the wave-390/391 Rondo float32-dispatch oracle (a float64 call bit-identical to the
float32 call), the same class behind MS-035/MS-038/MS-039; certified reproducer `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 12 checks).

---

## Summary

| witness (4x5, values ~3, margin 0.5) | `ops.margin_ranking_loss` float64 | NumPy float64 reference | relative error | bit-identical to the fp32 call |
|---|---|---|---|---|
| scale 1 | `1.02305722` | `1.0230571720809711` | `4.92e-08` | **True** |
| scale 1e3 | `808.7388916` | `808.7389053617352` | `1.70e-08` | **True** |
| scale 1e6 | `808539.125` | `808539.1053617352` | `2.43e-08` | **True** |
| scale 1e3, `reduction='sum'` | `16174.7783203125` | `16174.778107234704` | `1.32e-08` | **True** |
| control: `ops.smooth_l1_loss` (same data/scale) | - | - | **`0.0`** | False (float64 kernel) |
| control: `ops.hinge_embedding_loss(margin=1)` | - | - | **`0.0`** | False (float64 kernel) |

`ms.jit(margin_ranking_loss)` returns exactly the eager float64 value, so both execution lanes share
the fp32 kernel.

## Root cause

`margin_ranking_loss` is a composite wrapper: it builds `max(0, -t * (x - y) + margin)` from other
ops and reduces it, and the composed kernel is registered/specialised for float32, so the float64
inputs are down-cast on entry while the output dtype is taken from the Python-level dtype inference
of the wrapper (`Float64`). The result is a tensor whose dtype contract says float64 but whose
mantissa is float32.

## Why this is not a precision-hardware artifact

* The failure is **exact bit-identity** with the fp32 call after widening - not a small numeric
  difference. On CPU float64 is implemented in software and would be exact (as the sibling losses
  demonstrate on the very same witness); the errors above (`~1e-8`) are exactly float32 rounding.
* Sibling losses in the same module (`smooth_l1_loss`, `hinge_embedding_loss`) are fp64-exact with
  the same inputs, dtypes and reductions, so this is a per-op dispatch gap, not a device limitation.
* The dtype contract is documented: `ops.margin_ranking_loss` supports `float16`/`float32`/`float64`
  inputs (the API doc lists `float64` for all three input tensors).

## Minimal reproduction

```python
import numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE)

xv = np.array([[0.1, 0.3, -2.0]]); yv = np.array([[0.9, -1.1, 0.5]]); tv = np.array([[1., -1., 1.]])
x, y, t = (ms.Tensor(a, ms.float64) for a in (xv, yv, tv))

o64 = ops.margin_ranking_loss(x, y, t, margin=0.5, reduction='mean')
o32 = ops.margin_ranking_loss(*(ms.Tensor(a, ms.float32) for a in (xv, yv, tv)),
                              margin=0.5, reduction='mean')
print(o64.asnumpy(), o64.dtype)                          # 2.066666603088379 Float64
print(o32.asnumpy().astype(np.float64))                  # 2.066666603088379  <- same fp32 value
# float64 NumPy reference: 2.066666666666667  (relative error 3.1e-08)
print(np.mean(np.maximum(0., -tv * (xv - yv) + 0.5)))
```

## Related observation (weaker, not part of the certified claims)

`ops.cosine_embedding_loss` shows the same direction but not the exact bit-identity: on the same
witness its float64 value `0.4205920398235321` differs from the widened float32 call
`0.4205920696258545` by `3.0e-08`, i.e. it is only fp32-accurate while reporting `Float64`. Kept as
a note; the claim in this issue is the exact dispatch identity for `margin_ranking_loss`.

## Novelty check

Searched the MindSpore issue tracker / docs for `margin_ranking_loss` float64 precision and
float64 dispatch reports; the only matching item is a **documentation** issue about the
`nn.MarginRankingLoss` docstring lacking context (Gitee `I7W2JY`, August activity), which is
unrelated to the numeric behaviour. No existing report of the float64-dispatches-to-float32
behaviour for this op was found.
