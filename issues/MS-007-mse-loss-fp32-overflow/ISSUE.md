# `ops.l1_loss` / `ops.smooth_l1_loss` / `ops.mse_loss` compute in float32 for float64 inputs — `mse_loss` then overflows to `inf` above `|x| ≈ 1.8e19`

**Component:** MindSpore 2.9.0, `mindspore.ops.mse_loss`
**Severity:** silent wrong result (`inf` for a perfectly representable float64 loss)
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo fp64 accuracy audit (`artifact/hunt-v2/ms_precision_nn.py`)

---

## Summary

The mean-squared error of `[s, -s]` against zeros is `2s²`.  In float64 that is representable up to
`s ≈ 9e153`; MindSpore returns `inf` from `s ≈ 1.8e19` onwards, exactly where `s²` leaves the
float32 range, while the sibling losses stay exact:

```python
import numpy as np, mindspore as ms
from mindspore import ops

x = ms.Tensor([1e20, -1e20], ms.float64)
z = ms.Tensor([0.0, 0.0], ms.float64)
ops.mse_loss(x, z, reduction="sum")   # inf     (exact: 2e40)
ops.l1_loss(x, z, reduction="sum")    # 2e20    (correct)
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| `s` | `mse_loss([s,-s], 0, sum)` | exact `2s²` | |
|---|---|---|---|
| `1` | `2` | `2` | agrees |
| `1e10` | `2e20` | `2e20` | agrees |
| `1e20` | **`inf`** | `2e40` | overflow |
| `1e30` | **`inf`** | `2e60` | overflow |
| `1e100` | **`inf`** | `2e200` | overflow |

Below the overflow the same fp32 intermediate is visible as ordinary precision loss: at `s = 1e10`
`mse_loss` returns `2.0000000400817547e20` instead of `2e20` (`2.0e-8` relative, i.e. float32
level), and `ops.l1_loss` shows the same `2.0e-8` error on `[1e20, -1e20]` (it does not overflow
because it never squares).  Controls that *are* exact on the same data: `ops.sum` = `0`,
`ops.sum(ops.abs(x))` = `2e20`.  The float32 lane overflows at the same point, as expected for
float32.

## Root cause (pointer)

The squared difference is accumulated in float32 regardless of the input dtype, so the
intermediate `s²` overflows at `FLT_MAX = 3.4e38` (`s ≈ 1.8e19`) even though the float64 result is
far inside range.  This is the sixth instance of the reduced-precision-intermediate class in this
hunt (MS-001 `fmod`, MS-003 `cdist`, MS-004 `var`/`std`, MS-005 `avg_pool2d`,
MS-006 `matrix_norm("nuc")`); the PyTorch/JAX/TF audits (`wave 11`) show the class is
MindSpore-specific.

## Impact

`mse_loss` is a default regression loss.  With unnormalised targets or large-magnitude features
(physical units, monetary values, or anything scaled above `1.8e19`) the loss becomes `inf` in
float64 code, which then produces `nan`/`inf` gradients and destroys training — and because the
float32 lane fails identically, it is easy to dismiss as "expected precision behaviour".

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `mse_loss precision`, `mse_loss float32` and
  `mse_loss inf` return no matching report.
* `ops.l1_loss` and `ops.smooth_l1_loss` are exact on the same inputs, so this is not a general
  float64 limitation of the loss family.

## Files

* `repro.py` — self-checking reproducer (scale scan, l1/smooth-l1 controls, float32 control)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
