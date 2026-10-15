# `ops.avg_pool2d` accumulates in float32: fp32-level error for float64 inputs and `inf` where the exact average is representable

**Component:** MindSpore 2.9.0, `mindspore.ops.avg_pool2d` (PYNATIVE and GRAPH)
**Severity:** silent wrong result (`inf`, or ~1e-7 relative error, for perfectly representable float64 data)
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo fp64 accuracy audit (`artifact/hunt-v2/ms_precision_fuzz.py`)

---

## Summary

With float64 inputs the 2x2 average pool of a constant tensor must return exactly that constant.
MindSpore returns an fp32-rounded value above ~`1e19` and `inf` at `1e100`, while `ops.sum` and
`ops.mean` on the very same tensor are exact:

```python
import numpy as np, mindspore as ms
from mindspore import ops

x = ms.Tensor(np.full((1, 1, 2, 2), 1e100), ms.float64)
ops.avg_pool2d(x, 2, 2)     # inf       <-- exact: 1e100
ops.mean(x)                 # 1e100     (correct)
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| constant value `s` (float64) | `ops.avg_pool2d` | exact average | relative error |
|---|---|---|---|
| `1e10` | `1e10` | `1e10` | `0` |
| `1e19` | `1e19` | `1e19` | `2.0e-09` |
| `1e20` | `1e20` | `1e20` | `2.0e-08` |
| `1e30` | `1e30` | `1e30` | `1.5e-08` |
| `1e100` | **`inf`** | `1e100` | `inf` |

Controls on the *same* tensors:

| control | result |
|---|---|
| `ops.sum` at `1e20` | `4e20` (exact) |
| `ops.mean` at `1e20` | `1e20` (exact) |
| `ops.sum` / `ops.mean` at `1e100` | exact |
| `avg_pool2d` with a non-constant 4-element window | fp32-level error vs NumPy at `1e20` |
| float32 inputs | fp32-level error (expected for that dtype) |

`ops.pdist`, `ops.vector_norm` and `ops.matrix_norm('fro')` are also exact at these magnitudes, so
the pooling accumulation is the outlier.

## Root cause (pointer)

`AvgPool2d` sums the window and divides by the element count with a float32 accumulator regardless
of the input dtype, so the running sum leaves the float32 range (`(1.8e19)²` is not involved here,
the threshold comes from the *sum* of the window).  A dtype-preserving accumulation (as `ops.mean`
uses) would return the exact result.

This is the fourth instance of the reduced-precision-intermediate class found in this hunt
(MS-001 `fmod`, MS-003 `cdist`, MS-004 `var`/`std`, MS-005 here).

## Impact

Average pooling over large-magnitude feature maps (unnormalised activations, physical or monetary
units, or any data scaled above ~`1e19`) silently loses precision in float64 code and becomes `inf`
above `1e100`.  In a network the `inf` propagates to every downstream layer; the sub-threshold case
is worse because it is invisible — the result looks plausible and only differs in the last digits.

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `avg_pool2d precision`, `avg_pool2d float32` and
  `avg_pool2d overflow` return no matching report.
* `ops.mean` on the same input is exact, so this is not a documented dtype convention; NumPy's
  `mean`/`average` of the same window is exact as well.

## Files

* `repro.py` — self-checking reproducer (constant and non-constant windows, scale scan, PYNATIVE +
  GRAPH, `sum`/`mean`/`fro`-norm controls, float32 control)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
