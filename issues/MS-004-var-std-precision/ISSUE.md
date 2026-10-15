# `ops.var` / `ops.std` compute in float32: fp32-level error for float64 inputs, total loss at a large offset, and overflow to `inf` at `|x| ≳ 1.8e19`

**Component:** MindSpore 2.9.0, `mindspore.ops.var`, `mindspore.ops.std` (PYNATIVE and GRAPH)
**Severity:** silent wrong result — variance/standard deviation are wrong by ~1e-7 relative at
ordinary magnitudes, by 100 % at a large offset, and `inf`/`nan` beyond the float32 range
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo fp64 accuracy audit (`artifact/hunt-v2/ms_precision_fuzz.py`)

---

## Summary

For **float64** inputs MindSpore's variance and standard deviation behave like float32
computations, both in accuracy and in range.  NumPy and PyTorch return the exact float64 results
for every case below.

```python
import numpy as np, mindspore as ms
from mindspore import ops

ops.var(ms.Tensor([1.0, 2.0, 3.0, 4.0], ms.float64))     # 1.2500000367055151   (exact: 1.25)
ops.var(ms.Tensor([1e8+0.1, 1e8+0.2, 1e8+0.3, 1e8+0.4], ms.float64))   # 0.0   (exact: 0.0125)
ops.var(ms.Tensor([1e20, -1e20, 5e19, 1.0], ms.float64))  # inf                  (exact: 5.46875e39)
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

### 1. Accuracy: float64 inputs, float32-level error

| input (float64) | MindSpore `var` | NumPy / PyTorch | relative error |
|---|---|---|---|
| `[1, 2, 3, 4]` | `1.2500000367055151` | `1.25` (exact) | `2.9e-08` |
| `[1, -1, 0.5, 1]` | `0.6718749691111192` | `0.671875` (exact) | `4.6e-08` |
| `[0.1, 0.2, 0.3, 0.4, 0.5]` | `0.0199999976295553` | `0.02` (exact) | `1.2e-07` |
| 256 standard normals (seed 0) | `1.0237928968572305` | `1.0237929892653232` | `9.0e-08` |
| `[1e8+0.1 … 1e8+0.4]` | **`0.0`** | `0.0125000007450581` | **`1.0`** |

`ops.std` shows the same factor (`std([1,2,3,4]) = 1.1180340051651001` vs
`1.1180339887498949`, `1.5e-08`).  The error level matches what the *float32* lane produces for
float32 inputs (`1.2e-07` on the same data), i.e. the float64 path is not preserving the dtype.

### 2. Range: overflow where the float64 result is representable

| input `[s, -s, s/2, 1]` | MindSpore `var` | exact value | |
|---|---|---|---|
| `s = 1e10` | `5.46875e+19` | `5.46875e+19` | agrees |
| `s = 1e19` | `5.46875e+37` | `5.46875e+37` | agrees |
| `s = 1e20` | **`inf`** | `5.46875e+39` | overflow |
| `s = 1e30` | **`inf`** | `5.46875e+59` | overflow |
| `s = 1e100` | **`nan`** | `5.46875e+199` | overflow |

The transition is again the float32 range: `(1.8e19)² ≈ 3.4e38 = FLT_MAX`.  This is the same
reduced-precision-intermediate class as **MS-001** (`fmod` in GRAPH mode) and **MS-003** (`cdist`).

### 3. Controls (what is *not* wrong)

| check | result |
|---|---|
| `ops.sum` / `ops.mean` on the same float64 inputs | exact to `1e-16` |
| `ops.vector_norm` / `ops.pdist` / `ops.matrix_norm('fro')` at the same scales | exact |
| float32 inputs (`var` in float32) | fp32-level error, as expected for that dtype |
| `var(axis=-1)` and `var_mean` | same wrong values as the scalar form |
| GRAPH mode (`ms.jit`) | same wrong values |

## Root cause (pointer)

The reduction is a one-pass `E[x²] − E[x]²` with the accumulation and the squaring done in
float32 for float64 inputs:

* the squaring is what overflows above `1.8e19`, and
* the one-pass formula is what produces exactly `0.0` for the large-offset case (the two terms
  cancel), where NumPy's two-pass and PyTorch's Welford accumulation return `0.0125`.

A dtype-preserving, numerically stable reduction (two-pass or Welford, as in `at::var`) would fix
all three symptoms at once.

## Impact

Variance and standard deviation are used for normalisation layers, feature scaling, loss terms,
uncertainty estimates and metrics.  With fp32-level accuracy in float64 code:

* any test or computation that distinguishes single from double precision will misbehave, and
  downstream results inherit ~1e-7 relative error even for well-conditioned data;
* data with a large offset (sensor readings, timestamps, prices in minor units, activations with a
  large mean) can collapse to a variance of `0`, i.e. `std = 0` and then division by zero / `inf`
  in normalisation;
* data scaled above `1.8e19` (or the squared magnitude of the deviations) yields `inf`/`nan`, which
  propagates through the whole graph.

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `var precision float64`,
  `var float32 precision`, `variance large offset` and `std wrong result precision` return no
  matching report.
* NumPy and PyTorch agree with each other (exact `1.25`, `0.02`, `0.0125`) and disagree with
  MindSpore, so this is not a precision-convention question.  The float32 lane is a control: its
  error is expected, the float64 lane's is not.
* This is the third instance of the reduced-precision-intermediate class in this hunt (MS-001,
  MS-003, MS-004), so the individual report also serves as evidence that the class is systemic.

## Parameter coverage

The same float32 intermediate appears with `ddof=1`: `ops.std([1,2,3,4], ddof=1)` carries an fp32-level
relative error (~1e-7) and `ops.std([1e20, -1e20, 5e19, 1.0], ddof=1)` is `inf` where the exact value
`1.29e19` is finite.  The reproducer asserts both, so this report covers the `ddof` parameter rather
than only its default value.

## Files

* `repro.py` — self-checking reproducer (accuracy table, offset scan, overflow scan, dtype and
  `sum`/`mean`/`vector_norm` controls, GRAPH lane, `--scan`)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
