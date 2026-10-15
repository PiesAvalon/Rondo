# `ops.matrix_norm` with an SVD-based order (`"nuc"` and `2`) returns garbage for float64 matrices whose norm exceeds `FLT_MAX`

**Component:** MindSpore 2.9.0, `mindspore.ops.matrix_norm` with `ord="nuc"` and `ord=2` (spectral norm), PYNATIVE
**Severity:** silent wrong result — an uninitialized/garbage value unrelated to the input scale
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo fp64 accuracy audit (`artifact/hunt-v2/ms_precision_fuzz.py`)

---

## Summary

The nuclear norm (the sum of the singular values) of `[[s, -s], [s/2, 1]]` is `≈ 1.8 * s`.
MindSpore is correct up to `s = 1e38` (with fp32-level accuracy), and from `s = 1e39` - exactly
where the norm exceeds `FLT_MAX = 3.4e38` - it returns a garbage value that does not depend on the
input at all:

```python
import numpy as np, mindspore as ms
from mindspore import ops

A = np.array([[1e100, -1e100], [5e99, 1.0]])
ops.matrix_norm(ms.Tensor(A, ms.float64), "nuc").item()   # 5.9e-22  (another process: 2.1e12)
np.linalg.norm(A, ord="nuc")                              # 1.80278e+100
ops.matrix_norm(ms.Tensor(A, ms.float64), "fro")          # 1.5e+100  (correct)
```

The junk value is *not stable across processes* (`5.9e-22` in one run, `2.1e12` in another, and
`2.11e12`/`2.12e12` varying within a third), which indicates the accumulator reads uninitialized
storage once it overflows; repeated calls inside one process can happen to repeat the same bytes.

## Additional symptom (wave 416): the SVD-based orders break the output-dtype contract

The same code path also narrows the dtype: a **Float64** input comes back as a **Float32** tensor for the
SVD-based orders, while the direct orders keep Float64.  With a witness whose value is not an fp32
value (`A = [[1e10, -1e10], [5e9, 1]]`) the returned Float64 tensor is bit-for-bit the Float32 call
widened, i.e. the fp64 path is not merely inaccurate, it is never taken:

| call on a `Float64` matrix | output dtype | bit-identical to the `Float32` call widened |
|---|---|---|
| `matrix_norm(A, "nuc")` | **Float32** | yes |
| `matrix_norm(A, 2)` | **Float32** | yes |
| `matrix_norm(A, -2)` | **Float32** | yes |
| `norm(A, "nuc")` / `norm(A, 2)` (alias) | **Float32** | yes |
| `matrix_norm(A, "fro")` / `(A, 1)` / `(A, inf)` | Float64 | no (genuinely fp64) |

`repro.py` now certifies both: the dtype of every SVD-based call and the bit-identity with the Float32
lane, plus the `"fro"`/`1`/`inf` controls that keep Float64.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| scale `s` | `matrix_norm(A, "nuc")` | NumPy | relative error |
|---|---|---|---|
| `1` | `2.5` | `2.5` | `0` |
| `1e10` | `1.80278e+10` | `1.80278e+10` | `6.6e-08` |
| `1e20` | `1.80278e+20` | `1.80278e+20` | `9.1e-08` |
| `1e30` | `1.80278e+30` | `1.80278e+30` | `1.3e-07` |
| `1e38` | `1.80278e+38` | `1.80278e+38` | `1.5e-08` |
| `1e39` | **`5.9e-22`** (process dependent) | `1.80278e+39` | `1.0` |
| `1e100` | **`5.9e-22`** (process dependent) | `1.80278e+100` | `1.0` |
| `1e154` | **`5.9e-22`** (process dependent) | `1.80278e+154` | `1.0` |

The transition sits exactly at the float32 range: the nuclear norm at `s = 1e38` is `1.8e38`
(below `FLT_MAX`) and correct, at `s = 1e39` it is `1.8e39` (above `FLT_MAX`) and broken.

Controls: `matrix_norm(A, "fro")` is **exact and deterministic** at every scale, and the ordinary
`ops.svd` of the same matrix at `1e100` is deterministic and matches NumPy — so the breakdown is
specific to the SVD-based orders (`"nuc"` and `2`), not to SVD or to `matrix_norm` in general.
Both orders show the same two symptoms: an fp32-level error below the breakdown and a
scale-independent junk value above it.

## Root cause (pointer)

`"nuc"` sums the singular values through a **float32 accumulator** (visible as the `1.5e-8` error at
`s = 1e38`), and once the sum exceeds `FLT_MAX` the accumulator overflows into uninitialized
storage, so the caller receives a small junk number (`5.9e-22`, `2.1e12`, ...) instead of `inf`.
Summing the singular values in the input dtype (or reusing the `"fro"` path's accumulator) would
fix both symptoms.

This belongs to the reduced-precision-intermediate class confirmed four times already in this hunt
(MS-001 `fmod`, MS-003 `cdist`, MS-004 `var`/`std`, MS-005 `avg_pool2d`); `"fro"` being exact is
what makes `"nuc"` a defect rather than a dtype convention.

## Impact

Spectral-norm terms are used for spectral normalisation and Lipschitz penalties; a loss or bound
that should be `1e100` becomes `2.4e-10` (or a negative number), so the penalty silently vanishes
instead of dominating.  Nuclear-norm terms are used in low-rank regularisation, matrix completion, robust PCA and spectral
penalties.  A training loss that should be `1.8e100` becomes `5.9e-22`, so the regulariser silently
disappears instead of dominating, gradients computed from it are meaningless, and the value is not
even reproducible across processes.  Below the breakdown the fp32
error is small enough to be invisible in tests but still wrong for float64 workloads.

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `matrix_norm nuclear`, `nuc norm wrong` and
  `nuclear norm precision` return no matching report.
* The `"fro"` control (`matrix_norm(A, "fro")` exact at all scales) shows the defect is specific to
  the `"nuc"` path, not a general float64 limitation of `matrix_norm`.

## Files

* `repro.py` — self-checking reproducer (scale scan, determinism check across repeated calls,
  `"fro"` and `ops.svd` controls, NumPy reference)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment

## Addendum (wave 50): the stable signature is nondeterminism, not the magnitude

Re-running the reproducer eight times showed 2/8 failures, because an earlier version asserted that
the junk value is a specific number of orders of magnitude away from the exact norm.  The junk comes
from uninitialized storage, so its magnitude is not stable - what *is* stable is that the identical
call returns a different value every time.  Five consecutive identical calls at each broken scale:

| order | scale | exact | five consecutive results | worst rel. error |
|---|---|---|---|---|
| `nuc` | `1e39` | `2.618034e+39` | `10373.67, 10378.42, 10425.78, 10310.76, 10431.83` | `1.0` |
| `nuc` | `1e100` | `2.618034e+100` | `10402.25, 10370.52, 10420.03, 10370.52, 10409.72` | `1.0` |
| `nuc` | `1e154` | `2.618034e+154` | `10409.59, 10420.90, 10420.02, 10373.06, 10410.79` | `1.0` |
| `2` | `1e39` | `1.618034e+39` | `10396.16, 10420.02, 10373.33, 10403.34, 10402.02` | `1.0` |
| `2` | `1e100` | `1.618034e+100` | `10311.45, 10429.19, 10430.83, 10394.82, 10492.75` | `1.0` |
| `2` | `1e154` | `1.618034e+154` | `10386.47, 10386.82, 10429.06, 10425.31, 10385.83` | `1.0` |

Every case returns five distinct values; the garbage is around `1e4` regardless of the input scale,
i.e. it does not depend on the matrix at all.  `ops.svd` on the same input is deterministic and
matches NumPy, so the uninitialized read is in the norm computation, not in the decomposition.
The reproducer now asserts exactly these two stable facts (nondeterminism, and a large relative
error) and no longer depends on the junk magnitude.
