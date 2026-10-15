# `ops.slogdet` violates its `sign in {-1, 0, 1}` contract: `inf` for subnormal determinants and values like `0.9999999999999998` for huge ones

**Component:** MindSpore 2.9.0, `mindspore.ops.slogdet` (PYNATIVE and GRAPH)
**Severity:** silent wrong result — the documented sign is out of range, and `sign * exp(logdet)`
reconstruction produces `nan`/`inf` instead of the true determinant
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo linalg sweep with NumPy as reference oracle (`artifact/hunt-v2/ms_linalg_fuzz.py`)

---

## Summary

`slogdet(A)` returns `(sign, logabsdet)` with `sign = det(A) / |det(A)|`, which is `-1`, `0` or
`1` by definition, and exists precisely so that very small or very large determinants can be
represented.  MindSpore violates that contract in two ways, both from computing the sign by
division instead of taking it from the LU pivots:

* **`inf`** when the determinant is subnormal (`det / |det|` overflows),
* values like **`0.9999999999999998`** (not exactly `1.0`, not in the documented set) when the
  determinant is huge, and `0` for a matrix whose determinant is infinite.

```python
import mindspore as ms, numpy as np
from mindspore import ops

A = np.diag([2.0, 1e-320])          # det = 2e-320 (subnormal), exactly what slogdet is for
ops.slogdet(ms.Tensor(A, ms.float64))         # (inf, -736.13)   <-- sign out of range
np.linalg.slogdet(A)                          # (1.0, -736.13)   <-- sign is 1
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| matrix | `det` | MindSpore `sign` | NumPy `sign` | MindSpore `sign*exp(logdet)` | NumPy `det` |
|---|---|---|---|---|---|
| `diag(2.0, 1e-320)` | `2e-320` (subnormal) | **`inf`** | `1.0` | `inf` | `2e-320` |
| `diag(5e-324, 1e-45)` | underflows | **`inf`** | `1.0` | `nan` | `0.0` |
| `diag(inf, 1.0)`-ish `[[inf,1],[1,1]]` | `inf` | **`0`** | `1.0` | `nan` | `inf` |
| control `diag(1, 1)` | `1` | `1` | `1.0` | `1` | `1.0` |
| control `diag(-1, 1)` | `-1` | `-1` | `-1.0` | `-1` | `-1.0` |
| control `[[0,1],[1,0]]` | `-1` | `-1` | `-1.0` | `-1` | `-1.0` |
| control `[[1,2],[2,4]]` (singular) | `0` | `0` | `0.0` | `0` | `0.0` |
| control `diag(1e-200, 1e-200)` | underflows to `0` | `1` | `1.0` | `0` | `0.0` |
| control `diag(1e200, 1e200)` | overflows to `inf` | `1` | `1.0` | `inf` | `inf` |
| `diag(1e38, 1e38)` | `1e76` | **`0.9999999999999998`** | `1.0` | `1e76` | `1e76` |
| `diag(1e30, 1e-30)` | `1` | **`0.9999999999999999`** | `1.0` | `1` | `1` |
| `[[-1,2,3],[4,-5,6],[7,8,-9]]` | `360` | **`0.9999999999999998`** | `1.0` | `360` | `360` |

The out-of-range sign appears in **PYNATIVE and GRAPH** mode (it is not a compilation artifact), for
float64, and only when the determinant leaves the normal float range: `sign` stays `±1`/`0` for
every control case.

## Root cause (pointer)

The sign appears to be derived from the determinant value (`det / |det|`, or a division by
`|det|`) instead of from the LAPACK/LU pivots (the product of the permutation sign and the signs of
the U diagonal), which is how NumPy/BLAS define it.  When `|det|` is subnormal, dividing by it can
overflow to `inf` (giving `sign = inf`) or underflow to `0`; when `det` is infinite the ratio
collapses to `0`.  The log term is correct in all the cases tested, so only the sign path is
affected.

## Impact

`sign` is documented as being in `{-1, 0, 1}` and is used for parity/orientation decisions and to
reconstruct the determinant (`det = sign * exp(logabsdet)`).  With `sign = inf` that
reconstruction yields `nan` or `inf` — defeating the entire purpose of using `slogdet` for
near-singular or badly scaled matrices (normalizing-flow log-densities, Gaussian
log-likelihoods, Jacobian corrections, matrix-manifold code).

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `slogdet sign inf`, `slogdet wrong sign`,
  `logdet subnormal` and `slogdet underflow` return no matching report.
* The only `slogdet` issues in the tracker are **#364** and **#363** (open), which report that
  1x1 matrices are *rejected* — a shape/API issue unrelated to the sign of a subnormal
  determinant.
* NumPy is used as the reference oracle here, and it returns the mathematically required `±1`;
  this is not a precision difference (an `inf` sign is not a rounding of 1).

## Files

* `repro.py` — self-checking reproducer (PYNATIVE + GRAPH, `{-1,0,1}` range check, NumPy
  reference, reconstruction impact, control set, `--scan` over exponents)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment

## Anticipated objections and responses

* **"`sign` is only defined up to convention."**  It is documented as being in `{-1, 0, 1}`, NumPy
  returns exactly `1.0` for both failing cases, and the contract matters operationally:
  `sign * exp(logabsdet)` is how callers reconstruct the determinant, and with `sign = inf` that
  product is `inf`/`nan`.
* **"A subnormal determinant is a corner case."**  The second symptom is not: ordinary large
  determinants return `0.99999999999999978` instead of `1.0`, so the sign is not snapped to `+-1`
  for perfectly normal inputs either.  Both come from computing `det / |det|` instead of taking the
  sign from the LU pivots, which is also what makes the fix small.
* **"Why not just clamp the output to `{-1,0,1}`?"**  Clamping would hide the subnormal case
  (`inf` clamped to `1` happens to be right here, but the same expression produces `nan` when both
  `det` and `|det|` underflow).  The draft recommends deriving the sign from the factorisation.
