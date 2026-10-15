# MindSpore `ops.qr` returns a non-factorisation for finite matrices whose squared column norm overflows or underflows

**Component:** MindSpore 2.9.0 (CPU), `mindspore.ops.qr` in PyNative and under `ms.jit`, `float32` and
`float64`; `mode='reduced'` (default) and `mode='complete'` behave identically.
**Severity:** silent wrong result on **finite, valid input**.  A QR factorisation must satisfy
`Q @ R == A` with `Q` orthogonal; for `A = [[2a, a], [a, 2a]]` MindSpore returns factors that do not
reconstruct `A` at both ends of the dtype range, while numpy (LAPACK) and torch reconstruct it at
every scale.
**Status:** present in 2.9.0.  fp64 is correct up to `a = 1e153` and wrong from `a = 1e154` (the
squared column norm `5a^2` overflows); it is correct down to `a = 1e-150` and wrong from `a = 1e-200`
(`a^2` underflows).  fp32 switches one exponent range lower: correct at `1e18` / `1e-10`, wrong at
`1e19` / `1e-20`.
**Found with:** the wave-360/361 Rondo finite-extreme-value sweep of MindSpore linalg (the method
behind TF-025/TF-026 and JAX-018); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 76 checks).

---

## Summary

```python
import numpy as np, mindspore as ms
from mindspore import ops
a = 1e154
A = ms.Tensor([[2*a, a], [a, 2*a]])          # finite, cond(A) = 3
q, r = ops.qr(A)                              # q all-NaN, r = [[-inf, nan], [0, nan]]
(q.asnumpy() @ r.asnumpy())                   # all-NaN, while A is finite

a = 1e-200
A = ms.Tensor([[2*a, a], [a, 2*a]])
q, r = ops.qr(A)                              # q = I, r = [[2e-200, 1e-200], [0, 2e-200]]
(q.asnumpy() @ r.asnumpy())[1, 0]             # 0.0, but A[1, 0] = 1e-200
```

Lane table (`A = [[2a, a], [a, 2a]]`, `cond(A) = 3`, exact `|diag R| = [a*sqrt5, 3a/sqrt5]`,
`Q @ R == A`):

| dtype | `a` | PyNative `Q@R == A` | `ms.jit` `Q@R == A` | `|diag R|` | numpy `|diag R|` |
|---|---|---|---|---|---|
| fp64 | `1` | yes | yes | `[2.236, 1.342]` | same |
| fp64 | `1e153` | yes | yes | `[2.236e153, 1.342e153]` | same |
| fp64 | `1e154` | **no** | **no** | `[inf, nan]` | `[2.236e154, 1.342e154]` |
| fp64 | `1e200` | **no** | **no** | `[inf, nan]` | `[2.236e200, 1.342e200]` |
| fp64 | `1e-150` | yes | yes | `[2.236e-150, 1.342e-150]` | same |
| fp64 | `1e-200` | **no** | **no** | `[2e-200, 2e-200]` | `[2.236e-200, 1.342e-200]` |
| fp64 | `1e-300` | **no** | **no** | `[2e-300, 2e-300]` | `[2.236e-300, 1.342e-300]` |
| fp32 | `1e18` | yes | yes | `[2.236e18, 1.342e18]` | same |
| fp32 | `1e19` | **no** | **no** | `[inf, nan]` | `[2.236e19, 1.342e19]` |
| fp32 | `1e-10` | yes | yes | `[2.236e-10, 1.342e-10]` | same |
| fp32 | `1e-20` | **no** | **no** | `[2e-20, 2e-20]` | `[2.236e-20, 1.342e-20]` |

## Root cause

MindSpore's QR builds the Householder reflector from the **unscaled** column norm, i.e. it squares the
column entries before taking the root.  The first column is `(2a, a)` with norm `a*sqrt(5)`, so:

* **overflow**: `5a^2` leaves the dtype range (`a > sqrt(max/5)`: `6.0e153` fp64, `8.2e18` fp32) - the
  norm becomes `inf`, the reflector becomes `inf/inf = NaN`, and both factors are NaN;
* **underflow**: `5a^2` flushes to zero (`a < ~1e-162` fp64, `~1e-22` fp32 by the same squaring) - the
  reflector is the identity, so `Q = I` and `R = upper(A)`: a finite but **wrong** factorisation with
  `Q @ R != A` (the `(1,0)` entry is dropped) and the wrong `|diag R|`.

The correct implementation scales the norm (BLAS `nrm2`), which is why numpy and torch reconstruct `A`
at `1e200` and `1e-300`; MindSpore itself is exact at moderate scales and at `1e153`/`1e-150`, so the
algorithm is right and only its dynamic range is not.

## Impact

* Any QR-based pipeline - least-squares, orthogonalisation, whitening, rank/conditioning estimation,
  `lstsq`-style solves - silently produces a non-factorisation for a finite matrix whose entries are
  above `~6e153` (fp64) / `~8e18` (fp32) or below `~1e-162` / `~1e-22`.  Unnormalised feature or
  covariance matrices reach these magnitudes routinely in fp32.
* The underflow case is the more dangerous one: the output is finite and looks plausible (`Q = I`,
  upper-triangular `R`), but `Q @ R != A`, so reconstruction checks that use the factors fail or, worse,
  a downstream solve uses a silently different matrix.

## Suggested fix

Scale before factorising, as LAPACK does: compute the column norm with a scaled kernel (divide the
column by its largest magnitude first and multiply the resulting factors back), or reject/normalise
out-of-range magnitudes.  Regression test: `ops.qr([[2e154, 1e154], [1e154, 2e154]])` and
`ops.qr([[2e-200, 1e-200], [1e-200, 2e-200]])` must both satisfy `Q @ R == A` with
`|diag R| = [a*sqrt5, 3a/sqrt5]`.

## Controls (all in `repro.py`, 76 checks, 0 failing on each of 2 replays)

* **Defining-property oracle** - `Q @ R == A` (1e-6) and `Q^T Q == I` (1e-9) at moderate scales, for
  random 2x2/3x3/4x4 matrices and for `a = 1, 1e100, 1e150, 1e153, 1e-50, 1e-150`.
* **Representability control** - at every failing scale the exact `|diag R| = [a*sqrt5, 3a/sqrt5]` is
  finite and nonzero, and numpy's factors reconstruct `A`, so the input is not out of contract.
* **Lane controls** - PyNative and `ms.jit` agree at each threshold, and `mode='complete'` fails in the
  same way as `mode='reduced'`.
* **Underflow mechanism control** - the failing tiny-scale output is exactly `Q = I`, `R = upper(A)`,
  i.e. the first reflection is skipped (not just inaccurate).
* **Cross-framework control** - `torch.linalg.qr` reconstructs `A` at `1e154`, `1e200`, `1e-200` and
  `1e-300` with the exact `|diag R|` (checked in its own interpreter; `numpy.linalg.qr` in-process).

## Novelty

Searched the MindSpore GitHub mirror (`qr overflow`, `qr nan`, `qr large values`, `qr wrong`,
`qr decomposition`, `underflow linalg` - zero results), the Gitee/web tracker for MindSpore QR
overflow/underflow issues (only API docs for `ops.Qr` and `mint.linalg.qr`), and the campaign ledger
(the existing MindSpore `qr` row is a 24-op differential at moderate scales, "clean apart from the
known MS-004 class").  The neighbouring findings in this campaign are TF-026 (`tf.linalg.qr` overflows
on finite input, TensorFlow) and JAX-018 (`slogdet`'s 2x2 path); this is the MindSpore instance of the
unscaled-norm pattern and the first report for `ops.qr`.

## Environment

```
mindspore 2.9.0 (CPU), numpy 2.5.3, Python 3.12.13, WSL2; cross-check with
torch 2.12.1+cu130 (CPU) - see collect_env.txt
```

## Anticipated objections and responses

* **"Extreme magnitudes are out of contract."**  They are finite and the exact factors are
  representable; numpy, torch and MindSpore itself at `1e153`/`1e-150` all handle them, and `Q @ R != A`
  is a violation of the definition, not a precision loss.
* **"Maybe QR is only expected to be accurate for well-scaled input."**  The matrix is
  well-conditioned (`cond = 3`); the failure is a dynamic-range defect of the norm computation, and the
  overflow output is NaN rather than a loss of accuracy.
* **"NaN is acceptable for an overflow."**  `R11 = a*sqrt5 = 2.24e154` is a normal fp64 number, and the
  underflow case returns finite wrong values rather than NaN.
* **"This duplicates TF-026."**  That is a TensorFlow eager/XLA report; this is a different framework
  with its own kernel, a different failure shape (`Q = I`, `R = upper(A)` underflow) and different
  thresholds.
