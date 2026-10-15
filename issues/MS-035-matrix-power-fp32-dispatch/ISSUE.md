# MindSpore `ops.matrix_power` silently computes in float32 for float64 inputs

**Component:** MindSpore 2.9.0 (CPU), `mindspore.ops.matrix_power`, float64 inputs, in PyNative and
under `ms.jit` (the returned tensor's dtype still says `Float64`).
**Severity:** silent wrong values at float32 precision.  `matrix_power(A, 1)` must return a copy of
`A`; for a float64 input it returns the **float32** result cast back to float64, bit-for-bit.  The
error is therefore at the fp32 level (~1e-8 relative) for ordinary matrices, and any entry beyond the
float32 range produces `inf`/`NaN` on an input that float64 represents exactly.  The same op also
rejects `matrix_power(A, -1)` for the perfectly invertible `[[1, 1e4], [0, 1]]` (`det = 1`) with
"negative power can not apply to singular matrix".
**Status:** present in 2.9.0 on CPU.  `ops.matmul` on the same float64 tensor is exact, so the input
and the dtype are not the problem - the downcast is inside `matrix_power`.
**Found with:** the wave-366 Rondo matrix-function probe (the closed-form Jordan witness
`[[1, b], [0, 1]]`, `A^n = [[1, n b], [0, 1]]`, plus the "n = 1 must be a copy" identity);
certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`,
31 checks).

---

## Summary

```python
import numpy as np, mindspore as ms
from mindspore import ops
A = np.random.default_rng(0).normal(size=(3, 3))          # float64

r = ops.matrix_power(ms.Tensor(A), 1)                     # must be a copy of A
r.dtype                                                    # Float64   (the dtype lies)
np.asarray(r.asnumpy()) == A                               # False, max |diff| = 2.5e-08
np.asarray(r.asnumpy()) == A.astype(np.float32).astype(np.float64)   # True, bit-for-bit
```

Large single entries make the overflow explicit:

| `b` in `A = [[1, b], [0, 1]]`, `n = 1` | exact float64 | MindSpore (`dtype=Float64`) | `np.float32(b)` |
|---|---|---|---|
| `1e10` | `1e10` | `1e10` | `1e10` |
| `1e11` | `1e11` | `9.9999998e10` | `9.99999795e10` |
| `1e38` | `1e38` | `9.99999968e37` | `9.99999968e37` |
| `1e39` | `1e39` (representable) | `inf` (and `nan` on the diagonal) | `inf` |

and with `n >= 2` the same Jordan block gives all-NaN where the exact power `[[1, n b], [0, 1]]` is
finite (e.g. `b = 1e39, n = 2` -> `[[1, 2e39], [0, 1]]`).

## Root cause

The op's kernel is dispatched to a float32 implementation regardless of the input dtype: the float64
output is bit-for-bit `float32(A)^n` widened back to float64, and it overflows exactly at the float32
range (`|entry| > ~3.4e38`), not at the float64 range.  The attribution control is `ops.matmul` on the
same tensor, which returns the exact float64 product (`[[1, 2e11], [0, 1]]` for `b = 1e11`, dtype
`Float64`), and `ops.linalg.inv` on the same matrices, which is float64-exact.  The negative-power
branch computes the inverse through the same fp32 path and then decides the matrix is singular: for
`[[1, 1e4], [0, 1]]` the determinant is exactly 1, the inverse `[[1, -1e4], [0, 1]]` is exact and
representable, and numpy returns it, but MindSpore raises
`ValueError: For MatrixPower, negative power can not apply to singular matrix`.

## Impact

* `matrix_power` is used for Markov-chain transition matrices, powers of covariance/precision
  matrices, graph/walk computations and polynomial matrix expressions.  A float64 pipeline silently
  loses six to seven digits of precision, and any entry above the float32 range turns the result into
  `inf`/`NaN` - with `Float64` in the dtype field, so dtype-based checks do not catch it.
* The false "singular" rejection means negative powers of perfectly invertible float64 matrices cannot
  be computed at all above small magnitudes (~1e4).

## Suggested fix

Dispatch the kernel on the input dtype (a float64 kernel, or an explicit cast-free path), and derive
the negative-power singularity test from a float64 factorisation rather than the fp32 intermediate.
Regression test: `matrix_power(A, 1)` must equal `A` bit-for-bit for a float64 `A`, and
`matrix_power([[1, 1e39], [0, 1]], 1)` must be `[[1, 1e39], [0, 1]]`, plus
`matrix_power([[1, 1e4], [0, 1]], -1) == [[1, -1e4], [0, 1]]`.

## Controls (all in `repro.py`, 31 checks, 0 failing on each of 2 replays)

* **Identity control** - `matrix_power(A, 1)` is compared with both the float64 copy and the float32
  computation: it fails the former and matches the latter bit-for-bit (`max|diff| = 0.0`), with the
  output dtype asserted to be `Float64`.
* **Attribution controls** - `ops.matmul(A, A)` returns the exact float64 product bit-for-bit for the
  same tensor and reports `Float64`; numpy's `matrix_power` is the float64 reference throughout.
* **Range controls** - the failure boundary is the float32 range: exact at `b = 1e10`, fp32-rounded at
  `1e11` and `1e38`, `inf`/`NaN` at `1e39`; the fp32 lane itself is self-consistent and exact at
  `1e30`, and a small integer matrix is exact.
* **Negative-power controls** - `b = 100` returns the exact `[[1, -100], [0, 1]]` while `b = 1e4`
  raises the false-singular error, with `det = 1` and numpy's exact inverse asserted.
* **Lane control** - `ms.jit(matrix_power)` reproduces the fp32 result for `n = 1`, so the graph lane
  shares the kernel.

## Novelty

Searched the MindSpore tracker (GitHub mirror and the Gitee/web docs) for `matrix_power`,
`MatrixPower float32`, `matrix_power float64 precision` and `matrix_power overflow`: **zero results** -
the only documentation found is the `ops.matrix_power` API page.  The campaign's related records are
the drafted float64-with-float32-intermediate findings for `var`/`std` and `mse_loss`/`l1_loss`
(same class, different ops), the `ops.digamma` float64 dispatch (known mindspore#400) and the
"26 special functions: is the fp64 output really fp64?" negative row (which did not include
`matrix_power`).  This is the first report for `matrix_power`, and its `n = 1` witness makes the
downcast a bit-exact proof rather than a precision estimate.

## Environment

```
mindspore 2.9.0 (CPU), numpy 2.5.3, Python 3.12.13, WSL2 (see collect_env.txt)
```

## Anticipated objections and responses

* **"A ~1e-8 difference is just precision."**  The exact result of `matrix_power(A, 1)` is `A`; a
  2.5e-08 relative error is the fp32 result, proven bit-for-bit, and it becomes inf/NaN above the fp32
  range although float64 represents the answer.  This is a dtype-dispatch defect, not rounding.
* **"The dtype field says Float64, so my code is fine."**  That is exactly the trap: the metadata is
  float64 while the values are float32.
* **"Use `ops.matmul`/`ops.linalg.inv` instead."**  They are correct (and serve as the controls), but
  `matrix_power` is the documented API for integer powers and is the one that is wrong.
* **"The negative-power error is a documented restriction."**  The message claims the matrix is
  singular; `[[1, 1e4], [0, 1]]` has `det = 1` and an exact representable inverse, and the same call
  succeeds at `b = 100`, so the check is misfiring.
