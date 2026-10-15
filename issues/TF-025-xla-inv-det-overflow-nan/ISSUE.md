# Under XLA, `tf.linalg.inv`/`solve` return NaN for finite matrices whose determinant overflows

**Component:** TensorFlow 2.21.0 (CPU), `tf.linalg.inv` and `tf.linalg.solve` under
`tf.function(jit_compile=True)` (XLA), `float32` and `float64`.
**Severity:** silent wrong value on **finite, valid input**.  For `A = [[a, a], [a, 2a]]` the exact
inverse is `[[2/a, -1/a], [-1/a, 1/a]]`, which is finite and representable for every finite `a`; the
XLA lane returns **all NaN** while eager, `tf.function`, numpy, torch and JAX (which also executes XLA)
all return the exact finite inverse.
**Status:** present in 2.21.0 (XLA CPU).  The switch is exactly at the point where `det(A) = a**2`
overflows the dtype: fp64 is correct up to `a = 1e154` and all-NaN from `a = 1e155`; fp32 is correct
up to `1e19` and all-NaN from `1e20`.  A diagonal matrix with the same overflowing determinant is
correct in both lanes, so the failure needs the off-diagonal structure.
**Found with:** the wave-354 Rondo XLA-vs-eager structural sweep of `tf.linalg.*` on non-finite
inputs, followed by a threshold scan on finite extreme inputs; certified reproducer `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 42 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf
a = 1e300
A = tf.constant([[a, a], [a, 2*a]], dtype=tf.float64)     # finite; det(A) = a**2 overflows fp64
# exact inverse: [[2/a, -1/a], [-1/a, 1/a]] = [[2e-300, -1e-300], [-1e-300, 1e-300]]

tf.linalg.inv(A)                                    # [[2e-300, -1e-300], [-1e-300, 1e-300]]
tf.function(tf.linalg.inv)(A)                       # the same finite matrix
tf.function(tf.linalg.inv, jit_compile=True)(A)     # [[nan, nan], [nan, nan]]   <- XLA
np.linalg.inv([[a, a], [a, 2*a]])                   # the exact finite matrix
torch.linalg.inv(...)                               # the exact finite matrix
jnp.linalg.inv(...)  (eager and jit, JAX is XLA)    # the exact finite matrix
```

Threshold (fp64, `A = [[a, a], [a, 2a]]`, `det = a**2`, fp64 max `1.8e308`):

| `a` | `det(A)` | eager | XLA |
|---|---|---|---|
| `1e150` | `1e300` | `2e-150` | `2e-150` |
| `1e154` | `1e308` | `2e-154` | `2e-154` |
| `1e155` | overflow | `2e-155` | **all NaN** |
| `1e200` | overflow | `2e-200` | **all NaN** |
| `1e300` | overflow | `2e-300` | **all NaN** |

fp32 switches between `1e19` (`3.4e38` max, `a**2 = 1e38` fits) and `1e20` (`a**2` overflows).

## Root cause

The TensorFlow→XLA lowering of `MatrixInverse` (and `MatrixSolve`, which uses it) forms a quantity
that overflows when the determinant does - e.g. an unscaled LU/cofactor step or an explicit
`det`/adjugate - so `inf`/`inf` or `0 * inf` produces NaN, even though every entry of the true inverse
is small.  The determinant threshold (`a = sqrt(dtype_max)`) pins this: below it the lane is exact,
above it the result is NaN.

Two attribution controls separate this from the XLA shared kernels:

* **JAX runs XLA too**, and `jnp.linalg.inv` returns the exact finite inverse in both eager and jit
  mode, so the shared XLA triangular-solve/LU kernels are not at fault - the TensorFlow XLA path is.
* A **diagonal** matrix with the same overflowing determinant (`diag(1e300)`) is correct in the XLA
  lane, so the failure is specific to the non-diagonal factorisation used by the TF lowering.

This is the opposite attribution from the campaign's TF-024 (XLA CPU `cholesky` with an `inf`
diagonal), where JAX is wrong as well and the shared kernel is at fault.

## Impact

* Large-but-finite matrices are routine: covariance/precision matrices from unnormalised features,
  activations, or kernels computed in fp64 that are later scaled, and any ill-conditioned system
  whose entries are ~1e155+ while the inverse entries are ~1e-155.  Eager computes the inverse,
  `jit_compile=True` returns NaN, silently, with no error - a compiled-vs-eager divergence that a
  finite-accuracy test suite never sees (the small-scale tests pass).
* `solve(A, I)` shares the path, so linear solves inside a compiled model fail the same way.

## Suggested fix

Make the XLA `MatrixInverse`/`MatrixSolve` lowering scale-invariant, as the shared XLA kernels and
LAPACK are: factor `A = s * A'` with `s = max|A_ij|` (or use a scaled LU / iterative refinement) and
return `inv(A') / s`, so the determinant never has to be formed.  Regression test: `inv([[1e155,
1e155], [1e155, 2e155]])` must equal `[[2e-155, -1e-155], [-1e-155, 1e-155]]`, not NaN.

## Controls (all in `repro.py`, 42 checks, 0 failing on each of 2 replays)

* **Exact-inverse reference** - the closed form `[[2/a, -1/a], [-1/a, 1/a]]` is compared with eager,
  `tf.function` and numpy (all match to `1e-14` relative); only the XLA lane is NaN.
* **Threshold control** - the fp64 sweep `a = 1e150, 1e154, 1e155, 1e156, 1e200, 1e300` and the fp32
  sweep `1e10, 1e19, 1e20, 1e25` switch exactly when `a**2` crosses the dtype maximum.
* **Shape/structure control** - `diag(1e200)` and `diag(1e300)` (same overflowing determinant) are
  finite and equal to eager in the XLA lane.
* **Well-scaled control** - random 2x2/3x3/4x4 positive-definite matrices invert to the identity in
  every lane, with eager and XLA agreeing to `1e-10`.
* **Below-threshold reconstruction** - at `a = 1e150` the XLA result reconstructs the identity after
  scaling (`(A/a) @ (a * inv_XLA(A)) == I` to `1e-12`).
* **Cross-framework controls** - numpy, torch and JAX return the exact finite inverse for the `1e300`
  witness in their own interpreters.

## Novelty

Searched the TensorFlow and openxla/xla trackers for `linalg inv overflow xla`,
`matrix_inverse nan large values`, `tf.linalg.inv nan`, `linalg det overflow inverse xla`,
`matrix inverse overflow` and `MatrixInverse nan`: the only hit is an unrelated 2020 deprecation-warning
issue.  The adjacent TensorFlow reports are the **opposite** direction and a different input class:
#118378 (open) / #120237 (closed) "XLA jit_compile=True returns finite values for singular
`tf.linalg.solve` while eager raises" - there eager *rejects* and XLA *accepts*, whereas here the input
is finite and non-singular in the sense used by every reference implementation and XLA returns NaN
where all others return the exact inverse.  The campaign's TF-024 (XLA cholesky `inf` diagonal) is a
different operator, a non-finite input, and JAX is affected there but not here.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.3, Python 3.12.13, WSL2; cross-checks with
torch 2.12.1+cu130 (CPU) and jax/jaxlib 0.10.2 (CPU) - see collect_env.txt
```

## Anticipated objections and responses

* **"The matrix is ill-conditioned, NaN is acceptable."**  The input is finite, `det = 1e600` is an
  intermediate that overflows but the exact inverse is `~1e-300` and every reference implementation
  (LAPACK/Eigen in numpy/TF-eager/torch, and XLA itself through JAX) returns it exactly.  Returning
  NaN where a representable exact answer exists is a wrong value, not numerical noise.
* **"This is the known singular `solve` issue."**  Those reports (#118378/#120237) are about singular
  matrices where eager *raises* and XLA *accepts*; here eager succeeds and XLA fails, on a
  determinant-overflowing but invertible matrix.
* **"Maybe XLA is right and the others are wrong."**  The exact inverse is a rational function of
  `a`; four independent implementations agree on it to machine precision, and XLA through JAX agrees
  too.
* **"Only extreme values are affected."**  The threshold is `sqrt(dtype_max)` (1e154 fp64, 1.8e19
  fp32), which is reachable by any unnormalised covariance/feature matrix, and the same path is used
  by `solve`.
