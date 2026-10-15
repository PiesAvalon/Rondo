# `jnp.linalg.slogdet` loses the sign and log-determinant of 2×2 matrices at extreme finite scale

**Component:** JAX 0.10.2 (`jaxlib` 0.10.2), `jnp.linalg.slogdet` with the default `method='lu'`, on the
CPU and CUDA backends, `float32` and `float64`.
**Severity:** silent wrong value in the one function that exists to avoid determinant overflow.  For a
finite, well-conditioned 2×2 matrix whose determinant leaves the dtype range it returns
`(nan, nan)` or `(0.0, -inf)` where the sign is known and the log-determinant is a representable
number; the same call with `method='qr'`, the n = 1 and n ≥ 3 paths, `numpy` and `torch` are all exact.
**Status:** present in 0.10.2 on both backends.  The 2×2 fast path `_slogdet_2x2` forms
`a00*a11 - a01*a10` explicitly; the small-n determinant helpers `_det_2x2`/`_det_3x3` were rewritten by
the fix for **jax#39905 / PR #39923** (merged 2026-08-12), but `_slogdet_2x2` was not touched, so the
defect is not covered by that fix.
**Found with:** the wave-359 Rondo finite-extreme-value sweep of `jnp.linalg.*` (the method that
produced TF-025/TF-026); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 61 checks).

---

## Summary

```python
import numpy as np, jax.numpy as jnp
a = 1e155
A = jnp.asarray([[2*a, a], [a, 2*a]])          # finite, cond(A) = 3, log|det| = log(3) + 2 log(a)

jnp.linalg.slogdet(A)                          # (nan, nan)          <- default method='lu'
jnp.linalg.slogdet(A, method='qr')             # (1.0, 714.8999...)  <- correct
np.linalg.slogdet(np.asarray(A))               # (1.0, 714.8999...)
```

Lane table (`A = [[2a, a], [a, 2a]]`, `cond(A) = 3`, exact `log|det| = log 3 + 2 log a`):

| `a` | default `method='lu'` | `method='qr'` | numpy | exact |
|---|---|---|---|---|
| `1e100` | `(1, 461.616)` | `(1, 461.616)` | `(1, 461.616)` | `461.616` |
| `1e154` | `(1, **inf**)` | `(1, 710.295)` | `(1, 710.295)` | `710.295` |
| `1e155` | `(**nan, nan**)` | `(1, 714.900)` | `(1, 714.900)` | `714.900` |
| `1e200` | `(**nan, nan**)` | `(1, 922.133)` | `(1, 922.133)` | `922.133` |
| `1e300` | `(**nan, nan**)` | `(1, 1382.65)` | `(1, 1382.65)` | `1382.65` |
| `1e-200` | `(**0, -inf**)` | `(1, -919.935)` | `(1, -919.935)` | `-919.935` |
| `1e-300` | `(**0, -inf**)` | `(1, -1380.45)` | `(1, -1380.45)` | `-1380.45` |

float32 switches one exponent range lower and has the same shape: `a = 1e19` -> `(1, +inf)`,
`a = 1e20` -> `(nan, nan)`, `a = 1e-38` -> `(0, -inf)`, while `method='qr'` stays finite and correct.
`torch.linalg.slogdet` gives `(1, 714.9)`, `(1, 922.133)`, `(1, -919.935)` for the fp64 rows.

## Root cause

`jax/_src/numpy/linalg.py` special-cases small matrices:

```python
def _slogdet_2x2(a):
  a00, a01 = a[..., 0, 0], a[..., 0, 1]
  a10, a11 = a[..., 1, 0], a[..., 1, 1]
  det = (a00 * a11) - (a01 * a10)      # overflows/underflows with the determinant itself
  abs_det = ufuncs.abs(det)
  return ufuncs.sign(det), ufuncs.real(ufuncs.log(abs_det))
```

`slogdet` exists precisely so that `log|det|` survives when `det` does not fit the dtype; this path
materialises `det` first, so both outputs are lost together.  The other paths are unaffected at the
same scales: `_slogdet_1x1` uses the entry directly, and `_slogdet_lu` (n ≥ 3) and `_slogdet_qr`
(`method='qr'`) compute the sign and the sum of `log|pivot|` without forming the determinant - which is
why the n = 3 control and the `qr` option are exact in the table above.

The closely related **jax#39905** (closed 2026-08-12) reported the same *class* of defect for
`jnp.linalg.det`'s 2×2/3×3 closed forms (wrong sign / `inf` / `nan` on a moderately scaled
well-conditioned matrix) and was fixed by **PR #39923**, which replaced `_det_2x2` and `_det_3x3`; its
diff does not touch `_slogdet_2x2` (the reporter explicitly used `slogdet` as the *correct* reference
there, which holds only while the determinant fits the dtype).  This report is therefore the surviving
instance of the same small-n-helper pattern in a different function, with a pure overflow/underflow
trigger rather than cancellation.

## Impact

* `slogdet` is the standard way to evaluate log-determinants in Gaussian processes, normalising flows,
  multivariate-normal log-likelihoods and Bayesian model comparison.  A covariance/precision matrix
  whose determinant exceeds the dtype range (large unnormalised features, `1e155`+ fp64 or `1e19`+
  fp32) makes the default `slogdet` return NaN, while the mathematically equivalent `method='qr'` and
  every other framework return the finite log-determinant - a silent failure exactly in the regime
  `slogdet` was designed for.
* The underflow side is worse than a NaN in one sense: the sign comes back `0.0`, which is not a valid
  sign and silently propagates into products of signs.

## Suggested fix

Route the n = 2 case through the LU/QR machinery (or the same closure used by `method='qr'`), or make
the closed form scale-invariant: factor out `s = max|a_ij|`, compute the sign and
`log|det(A/s)|`, then add `2*log(s)` to the log and keep the sign from the scaled determinant.
Regression test: the fp64 rows of the table above must return `(1, log 3 + 2 log a)` for
`a = 1e-300 ... 1e300`, matching `method='qr'`.

## Controls (all in `repro.py`, 61 checks, 0 failing on each of 2 replays)

* **In-library path controls** - at `a = 1e155, 1e200, 1e-200` the n = 1 path is exact, the n = 3 LU
  path matches numpy exactly, and `method='qr'` returns the exact `(1, log 3 + 2 log a)`; only the
  default 2×2 LU path fails.
* **Framework controls** - `numpy.linalg.slogdet` and `torch.linalg.slogdet` are exact on every row.
* **Conditioning control** - `cond(A) = 3`, so this is overflow, not cancellation or ill-conditioning;
  the same matrices at `a = 1, 1e50, 1e100` are exact in the default path.
* **Device control** - the CUDA lane reproduces `(nan, nan)`, i.e. the defect is in the Python-level
  helper rather than a device kernel.
* **fp32 control** - the thresholds are one exponent range lower (`1e19` -> `inf`, `1e20` -> `nan`,
  `1e-38` -> `(0, -inf)`) and `method='qr'` stays finite.
* **Ordinary controls** - `[[1,2],[3,4]]`, the identity and a diagonal negative matrix match numpy.

## Novelty

The tracker search for `slogdet` + overflow/inf/nan/2x2/large determinant returns **jax#39905**
(closed) and PR #39923, both about `jnp.linalg.det`'s 2×2/3×3 expansion; the issue states that
JAX's `slogdet` is correct on its inputs (true at moderate scale) and the merged fix replaces only
`_det_2x2`/`_det_3x3`.  The remaining hits are unrelated (#2380 singular-matrix gradient, #31428 a
Darwin test failure, #29883 explicit sharding, #4114 a custom-VJP error).  The campaign's JAX-013 is
the same *function name* on CUDA but a different defect (NaN **input** substituted by 0.0 in the LU
kernel), and its CPU witnesses are finite-input-free.  No report covers a finite 2×2 input whose
determinant simply leaves the dtype range.

## Environment

```
jax 0.10.2 / jaxlib 0.10.2, numpy 2.5.3, Python 3.12.13, WSL2, CPU and CUDA (RTX 4090);
cross-checks with torch 2.12.1+cu130 - see collect_env.txt
```

## Anticipated objections and responses

* **"Extreme inputs overflow; use `method='qr'`."**  The whole point of `slogdet` is to return a
  representable `log|det|` when `det` overflows; n ≥ 3, n = 1, `method='qr'`, numpy and torch all do
  it on the same matrices, so the default 2×2 path is the outlier, not the input.
* **"This is jax#39905."**  #39905 is `jnp.linalg.det` and was closed by a fix that rewrote
  `_det_2x2`/`_det_3x3`; it explicitly used `slogdet` as the correct lane and did not change
  `_slogdet_2x2`, which is what fails here.  The trigger is also different (pure overflow/underflow on
  a `cond = 3` matrix versus cancellation in the determinant expansion).
* **"NaN is an acceptable answer for a singular/ill-conditioned matrix."**  The matrix is
  well-conditioned (`cond = 3`) and non-singular; the correct `log|det|` is a normal number.
* **"Only the 2×2 case."**  Yes - and 2×2 is the common size for covariance blocks, and the n ≥ 3
  control shows a correct implementation is right there in the same file.
