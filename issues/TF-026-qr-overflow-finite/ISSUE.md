# `tf.linalg.qr` overflows to inf/NaN for finite matrices whose exact R is representable

**Component:** TensorFlow 2.21.0 (CPU), `tf.linalg.qr` in eager mode **and** under
`tf.function(jit_compile=True)` (XLA), `float32` and `float64`.
**Severity:** silent wrong value on **finite, valid input**.  For `A = [[2a, a], [a, 2a]]` the exact
factor has `R11 = -a*sqrt(5)` and `R22 = 3a/sqrt(5)`, both finite and representable for every finite
`a` (checked to `1e200`); TensorFlow returns inf/NaN instead, and at `a = 1e154` the same TensorFlow's
**XLA lane returns the exact R while eager returns inf/NaN**, so the correct answer is demonstrably
reachable inside TF.
**Status:** present in 2.21.0.  Eager stops being finite from `a ~ sqrt(max/5)` (fp64 `1e154`, fp32
`1e19`, i.e. when the squared first-column norm `5a^2` overflows) and the XLA lane stops one scale
later (fp64 `1e155`, fp32 `1e20`, when `a^2` overflows).  numpy (LAPACK, scaled `nrm2`) and JAX - which
also executes XLA - stay exact at every scale tested.
**Found with:** the wave-356/357 Rondo finite-extreme-value probe of `tf.linalg.*` (the methodology
that produced TF-025); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 54 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf
a = 1e154
A = tf.constant([[2*a, a], [a, 2*a]], dtype=tf.float64)   # finite
# exact R diagonal: [-a*sqrt(5), 3a/sqrt(5)] = [-2.23606797749979e+154, 1.3416407864998738e+154]

tf.linalg.qr(A)[1]                                   # R has inf and NaN entries
tf.function(lambda x: tf.linalg.qr(x), jit_compile=True)(A)[1]   # exact R, finite
np.linalg.qr([[2*a, a], [a, 2*a]])[1]                # exact R, finite
jnp.linalg.qr(...)  (eager and jit, JAX is XLA)      # exact R, finite
```

Threshold table (fp64, `A = [[2a, a], [a, 2a]]`; numpy and JAX are finite and exact at every row):

| `a` | exact `R11 = -a*sqrt5` | eager | TF-XLA | numpy |
|---|---|---|---|---|
| `1e153` | `-2.236e153` | finite, exact | finite, exact | finite, exact |
| `1e154` | `-2.236e154` | **inf/NaN** | finite, exact | finite, exact |
| `1e155` | `-2.236e155` | **inf/NaN** | **inf/NaN** | finite, exact |
| `1e200` | `-2.236e200` | **inf/NaN** | **inf/NaN** | finite, exact |

fp32 has the same structure: eager is correct at `1e18` and non-finite from `1e19`, XLA from `1e20`,
numpy exact at `1e25`.

## Root cause

The eager implementation forms the Householder reflector from the unscaled column norm, i.e. it
squares the column entries: the first-column norm is `sqrt((2a)^2 + a^2) = a*sqrt(5)`, so
`5a^2` overflows for `a > sqrt(dtype_max/5)` - exactly the observed eager threshold (fp64 `6.0e153`
-> fails at `1e154`; fp32 `8.2e18` -> fails at `1e19`).  The XLA lane uses an intermediate whose
overflow threshold is `a^2 > dtype_max` (`1.34e154` fp64 -> fails at `1e155`; `1.8e19` fp32 -> fails
at `1e20`).  The result is `inf`/`NaN` in `R` even though every entry of the exact `R` is a normal
number, and `Q`/`R` no longer reconstruct `A`.

LAPACK (`numpy.linalg.qr`) and the XLA kernels reached through JAX scale the norm computation
(`nrm2`), so they are exact at `1e200`; the same TensorFlow XLA lane is exact at `1e154`, which
isolates the eager overflow rather than the problem being intrinsic to the input.

## Impact

* QR is used for least-squares solves, orthogonalisation, rank/conditioning work and in
  `tf.linalg.lstsq`/`tf.linalg.qr`-based whitening.  Any finite matrix whose column norm exceeds
  `sqrt(dtype_max/5)` (~`6e153` fp64, `8.2e18` fp32) silently yields inf/NaN in eager, while the same
  call under `jit_compile=True` (at `1e154`) and in numpy/JAX succeeds - a compiled-vs-eager
  divergence in the opposite direction from the usual one, and invisible in small-scale tests.
* fp32 is affected at much more ordinary magnitudes (~`1e19`), reachable by unnormalised features or
  covariance-like matrices.

## Suggested fix

Scale before factorising, as LAPACK does: compute the column norm with a scaled `nrm2` (or divide `A`
by `max|A_ij|` first and rescale `R` afterwards), so the reflector never squares an entry that can
overflow.  Regression test: `qr([[2e154, 1e154], [1e154, 2e154]])` must have the finite diagonal
`[-2.23606797749979e154, 1.3416407864998738e154]` in **both** lanes.

## Controls (all in `repro.py`, 54 checks, 0 failing on each of 2 replays)

* **Closed-form reference** - `R11 = -a*sqrt(5)`, `R22 = 3a/sqrt(5)` (verified against numpy in
  process) is compared with every finite lane to `1e-12` (fp64) / `1e-5` (fp32).
* **Internal lane control** - at `a = 1e154` the eager lane is inf/NaN while the *same TensorFlow's*
  XLA lane returns the exact R, so the correct value is reachable in TF.
* **Threshold law** - the fp64 sweep `1e100, 1e150, 1e153, 1e154, 1e155, 1e200` and the fp32 sweep
  `1e18, 1e19, 1e20, 1e25` show the two thresholds (`5a^2` for eager, `a^2` for XLA) and that numpy
  stays finite and exact at every point.
* **Structure controls** - where a lane is finite, `Q` is orthogonal (`Q @ Q.T = I` to `1e-12`) and `R`
  is upper triangular.
* **Ordinary-scale control** - random 2x2/3x3/4x4 matrices are finite in both lanes, reconstruct the
  input and agree to `1e-10`.
* **Cross-framework controls** - `numpy.linalg.qr` and `jnp.linalg.qr` (eager and jit) return the
  exact R at `a = 1e200`.

## Novelty

Searched the TensorFlow, JAX and openxla/xla trackers for `linalg qr overflow large`,
`tf.linalg.qr nan huge`, `qr decomposition inf large values`, `qr overflow` and
`linalg eigh eigvalsh large values wrong`: no matching report (the only JAX hit is an unrelated
`ormqr` PR).  The campaign's neighbouring linalg findings are TF-024 (XLA `cholesky` with an `inf`
diagonal), TF-025 (XLA `inv`/`solve` determinant overflow) and TF-020 (XLA `eigvalsh` ignoring
**non-finite** off-diagonals); none is `qr`, and none is an eager-lane overflow with a closed-form
finite answer.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.3, Python 3.12.13, WSL2; cross-checks with
jax/jaxlib 0.10.2 (CPU) - see collect_env.txt
```

## Anticipated objections and responses

* **"Very large inputs overflow, that is expected."**  The exact `R` is finite and representable
  (`-2.24e154` for a `1e154` input), numpy/LAPACK and JAX compute it exactly, and TF's own XLA lane
  computes it at `1e154` - the eager lane is the outlier, not the input.
* **"This is the same as the inv/cholesky overflow reports."**  Those are different operators
  (`inv`/`solve`/`cholesky`); the `qr` thresholds are different (`5a^2` eager, `a^2` XLA), and no `qr`
  report exists.
* **"Only fp64 extreme values are affected."**  fp32 breaks at `a ~ 1e19`, which is reachable by
  ordinary unnormalised data.
* **"Eager is deprecated for performance work."**  Eager is the reference lane the whole ecosystem
  compares against, and `tf.function` without XLA is the same implementation; the XLA lane also
  breaks at `1e155`.
