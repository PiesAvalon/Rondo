# XLA's CPU Cholesky returns NaN where the exact factor is `+inf`

**Component:** TensorFlow 2.21.0 (CPU) `tf.linalg.cholesky` under `tf.function(jit_compile=True)`
(XLA), `float32` and `float64`; the same XLA CPU kernel is reached from JAX 0.10.2, where
`jnp.linalg.cholesky` returns the NaN in **both** its eager and jit lanes.
**Severity:** silent wrong value.  For a positive-(semi)definite matrix with a `+inf` on the diagonal
the exact Cholesky factor is well defined (`sqrt(inf) = inf`), and numpy, torch and TF's own
non-XLA lanes all return `inf`; the XLA lane returns `NaN`, so `L @ L.T` no longer reconstructs the
input.
**Status:** present in 2.21.0 (XLA CPU).  Only the diagonal entry that is `+inf` changes; a `1x1`
input is correct in every lane, so the defect is in the blocked `n >= 2` Cholesky kernel rather than
in the TF bridge or the dispatcher.
**Found with:** the wave-354 Rondo XLA-vs-eager structural sweep of `tf.linalg.*` / `tf.signal.*` /
selection ops on non-finite inputs; certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 112 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf
A = tf.constant([[float("inf"), 0.0], [0.0, 1.0]], dtype=tf.float64)

tf.linalg.cholesky(A)                                   # [[inf, 0], [0, 1]]   correct
tf.function(tf.linalg.cholesky)(A)                      # [[inf, 0], [0, 1]]   correct
tf.function(tf.linalg.cholesky, jit_compile=True)(A)    # [[nan, 0], [0, 1]]   XLA, wrong
np.linalg.cholesky([[np.inf, 0.0], [0.0, 1.0]])         # [[inf, 0], [0, 1]]
torch.linalg.cholesky(...)                              # [[inf, 0], [0, 1]]
```

Lane table (fp64, diagonal matrices; only the diagonal entries that are `inf` change):

| input | eager / `tf.function` | XLA |
|---|---|---|
| `[[inf]]` (1x1) | `[inf]` | `[inf]` (correct) |
| `diag(inf, 1)` | `diag(inf, 1)` | `diag(nan, 1)` |
| `diag(1, inf)` | `diag(1, inf)` | `diag(1, nan)` |
| `diag(inf, 2, 3)` | `diag(inf, 1.414…, 1.732…)` | `diag(nan, 1.414…, 1.732…)` |
| `diag(2, inf, 3)` | `diag(1.414…, inf, 1.732…)` | `diag(1.414…, nan, 1.732…)` |

The exact factor of a diagonal matrix is `diag(sqrt(d_i))`, and `sqrt(+inf) = +inf` in IEEE-754, so
the eager result is the mathematically correct one; the finite entries agree between the lanes to
1-2 ulp, and the off-diagonal entries stay `0`.

## Root cause

`tf.linalg.cholesky` under `jit_compile=True` is lowered to the XLA `Cholesky` custom call, whose CPU
implementation is the blocked LAPACK-style factorisation.  A `1x1` input goes through a direct
`sqrt`, but for `n >= 2` the algorithm uses the blocked/diagonal-update form, in which an `inf`
diagonal entry that would be `sqrt(inf) = inf` instead becomes `NaN`.  This is consistent with an
`x * rsqrt(x)` diagonal step (`inf * 0 = NaN`) or an `inf - inf` update; either way the kernel never
returns the exact `inf`.  Eager TF uses its own (Eigen) implementation and numpy/torch use LAPACK,
all of which return `inf`.

The strongest attribution control is JAX: JAX executes XLA kernels in **every** lane, and
`jnp.linalg.cholesky([[inf, 0], [0, 1]])` returns `[nan, 0, 0, 1]` in both its eager and jit lanes.
That localises the defect to the XLA CPU Cholesky kernel rather than to the TF→XLA bridge (the
bridge merely forwards the call), and it means JAX users see the wrong value unconditionally.

## Impact

* An infinite entry on the diagonal arises naturally when a model produces an infinite
  variance/precision (an `exp` overflow, a division by a zero variance, an unbounded prior), and
  `cholesky` is the standard entry point for whitening, Gaussian processes, multivariate normals and
  Kalman updates.  Under `jit_compile=True` the whole factor becomes NaN for such an input, silently,
  while the same call without XLA is correct - a compiled-vs-eager divergence that is invisible in
  finite tests.
* For JAX the same input is NaN in every lane, so the impact is unconditional there.

## Suggested fix

In the XLA CPU Cholesky kernel, keep the exact `sqrt` semantics for the diagonal: compute the
diagonal step so that `sqrt(+inf)` returns `+inf` (e.g. use `sqrt` of the updated diagonal rather
than `x * rsqrt(x)`, or special-case a `+inf` diagonal, as numpy/LAPACK do).  A regression test is the
`diag(inf, 1)` matrix above: the XLA lane must equal the eager lane (`inf`, not NaN).

## Controls (all in `repro.py`, 112 checks, 0 failing on each of 2 replays)

* **Exact-factor reference** — the exact factor of a diagonal matrix (`diag(sqrt(d_i))`, computed in
  fp64) is compared against every lane: eager and `tf.function` match it, XLA differs exactly at the
  infinite diagonal entries.
* **Reconstruction** — `L @ L.T` keeps `+inf` on the `[0,0]` diagonal for eager and becomes NaN for
  XLA; every finite diagonal entry stays finite in all lanes.
* **Size control** — `1x1 [[inf]]` is `inf` in every lane (including XLA), so the defect requires the
  blocked `n >= 2` path.
* **Finite control** — random positive-definite matrices of size 2-5 are finite, lower-triangular and
  reconstruct the input in every lane, and agree between eager and XLA to 1e-12; a `diag(1e154, 1)`
  (finite, `sqrt = 1e77`) is also identical across lanes.
* **Cross-framework controls** — `numpy.linalg.cholesky` and `torch.linalg.cholesky` return `inf` for
  the witness, matching eager.
* **Attribution control** — `jnp.linalg.cholesky` returns NaN in eager **and** jit (JAX is always
  XLA), so the fault is the XLA CPU kernel, not the TF bridge.

## Novelty

Searched the TensorFlow, JAX and openxla/xla trackers:

* **tensorflow** `cholesky inf xla` (1 result: an unrelated 2018 build failure) and
  `tf.linalg.cholesky nan jit_compile` (1 result: a 2023 half-dtype kernel registration PR) - nothing
  on non-finite input.
* **jax** `cholesky inf` (8 results: `solve`/`jit` x64 issues, `BCOO`, autograd NaNs) and
  `cholesky nan` (19 results: `cross(x,x)` gradient NaNs #38611, `multivariate_normal` NaN at large
  dimension #14345, ROCm skips) - none is the `inf` diagonal in the forward factor.
* **openxla/xla** `cholesky inf` and `cholesky nan` - zero results.

The campaign's neighbouring linalg-NaN drafts are TF-020 (XLA `eigvalsh` ignoring non-finite
off-diagonal entries) and JAX-013 (CUDA LU replacing NaN with 0.0); both are different operators and
different failure modes, and neither covers an `inf` diagonal turning into NaN in the CPU Cholesky
kernel.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.3, Python 3.12.13, WSL2; cross-checks with
torch 2.12.1+cu130 (CPU) and jax/jaxlib 0.10.2 (CPU) - see collect_env.txt
```

## Anticipated objections and responses

* **"An `inf` entry is an invalid input."**  `sqrt(+inf) = +inf` is exact IEEE arithmetic, the factor
  is well defined, and every other lane (eager TF, numpy, torch) computes it.  Even if one treats it
  as out of contract, returning a *finite* result would be one thing - returning NaN where the exact
  value is `inf` is a wrong value, and it is compiled-vs-eager only.
* **"This is the same as the known XLA linalg NaN issues."**  TF-020 is `eigvalsh` with non-finite
  inputs and JAX-013 is CUDA LU substituting zeros; neither is the `inf` diagonal in `cholesky`, and
  neither project's tracker has a matching report.
* **"`L @ L.T` is also NaN off-diagonal because `inf * 0`."**  That affects both lanes equally; the
  repro compares the diagonal, where eager gives `inf` and XLA gives NaN.
* **"Maybe XLA is right and numpy/torch are wrong."**  The exact factor of `diag(inf, 1)` is
  `diag(inf, 1)`; XLA's own `1x1` result (`inf`) shows the intended semantics, and JAX's `1x1`
  is `inf` as well.
