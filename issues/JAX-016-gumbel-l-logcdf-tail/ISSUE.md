# `jax.scipy.stats.gumbel_l.logcdf` returns exactly `0.0` for the whole upper tail

**Component:** JAX 0.10.2 (`jaxlib` 0.10.2), CPU and CUDA, `jax.scipy.stats.gumbel_l.logcdf`,
`float32` and `float64`.
**Severity:** silent wrong value in log space.  For `3.7 <= x <= 6.6` the true `logcdf` is a nonzero
representable number (`-2.72e-18` at `x = 3.7`, `-1.94e-24` at `x = 4`, `-1.36e-289` at `x = 6.5`) and
SciPy returns it, while JAX returns **exactly `0.0`** - a 100% relative error in the one function whose
purpose is to preserve tail probabilities in log space.  `grad` collapses with it (`0.0` where the
derivative `pdf/cdf` is `1.06e-22`).  Absolute magnitudes are tiny, so the impact is limited to
log-space arithmetic, relative-error-based ranking and gradient-based tail work - but the value is
unambiguously wrong, and the fix is one line using a helper already imported in the same package.
**Status:** present in 0.10.2 on CPU and CUDA, in eager, `jit`, `vmap`, `grad` and both widths, and for
any `loc`/`scale` (it depends only on `z = (x - loc)/scale`).
**Found with:** the wave-344 Rondo sweep of `jax.scipy.stats` against `scipy.stats` (pdf/cdf/sf/logpdf/
logcdf/ppf/isf over 22 continuous and 5 discrete distributions, fp64), which flags exact-zero tails
where the reference is a nonzero number; certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 53 checks).

---

## Summary

```python
import jax, jax.numpy as jnp, scipy.stats as S
import jax.scipy.stats as J
jax.config.update("jax_enable_x64", True)

J.gumbel_l.logcdf(jnp.asarray(4.0))   # 0.0                     <- exactly zero
S.gumbel_l.logcdf(4.0)                # -1.9423376049564073e-24  <- the true tail
-np.exp(-np.exp(4.0))                 # -1.9423376049564073e-24  <- analytic check
```

The left-skewed Gumbel CDF is `cdf(x) = 1 - exp(-exp(x))`, so `logcdf(x) = log1p(-w)` with
`w = exp(-exp(x))`, and `logcdf(x) ~= -w` for large `x`.  Measured across the tail (`loc=0`,
`scale=1`):

| `x` | JAX | SciPy / analytic `-exp(-exp(x))` | relative error |
|---|---|---|---|
| 2.0 | `-0.0006181700170515628` | `-0.0006181700170515202` | 6.9e-14 |
| 3.0 | `-1.8921787488106934e-09` | `-1.8921786948382924e-09` | 2.8e-08 |
| 3.5 | `-4.107825191113088e-15` | `-4.150896920109045e-15` | 1.0e-02 |
| 3.6 | `-1.1102230246251565e-16` | `-1.2752306245162443e-16` | 1.3e-01 |
| 3.7 | **`0.0`** | `-2.7161822609294545e-18` | 1.0 |
| 4.0 | **`0.0`** | `-1.9423376049564073e-24` | 1.0 |
| 5.0 | **`0.0`** | `-3.5073891964646352e-65` | 1.0 |
| 6.5 | **`0.0`** | `-1.3572476073250428e-289` | 1.0 |
| 7.0 | `0.0` | `-0.0` (underflow) | correct |

## Root cause

`jax/_src/scipy/stats/gumbel_l.py`:

```python
z = (x - loc) / scale
neg_exp_z = lax.neg(lax.exp(z))
# xlog1p fails here, that's why log1p is used here
# even log1p fails for some cases when using float64 mode
# so we're using this formula which is stable
log_cdf = lax.log(-lax.expm1(neg_exp_z))
```

`log(-expm1(-w)) = log(1 - exp(-w))` is algebraically right, but `expm1(-w)` has no room to carry
`exp(-w)` once that term is below half an ulp of 1: for `w = exp(x) > ~36.7`,
`expm1(-w)` rounds to exactly `-1.0`, so `-expm1(...)` is exactly `1.0` and `log(1.0)` is exactly
`0.0`.  The stable form is `log1mexp(w) = log1p(-exp(-w))`, which keeps the small term.

The same package already does this correctly for the mirror symmetry: `gumbel_r.logsf` is

```python
log_sf = log1mexp(lax.exp(neg_z))
```

and it reproduces `-1.9423376049564073e-24` at `x = -4.0` exactly (see the controls), while
`gumbel_r.logcdf` is computed directly as `-exp(-z)` and is also accurate.  So `gumbel_l.logcdf` is
the only one of the four tails that loses the term, and the helper it needs
(`jax.scipy.special.log1mexp`) is already imported in the module.

## Suggested fix

```python
# jax/_src/scipy/stats/gumbel_l.py, logcdf()
from jax._src.scipy.special import log1mexp   # already imported in gumbel_r.py
log_cdf = log1mexp(lax.exp(z))                # log1p(-exp(-exp(z)))
```

Regression test: `gumbel_l.logcdf([3.7, 4.0, 5.0, 6.5])` must match
`-exp(-exp(x))` to relative `1e-12` (and `gumbel_r.logsf` at the mirrored points is the working
control to copy).

## Controls (all in `repro.py`, 53 checks, 0 failing on each of 2 replays)

* **Reference** — SciPy's `gumbel_l.logcdf` and the analytic `log1p(-exp(-exp(x)))`, which agree to
  relative `1e-12` at every affected point.
* **Mirror control** — `gumbel_r.logsf(-4.0)` = `-1.9423376049564073e-24` in JAX, equal to SciPy and
  to the analytic value, so the stable helper works and is the model for the fix.
* **Unaffected control** — `x = 0, 1, 2` match SciPy to relative `1e-12`; `cdf(4.0)` rounds to `1.0`
  in both libraries, so the lost tail is only observable in the log-space function.
* **Underflow control** — at `x = 7.0` the true tail has underflowed (`exp(-exp(7)) == 0.0`) and
  `0.0` is correct in both libraries, so the bug window is `3.7 <= x <= ~6.6`, not "all large x".
* **Mechanism control** — at `x = 3.7`, `exp(-exp(x)) = 2.7e-18 < 2**-53` and `expm1(-exp(x))` is
  exactly `-1.0`, which is the rounding step that zeroes the logarithm.
* **Transform/dtype controls** — `jit`, `vmap`, `grad`, `float32` and a `loc=2, scale=3` shift all
  reproduce it; the value is exact in every lane that is supposed to be exact.

## Novelty

Searched the JAX tracker for `gumbel` + `ppf`/`logcdf`/`logsf`, `scipy.stats` `logcdf` tails and
exact-zero boundary values.  Adjacent items found and distinguished:

* **jax-ml/jax#38436** (PR, open) "Handle Gumbel ppf endpoint domains" and **#38318** (issue, open) /
  **#38435** (PR, closed) for `cauchy.ppf` — these are about the *inverse* functions at `q = 0/1`
  (NaN or large finite values at the domain endpoints), not about the CDF tail in log space.
* **jax-ml/jax#38316** (issue, open) + **#40152**/**#38456** (PRs, open) — spurious NaN in
  `chi2.logpdf` at `x = 0, inf` for `df = 2`; a different distribution and a different failure mode
  (`NaN` at a boundary, not a lost tail).
* **jax-ml/jax#38606** (issue, open) — `truncnorm` cdf/sf/logcdf swallow NaN input; again a different
  observable.
* **jax-ml/jax#29343** (PR, closed) added the Gumbel distributions, and the code comment in
  `gumbel_l.logcdf` shows the author deliberately changed the formula to avoid `xlog1p`/`log1p`
  problems - the change fixed the NaN cases but introduced this tail loss, which is not covered by any
  item found.

## Environment

```
jax 0.10.2 / jaxlib 0.10.2, numpy 2.5.3, scipy (see collect_env.txt)
Python 3.12.13, WSL2, CPU and CUDA (RTX 4090)
```

## Anticipated objections and responses

* **"The absolute error is < 1e-18, this is a precision detail."**  It is not a rounding difference
  in a value that is otherwise reproduced: for `x >= 3.7` the function returns the *exact* constant
  `0.0` instead of a representable nonzero number, i.e. 100% relative error, and it also zeroes the
  gradient.  SciPy, and JAX's own `gumbel_r.logsf`, carry the same term correctly.
* **"`logcdf` is allowed to round to 0 at double precision."**  `log1p` exists precisely to avoid
  this; the value is representable (`-1.94e-24` is a normal fp64 number), and the mirror function in
  the same module already returns it exactly.
* **"Only affects an extreme corner of the parameter space."**  The window is `z >= 3.7` for any
  `loc`/`scale`, i.e. four standard deviations of a Gumbel, and it grows with `scale`; it is exactly
  the region where log-space evaluation is used.
