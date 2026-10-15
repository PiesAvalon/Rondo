# `jax.scipy.stats.norm.isf` returns `inf` for every upper-tail probability below ~1e-16 because it is computed as `ppf(1 - q)`

**Component:** `jax.scipy.stats` — `norm.isf`, and the same construction in `cauchy.isf` (JAX 0.10.2, CPU and any backend; `jax.jit` matches eager everywhere, so this is not a compiler issue)
**Severity:** silent wrong value — the inverse survival function, which is how upper-tail quantiles are obtained, becomes `inf` where the correct answer is a finite number, and is already wrong in the fourth digit just above the threshold
**Status:** present in JAX 0.10.2
**Found with:** the `jax.scipy.stats` edge sweep (`artifact/hunt-v2/tools_jax_stats_edge_probe.py`, 58 groups against SciPy); certified reproducer `repro.py` here, `python repro.py --replays 2` -> `RESULT: 0 failing check(s).`

---

## Summary

`isf(q)` should return `x` such that `sf(x) = q`; for the normal distribution `isf(1e-300) = 37.0470963`.
JAX returns `inf` from `q = 1e-17` downwards, and wrong finite values at `1e-15` and `1e-16`:

| `q` | SciPy `norm.isf(q)` | JAX `norm.isf(q)` | JAX `norm.ppf(1-q)` |
|---|---|---|---|
| `1e-10` | `6.361340902` | `6.361340890` | `6.361340890` |
| `1e-15` | `7.941345326` | `7.941444487` (rel. err 1.2e-5) | `7.941444487` |
| `1e-16` | `8.222082216` | `8.209536152` (rel. err 1.5e-3) | `8.209536152` |
| `1e-17` | `8.493793224` | `inf` | `inf` |
| `1e-20` | `9.262340090` | `inf` | `inf` |
| `1e-100` | `21.273453560` | `inf` | `inf` |
| `1e-300` | `37.047096299` | `inf` | `inf` |

The third column is identical to the second at every point, which is the whole story.

## Root cause

`jax/_src/scipy/stats/norm.py` implements the inverse survival function as a subtraction:

```python
def isf(q, loc=0, scale=1):
  return ppf(lax.sub(_lax_const(q, 1), q), loc, scale)
```

For `q < ~1.1e-16`, `1 - q` rounds to exactly `1.0` in float64, and `ppf(1) = inf`.  Below that
threshold the survival probability has been destroyed before the quantile is computed; above it (down
to about `1e-15`) enough of it survives to keep the value finite but not accurate.

The same construction saturates in `cauchy.isf`: `cauchy.isf(1e-300)` returns `1.633123935319537e16`
(the largest value `tan(pi/2)` takes in float64) where the correct quantile is `3.183098861837907e299`.

## Impact

`isf` is the standard way to turn a tail probability into a quantile — significance thresholds,
z-scores from p-values, confidence bounds.  A p-value of `1e-20` becomes `inf` instead of `9.26`, and
`1 / isf(q)` becomes `0`, `isf(q) - isf(q')` becomes `nan`, and so on.  The failure is silent: no
warning, and the value is a plausible-looking infinity.  The reproducer also shows that `isf` no
longer inverts `sf` at all - `norm.sf(norm.isf(1e-20))` is `0.0`, not `1e-20` - while the SciPy
roundtrip is exact.

## Reproduction

```bash
python repro.py --replays 2      # RESULT: 0 failing check(s).
```

17 checks: the seven table rows, the two quantile-accuracy rows, the accurate mid-range control, the
`isf == ppf(1-q)` identity at four points, `logsf`'s accuracy at the correct quantile
(`-logsf(37.0470963) = 690.7755 = -log(1e-300)`, i.e. the building block for a correct implementation
is present), the cauchy saturation, the `sf(isf(q))` roundtrip, and a control (`logistic.isf`, which
is implemented directly, is correct at `1e-300`).

## Novelty assessment

Searches on the JAX tracker (`jax-ml/jax`) for `norm.isf`, `isf inf`, `inverse survival`,
`ppf boundary inf` and `stats nan propagation` return no report about `isf`.  The closest items are
[jax-ml/jax#13291](https://github.com/jax-ml/jax/issues/13291) (open) which asks for *missing* `cdf`
and `ppf` functions, and the meta-issue
[jax-ml/jax#38651](https://github.com/jax-ml/jax/issues/38651) which tracks stats/special edge cases in
general without an `isf` entry.  Reports that do exist for this surface are about *other* functions
and symptoms: #38317 (`uniform.cdf/pdf/logpdf` swallow NaN), #38606 (`truncnorm` swallows NaN), and
#38622 (closed: `poisson.cdf(inf)` was `nan`).

## Proposed fix sketch

Compute the inverse survival directly from the log-survival: `logsf` is accurate at the correct
quantile (`-logsf(37.0470963) = 690.7755`), and for a normal distribution the quantile can be derived
from `log(q)` without forming `1 - q`, which is what SciPy does.  A cheaper patch is a branch for
`q < 0.5 * eps` that uses the logarithmic form; the reproducer's mid-range controls pin the values
that must not change.
