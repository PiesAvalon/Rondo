# `jax.scipy.special.expn(n, 0.0)` returns `1/n` where the value is `1/(n-1)`

**Component:** JAX 0.10.2, `jax.scipy.special.expn`, CPU, `jax_enable_x64=True` (float32 has the same
defect), eager and `jax.jit`
**Severity:** silent wrong **finite** value for every `n >= 2` — not a NaN, not an infinity, not a
rounding difference. `expn(2, 0.0)` is `0.5` where the value is `1.0`; `expn(3, 0.0)` is
`0.3333333333333333` where the value is `0.5`.
**Status:** present in JAX 0.10.2
**Found with:** a `jax.scipy.special` vs SciPy differential over the boundary grid
(`artifact/hunt-v2/tools_jax_special_probe.py`), minimised to the exact-zero branch.

---

## Summary

`E_n(x) = x^{n-1} ∫_x^∞ e^{-t} t^{-n} dt`, so at `x = 0`

```
E_n(0) = ∫_1^∞ t^{-n} dt = 1/(n-1)          for n >= 2
```

SciPy (`scipy.special.expn`) returns `1/(n-1)` (or exactly `inf` for `n = 1`).  JAX returns `1/n`:

| `n` | JAX `expn(n, 0.0)` | correct `1/(n-1)` | SciPy |
|---|---|---|---|
| 2 | `0.5` | `1.0` | `1.0` |
| 3 | `0.3333333333333333` | `0.5` | `0.5` |
| 4 | `0.25` | `0.3333333333333333` | `0.3333333333333333` |
| 5 | `0.2` | `0.25` | `0.25` |
| 10 | `0.1` | `0.1111111111111111` | `0.1111111111111111` |
| 100 | `0.01` | `0.010101010101010102` | `0.010101010101010102` |
| 1 | `inf` | `inf` | `inf` (control) |

The error is a clean factor `n/(n-1)`, so it is not an accuracy problem: relative error is 50 % at
`n = 2`, 10 % at `n = 10`, 1 % at `n = 100`.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(50 checks per replay; every `n` is pinned on both sides plus SciPy and an adaptive-quadrature
reference, and the neighbours are controls.)

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.scipy.special as jsp, scipy.special as sp

jsp.expn(jnp.float64(2.0), jnp.float64(0.0))   # 0.5   <-- should be 1.0
sp.expn(2, 0.0)                                # 1.0
```

What the reproducer pins:

| dimension | result |
|---|---|
| `n = 2, 3, 4, 5, 10, 100` | every one is `1/n`, never `1/(n-1)` |
| `n = 1`, `n = 0` | `inf`, matching SciPy (controls) |
| exact zero vs near zero | `expn(2, 1e-8) = 0.9999999900000001` and `expn(2, 0.5)`, `expn(2, 1.0)`, `expn(2, 10.0)` all match SciPy to fp64 — **only the `x == 0` branch is wrong** |
| lanes | `jax.jit(jsp.expn)(3.0, 0.0)` gives `0.3333333333333333` as well |
| dtypes | `float32` gives `float32(1/3)`; the defect is not an x64 artefact |
| vector input | `expn([2,3,4], [0,0,0])` gives `[0.5, 1/3, 0.25]` elementwise |
| out-of-domain | `expn(2, -1.0) = nan` (matches SciPy, control) |
| independent reference | `scipy.integrate.quad(lambda u: u**-n, 1, inf)` is `1/(n-1)` to 1e-9 for `n = 2, 3, 10` |

## Root cause (pointer)

`jax/_src/scipy/special.py::expn` (JAX 0.10.2, line ~2981):

```python
conds = [
  (n < 0) | (x < 0),
  (x == 0) & (n < 2),
  (x == 0) & (n >= 2),      # <- this branch
  ...
]
n1 = jnp.where(n == _c(n, 1), n + n, n)
vals = [
  np.nan,
  np.inf,
  one / n1,                 # should be one / (n - 1)
  ...
]
```

`n1` is `n` for every `n != 1`, and `n == 1` never reaches this branch (it takes the `n < 2` branch and
returns `inf`), so the guard only serves to avoid a division that cannot happen and the value is
`1/n`.  Replacing `n1` with `n - 1` (keeping the `inf` branch for `n < 2`) fixes it; a regression test
is `jsp.expn(n, 0.0) == 1/(n-1)` for a few `n >= 2`, cross-checked against `sp.expn`.

## Novelty

* Tracker searches on `jax-ml/jax` for `expn`, `expn zero`, `expn 0`, `expn wrong`, `expn
  precision` return only:
  * [#38519](https://github.com/jax-ml/jax/issues/38519) (open) — "`exp1`/`expn`/`expi` return NaN at
    **infinite** argument". Its title, body and repro all use `x = inf`; its root-cause table writes
    the `x == 0` branch as `one / n1`, i.e. the zero branch was believed correct and is **not** part
    of that report. The wrong finite value at `x = 0` is a separate branch.
  * [#27088](https://github.com/jax-ml/jax/issues/27088) — the umbrella "state of jax.scipy.special"
    tracking issue, whose `expn` row says "tested on R+ (float32 only)". The defect is exactly at the
    boundary `x = 0`, which that testing did not cover.
* The other members of the same boundary family are already filed and were therefore **not** drafted
  from the same sweep: `digamma(±0)` [#37763](https://github.com/jax-ml/jax/issues/37763),
  `loggamma(±0)` [#38240](https://github.com/jax-ml/jax/issues/38240),
  `rel_entr(0, nan)` [#38262](https://github.com/jax-ml/jax/issues/38262),
  `beta` [#38270](https://github.com/jax-ml/jax/issues/38270) /
  [#39105](https://github.com/jax-ml/jax/issues/39105),
  `betaln` [#39106](https://github.com/jax-ml/jax/issues/39106),
  `gammainc`/`gammaincc` at `a = inf`
  [#39107](https://github.com/jax-ml/jax/issues/39107) /
  [#39108](https://github.com/jax-ml/jax/issues/39108),
  `gammaln(-inf)` [#38264](https://github.com/jax-ml/jax/issues/38264),
  `xlog1py(0, nan)` [#38260](https://github.com/jax-ml/jax/issues/38260).
  `expi(0.0) = nan` (SciPy `-inf`) belongs to the same `expi` gap that #38519 describes.
  `poch` (`poch(-1, 1) = nan`, `poch(1e300, 1) = nan`) and `hyp2f1` outside the unit disc are
  recorded as candidates: the first is only covered by the #27088 umbrella, the second is documented
  as unreliable for large intermediate values.
