# `jax.scipy.special.poch` overflows to `inf`/`nan` for arguments whose Pochhammer value is representable

**Component:** JAX 0.10.2, `jax.scipy.special.poch`, float64 (`jax_enable_x64=True`) and float32,
eager and `jax.jit`
**Severity:** silent wrong value (`nan`, or `+inf` for one band) whenever `z + m` crosses the `gamma`
overflow bound, i.e. `z + m >= ~171.62` in float64 and `>= ~34.84` in float32. `poch(200.0, 1.0)`
returns `nan` although `(200)_1 = 200`, and in the default float32 configuration `poch(40.0, 1.0)`
already returns `nan`.
**Status:** present in JAX 0.10.2
**Found with:** a `jax.scipy.special` vs SciPy boundary differential
(`artifact/hunt-v2/tools_jax_special_probe.py`, wave 138), minimised to this function with the probe
shipped next to this file.

---

## Summary

`jax.scipy.special.poch` implements the documented gamma ratio literally:

```python
# jax/_src/scipy/special.py:3190
return jnp.where(m == 0., jnp.array(1, dtype=z.dtype), gamma(z + m) / gamma(z))
```

`gamma` overflows at `171.62` in float64 and `34.84` in float32, so as soon as `z + m` crosses that
bound the ratio becomes `inf / inf = nan` (or `inf / finite = inf` for the first band) even though
`(z)_m` itself is small and well within range:

| call | JAX 0.10.2 | SciPy 1.18.0 | exact |
|---|---|---|---|
| `poch(171.0, 1.0)` (float64) | `inf` | `171.0` | `171` |
| `poch(172.0, 1.0)` (float64) | `nan` | `172.0` | `172` |
| `poch(180.0, 1.0)` (float64) | `nan` | `180.0` | `180` |
| `poch(1000.0, 1.0)` (float64) | `nan` | `1000.0` | `1000` |
| `poch(1e4, 2.0)` (float64) | `nan` | `100010000.0` | `100010000` |
| `poch(1e10, 1.0)` (float64) | `nan` | `1e10` | `1e10` |
| `poch(1e300, 1.0)` (float64) | `nan` | `1e300` | `1e300` |
| `poch(-1.0, 1.0)` | `nan` | `-1.0` | `-1` |
| `poch(-2.0, 3.0)` | `nan` | `0.0` | `0` |
| `poch(35.0, 1.0)` (float32, default config) | `inf` | `35.0` | `35` |
| `poch(40.0, 1.0)` (float32, default config) | `nan` | `40.0` | `40` |

Because `poch(z, 1) == z` and `poch(z, 2) == z(z + 1)` identically, this is not an accuracy question:
the returned value is `nan`/`inf` where SciPy returns the exact finite result.  The identity
`poch(z, 0) == 1` is special-cased and therefore still works above the overflow
(`poch(1e10, 0.0) = 1.0`), which is why only the ratio path is exposed.

`gammaln` is finite throughout the failing range, and
`exp(gammaln(z + m) - gammaln(z))` recovers `199.9999999999852` for the `poch(200.0, 1.0)` case, so
the fix needs no new special functions.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(82 checks per replay, 2 fresh processes.)

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy import special as jsp
import scipy.special as sp

float(jsp.poch(jnp.float64(200.0), jnp.float64(1.0)))   # nan   <-- should be 200.0
float(sp.poch(200.0, 1.0))                                # 200.0
float(jsp.poch(jnp.float64(171.0), jnp.float64(1.0)))    # inf   <-- should be 171.0
```

What the reproducer pins:

| dimension | result |
|---|---|
| cells | 8 float64 overflow cells (`(200,1)`, `(171,1)`, `(172,1)`, `(1e4,2)`, `(1e10,1)`, `(1e300,1)`, `(-1,1)`, `(-2,3)`), each pinned as `nan`/`inf`, as "not the exact value", as SciPy's exact value, and as "disagrees with SciPy" |
| boundary | `poch(x, 1.0) == x` holds for `x <= 170` and fails for `x in {171, 172, 175, 180, 1000, 1e6}`, while SciPy returns `x` everywhere |
| float32 lane | `poch(40.0, 1.0) = nan` and `poch(35.0, 1.0) = inf` in the default configuration (x64 disabled) where SciPy returns `40.0` and `35.0` |
| root cause | `gamma(200.0) = inf` while `gammaln(200.0) = 857.9336698258572`; `gamma(201)/gamma(200) = nan`, `exp(gammaln(201) - gammaln(200)) ≈ 200` |
| graph lane | a retained `jax.jit` callable returns `inf`, `nan`, `nan` on two consecutive calls; the non-overflowing `poch(2.0, 3.0)` is unchanged by jit |
| controls | below the overflow the ratio is accurate to ~1e-13: `poch(2,3) = 23.999999999999993`, `poch(0.5,2) = 0.7500000000000008`, `poch(100,1) = 99.99999999999808`, `poch(170,1) = 170.0000000000085`, `poch(1e10, 0.0) = 1.0` |

## Root cause (pointer)

Direct evaluation of `Gamma(z + m) / Gamma(z)`: both gammas overflow independently, so the quotient
carries no information once either one saturates.  Evaluating it as
`exp(gammaln(z + m) - gammaln(z))` (with the existing `m == 0` short-circuit and a sign/zero guard for
the negative-integer cases SciPy handles) keeps the representable range; a regression test is
`poch(x, 1.0) == x` and `poch(x, 2.0) == x * (x + 1.0)` for `x` in `{170.0, 171.0, 200.0, 1e4}` plus
`poch(200.0, 1.0) == 200.0` in float32 for `x in {35.0, 40.0}`.

## Novelty

* `gh search issues --repo jax-ml/jax "poch"` returns exactly one hit: **jax#27088**, the
  "State of `jax.scipy.special` functions" umbrella.  Its `poch` row reads
  `` `poch` | :+1:/:+1: | tested on R+ `` - it lists no defect for `poch` and does not mention the
  gamma overflow; the umbrella merely asserts the function is tested.
* The campaign closed the rest of the `jax.scipy.special` boundary surface against SciPy in wave 138
  (jax#38519 `expn`/`exp1`/`expi`, jax#37763 `digamma`, jax#38240 `loggamma`, jax#38262 `rel_entr`,
  jax#38270/#39105 `beta`, jax#39106 `betaln`, jax#39107/#39108 `gammainc`, jax#38264 `gammaln`,
  jax#38260 `xlog1py`, jax#2991/#28168 `hyp2f1`); `poch` was recorded as an undrafted candidate
  there, and the overflow reported here (as opposed to the earlier one-point observation) is new.
* The other JAX drafts cover `expn`, `loggamma`/`gammaln` at infinity, `beta` and signed zeros, so
  `poch` is a new site.
