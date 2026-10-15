# `jnp.modf` returns `nan` for infinite inputs and drops the sign of a zero fractional part for negative integers

**Component:** JAX 0.10.2 (`jax.numpy.modf`) on `float32`/`float64`, eager and `jax.jit`
**Severity:** wrong value — an infinite input produces a `nan` fractional part where the C99/NumPy
result is `0.0`, so a plain `modf` turns a well-defined saturation into a `nan`; the sign of zero is
lost for every negative integer
**Status:** present in JAX 0.10.2; NumPy, `math.modf` and the C99 standard define
`modf(inf) = (0.0, inf)`, `modf(-inf) = (-0.0, -inf)` and `modf(-n) = (-0.0, -n)`
**Found with:** the wave-166 stepping grid (`artifact/hunt-v2/results/w166_*`), triaged by Rondo's
`check_consensus` oracle with `math.frexp`/`math.modf`/`math.ulp` as arbiters; the certified
reproducer is `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`jnp.modf(x)` returns `(fractional, integral)`.  Two classes of inputs are wrong:

```python
import math, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

print(jnp.modf(jnp.float64(math.inf)))    # (nan, inf)     numpy: (0.0, inf)
print(jnp.modf(jnp.float64(-math.inf)))   # (nan, -inf)    numpy: (-0.0, -inf)
print(jnp.modf(jnp.float64(-1.0)))        # (0.0, -1.0)    numpy: (-0.0, -1.0)
print(jnp.modf(jnp.float64(-0.0)))        # (0.0, -0.0)    numpy: (-0.0, -0.0)
```

```text
                      jnp.modf (JAX 0.10.2)   numpy / math.modf
modf( inf)               (nan,  inf)           ( 0.0,  inf)
modf(-inf)               (nan, -inf)           (-0.0, -inf)
modf(-0.0)               ( 0.0, -0.0)          (-0.0, -0.0)
modf(-1.0)               ( 0.0, -1.0)          (-0.0, -1.0)
modf(-2.0)               ( 0.0, -2.0)          (-0.0, -2.0)
modf(-1e308)             ( 0.0, -1e308)        (-0.0, -1e308)
```

Everything else matches NumPy exactly: positive integers, all non-integer values (`+-0.5`, `+-1.5`,
`+-3.75`), subnormals (`+-1e-320`), `nan` -> `(nan, nan)` and every integral part.  The two symptoms
are independent of the others: the `nan` comes from the infinite input, the `+0.0` from the sign of
the fractional part of a negative integer (the integral part keeps its sign correctly).  Both appear
in eager and under `jax.jit`, in scalar and vector form, at float64 and float32.

## Root cause

`jnp.modf` is implemented from the `floor`/`trunc` decomposition of the input (the fractional part
is computed as `x - trunc(x)` and then repaired), which (a) makes `inf - inf = nan` for an infinite
input instead of the special-cased `0.0`, and (b) loses the sign of the zero that `x - trunc(x)`
produces for a negative integer.  The `nan` for `inf` is the serious half: it is a value error in a
function whose whole purpose is to split a number into a bounded and an unbounded part.

## Suggested fix

Special-case the non-finite inputs (`modf(+-inf) = (+-0.0, +-inf)`, `modf(nan) = (nan, nan)`) and
keep the sign of the fractional zero for negative integers (e.g. compute the fractional part as
`x - trunc(x)` in a way that preserves `-0.0`, or `copysign(|x - trunc(x)|, x)` when the difference
is zero).  A regression test can compare against `numpy.modf` over the values above.

## Impact

`modf` is the standard split used by argument-reduction and fixed-point code and by the JAX
implementation of `frexp`-style decompositions.  Passing an infinite value (for example a
log-probability that saturated, or a division result that overflowed) turns a well-defined
`(0.0, inf)` into `(nan, inf)`, and the `nan` then propagates through the fractional branch.  The
signed-zero loss additionally breaks code that tests `signbit(modf(x)[0])` to recover the sign of a
value, and `jnp.frexp`-style round trips.

## Controls (all in `repro.py`)

* `modf(0.0)`, `modf(1.0)`, `modf(2.0)`, `modf(1e308)`, `modf(nan)`, `modf(+-0.5)`, `modf(+-1.5)`,
  `modf(+-3.75)`, `modf(+-1e-320)` all match NumPy at float64.
* The integral parts of the failing inputs are correct, so the defect is confined to the fractional
  component (plus the zero sign).
* The float32 lane and `jax.jit` show the same two symptoms as eager, so this is not a trace-time
  artifact.

## Novelty

Searched the JAX tracker for `modf`, `jnp.modf`, `modf inf`, `modf nan`, `modf negative zero`: no
report of any `jnp.modf` behaviour exists (the only hits are the closed process issues #7570 and
#3689 about listing NumPy API coverage).  The campaign's earlier JAX findings are other functions
(JAX-001 `log(power(-1, 0.5))` under jit, JAX-002 `norm.isf`, JAX-003 `jax.export` weak types,
JAX-004 `expn(n, 0)`, JAX-005 `poch` overflow, JAX-006 `spence` reverse mode), and the 60+ cell
JAX complex signed-zero family recorded in `KNOWN-BOUNDARIES.md` is a different surface.

## Environment

JAX 0.10.2, NumPy 2.5.1, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64, RTX 4090 (CUDA 13.0).
Full dump in `collect_env.txt`; element table in `evidence.txt`.

---

## Anticipated objections and responses

* **"`modf(inf)` is a corner case."**  It is, but it is a *value* error: the correct answer is a
  finite `0.0`, and every other framework and the C standard return it.  An infinite value is the
  normal saturation result of a previous computation, so the `nan` is reachable in ordinary code.
* **"The signed-zero difference is cosmetic."**  Taken alone it would be low severity; it is
  included because it shares the implementation and because `signbit` on the fractional part is a
  real idiom.  The `nan` is the filing argument.
* **"XLA may not define `modf` for inf."**  NumPy (which `jnp.modf` documents itself as mirroring)
  and `math.modf` both define it, and XLA's `Remainder`-style primitives do too; the docstring of
  `jnp.modf` claims NumPy semantics.
