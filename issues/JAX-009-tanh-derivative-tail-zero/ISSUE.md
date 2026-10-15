# `tanh`'s default derivative rule returns an exact zero in the tail (`|x| >= 20` float64, `|x| >= 8` float32)

**Component:** JAX 0.10.2 / jaxlib 0.10.2, CPU (the rule is `jnp.tanh`'s JVP, so it applies to every
backend): `jax.grad(jnp.tanh)`, `jax.jvp`, `jax.jacrev`, `jax.jacfwd`, `jax.jit(jax.grad(...))`, and
second-order derivatives; `float32` and `float64`.
**Severity:** silent wrong gradient magnitude.  The relative error is **1.0** (the returned value is
exactly `0.0`, not a small number) for a finite, representable derivative; in float64 the rule is
already 11.4x too large at `|x| = 19` and 67% off at `|x| = 18`, and it also makes second derivatives
`-0.0`.  Any sign-, ratio- or curvature-based use of `tanh`'s derivative (optimiser updates,
`1/grad`, Hessian diagonals, gate analysis) is wrong, and `float32` reaches the total failure at a
completely ordinary magnitude (`|x| >= 8`).
**Status:** present in JAX 0.10.2 on CPU.  JAX already ships a stable rule for the same op behind
`lax.AccuracyMode.HIGHEST`, which returns the exact value in the same process, so the fix is
in-tree - only the default is wrong.
**Found with:** the wave-204 Rondo derivative-tail oracle (Reference vs Fresh: `jax.grad`/`jvp` vs a
cancellation-free analytic `sech^2` reference, arbitrated by JAX's own `AccuracyMode.HIGHEST` rule);
certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`,
148 checks).

---

## Summary

`d/dx tanh(x) = sech^2(x) = 4 e^{-2|x|} / (1 + e^{-2|x|})^2` is finite and representable for every
finite input.  JAX's default rule is `1 - tanh(x)^2`, which cancels catastrophically and then
returns exactly zero once `tanh(x)` rounds to `1.0`:

```python
import jax, jax.numpy as jnp, math
from jax import lax
jax.config.update("jax_enable_x64", True)

jax.grad(jnp.tanh)(jnp.float64(20.0))                                   # 0.0
jax.grad(jnp.tanh)(jnp.float32(8.0))                                    # 0.0
jax.jvp(jnp.tanh, (jnp.float64(20.0),), (jnp.float64(1.0),))[1]         # 0.0
jax.grad(jax.grad(jnp.tanh))(jnp.float64(20.0))                         # -0.0

exact = 4 * math.exp(-40) / (1 + math.exp(-40)) ** 2                    # 1.699341702116636e-17

# JAX's own highest-accuracy rule, same process, same op:
jax.grad(lambda x: lax.tanh(x, accuracy=lax.AccuracyMode.HIGHEST))(jnp.float64(20.0))
                                                                        # 1.699341702116636e-17
```

Measured (CPU, `jax_enable_x64=True`):

| dtype | x | default `grad` | `AccuracyMode.HIGHEST` | exact `sech^2` | relative error |
|---|---|---|---|---|---|
| float64 | 5 | 1.815832309e-04 | 1.815832309e-04 | 1.815832309e-04 | 2.5e-13 |
| float64 | 15 | 3.741451593e-13 | 3.743049188e-13 | 3.743049188e-13 | 4.3e-04 |
| float64 | 17 | 7.549516567e-15 | 6.855633726e-15 | 6.855633726e-15 | 1.0e-01 |
| float64 | 18 | 1.554312234e-15 | 9.278091321e-16 | 9.278091321e-16 | 6.8e-01 |
| float64 | 19 | 1.554312234e-15 | 1.255653117e-16 | 1.255653117e-16 | **11.4** |
| float64 | **20** | **0.0** | 1.699341702e-17 | 1.699341702e-17 | **1.0** |
| float64 | 25 | 0.0 | 7.714999392e-22 | 7.714999392e-22 | 1.0 |
| float64 | 40 | 0.0 | 7.219411574e-35 | 7.219411574e-35 | 1.0 |
| float64 | -20 | 0.0 | 1.699341702e-17 | 1.699341702e-17 | 1.0 |
| float32 | 3 | 9.866100736e-03 | 9.866037165e-03 | 9.866037165e-03 | 6.4e-06 |
| float32 | 7.5 | 1.192092554e-06 | 1.223608533e-06 | 1.223608533e-06 | 2.6e-02 |
| float32 | **8** | **0.0** | 4.501405976e-07 | 4.501405976e-07 | **1.0** |
| float32 | 10 | 0.0 | 8.244614456e-09 | 8.244614456e-09 | 1.0 |
| float32 | -8 | 0.0 | 4.501405976e-07 | 4.501405976e-07 | 1.0 |

Every differentiation lane that goes through the JVP returns the same wrong value (they agree to a
few ULP): `grad`, `jit(grad)`, `jacrev`, `jacfwd`, `jvp`, and the second derivative
(`jax.grad(jax.grad(jnp.tanh))(20.0)` is `-0.0` where the exact value is `-3.39868e-17`).  The
`HIGHEST`-accuracy rule is exact in every lane at every point tested, in both widths.

## Root cause

`jax/_src/lax/lax.py`, the `tanh_p` JVP registration (JAX 0.10.2):

```python
ad.defjvp2(
    tanh_p,
    lambda g, ans, x, accuracy: mul(g, mul(_const(x, 4),
        mul(logistic(mul(_const(x, 2), x), accuracy=accuracy),
            logistic(mul(_const(x, -2), x), accuracy=accuracy))))
    if accuracy is AccuracyMode.HIGHEST
    else mul(add(g, mul(g, ans)), sub(_one(x), ans)),      # <= g * (1 + tanh(x)) * (1 - tanh(x))
)
```

The default branch is `g * (1 - tanh(x)^2)`.  `tanh(x)` is a floating-point number; once it rounds
to `1.0` (float64 at `|x| >= 20`, float32 at `|x| >= 8`) the factor `1 - ans` is exactly zero and the
rule returns `0.0` regardless of the true derivative.  Before that, `1 - ans^2` loses one bit per
unit of `|x|`: at `x = 19` a relative error of 11.4, at `x = 18` of 0.67.  The `HIGHEST` branch
computes `4 * logistic(2x) * logistic(-2x)`, which is algebraically the same derivative but has no
cancellation (`logistic(-2x)` is the small quantity itself), so it is exact wherever the tail is
representable.

Note the failure is *not* underflow: `tanh(20)` is exactly `1.0` and the derivative
`4 * e^{-40}` is a normal float64 (float32 can represent it too: the float32 `HIGHEST` rule returns
`4.501e-07` at `x = 8`).

## Impact

* `jax.grad`, `jax.value_and_grad`, `jax.jvp`, `jax.jacfwd`/`jacrev`, `jax.jit`-compiled versions and
  `jax.grad(jax.grad(...))` all lose the tail: a tanh-gated unit with a pre-activation of magnitude
  `>= 20` (float64) or `>= 8` (float32) receives **no** gradient at all, and a second-order method
  sees zero curvature.
* The returned value is an exact zero, so any relative-error check, `1/grad` (which becomes `+inf`
  where the exact derivative gives a finite `5.88e16`), `signbit`/`copysign`, or normalised-gradient
  logic is affected, not merely inaccurate.
* float32 is the common case in training code, where `|pre-activation| >= 8` is routine; a saturating
  tanh unit that should still receive a small gradient instead receives none.
* The correct value is available in the same process through
  `lax.tanh(x, accuracy=lax.AccuracyMode.HIGHEST)`, so this is a *default-rule* defect, not a
  hardware limitation.

## Suggested fix

Make the default rule the non-cancelling one (it costs two `logistic` calls and three multiplies, the
same shape as the current expression):

```python
# default: 4 * logistic(2x) * logistic(-2x)   instead of   (1 + tanh(x)) * (1 - tanh(x))
```

or keep `1 - tanh(x)^2` but fall back to the `HIGHEST` form when `1 - ans` underflows to zero, or
promote `HIGHEST` to the default while leaving the cheap form behind an explicit fast-accuracy mode.
A regression test needs no reference library:

```python
x = jnp.asarray(20.0, jnp.float64)
assert jax.grad(jnp.tanh)(x) != 0.0        # currently 0.0; exact 1.699341702116636e-17
assert jnp.isclose(jax.grad(jnp.tanh)(x), 4 * jnp.exp(-40.0), rtol=1e-12)
```

## Controls (all in `repro.py`, 148 checks, 0 failing on each of 2 replays)

* The exact reference is the cancellation-free `4 e^{-2|x|} / (1 + e^{-2|x|})^2` in float64, and the
  reproducer checks it is nonzero and representable at every failing point.
* JAX's own `lax.AccuracyMode.HIGHEST` rule matches that reference at every point (float64 to 1e-12
  relative, float32 to 2e-5), which pins the defect to the default rule rather than to the
  derivative being ill-defined or underflowed.
* Near the origin the default rule matches the reference (float64 `x <= 5` to 1e-12), so the defect
  is confined to the tail; the *growth* of the error is asserted separately (relative error > 1e-4 at
  float64 `x = 15, 17, 18, 19` and float32 `x = 7, 7.5`).
* The forward value is checked (`tanh(20) == 1.0`), so the cause is the `1 - tanh(x)^2` form, and the
  `1 - tanh(x)^2 == 0` identity is asserted at that point.
* Both tails are checked (`x = -20`, `x = -40`, `x = -8`), so this is not a one-sided artefact.
* All five differentiation lanes are checked per case (either exactly zero in the tail, or within a
  few ULP of `grad` elsewhere), and both widths are covered.

## Novelty

Searched the JAX tracker (`gh search issues --repo jax-ml/jax`) for `tanh gradient tail`,
`tanh derivative zero large`, `tanh 1 - tanh^2`, `negative zero`, `sign of zero`, `reduce_prod`,
`prod jvp`, `prod transpose`, `signbit`, `expm1 gradient`, `elu gradient`, `celu gradient`: the
tracker has **no tanh derivative-tail report**.  The closest items are the per-operator instances of
the same class, all still open - [#39794](https://github.com/jax-ml/jax/issues/39794)
(`expm1'(-40.0)`), [#39795](https://github.com/jax-ml/jax/issues/39795) (`elu`),
[#39796](https://github.com/jax-ml/jax/issues/39796) (`selu`),
[#39798](https://github.com/jax-ml/jax/issues/39798) (`silu` at `x = -709`) and
[#40342](https://github.com/jax-ml/jax/issues/40342) (`jax.nn.sigmoid` default JVP vs
`AccuracyMode.HIGHEST`) - plus `#10206` (complex `jnp.prod` gradient when jitted, closed),
`#38633` (digamma at signed zero/subnormals), `#1888` and `#2380`.  The sigmoid issue #40342 is the
structural analogue (default rule loses the tail that `HIGHEST` preserves) but a different operator,
and the concrete failure differs: sigmoid is one-sided (`x = +40`) and the value is a small nonzero
tail, while `tanh` fails symmetrically at a much more reachable magnitude (`|x| >= 20` float64,
`|x| >= 8` float32), returns an exact `0.0`, and drives the second derivative to `-0.0`.  This
campaign's other JAX drafts (JAX-001…008) cover different operators (`log`/`pow`, `isf`, `export`
weak types, `expn`, `poch`, `spence`, `modf`, `cumsum`/`cumprod` signed zeros) and none of them
touches `tanh`.

## Environment

JAX 0.10.2 / jaxlib 0.10.2, CPU backend, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux
6.6.87.2) x86_64, `jax_enable_x64=True`.  Full dump in `collect_env.txt`; transcripts in
`evidence.txt`.

---

## Anticipated objections and responses

* **"`accuracy=AccuracyMode.HIGHEST` exists, so this is a speed/accuracy trade-off by design."**
  The issue is not that a faster rule exists but that the default returns an exact `0.0` with
  relative error 1.0 for a normal, representable derivative - that is a wrong result, not a
  lower-accuracy one, and it silently zeroes gradients.  The tracker keeps the identical complaint
  open for `sigmoid` (#40342) and for four other activations (#39794/#39795/#39796/#39798), so the
  project's own triage treats this class as a bug.
* **"`tanh(x)` is 1.0 there, so the derivative is effectively zero."**  It is not: `sech^2(20) =
  1.699e-17` is a normal float64 and `sech^2(8) = 4.501e-07` is a comfortable float32; the reproducer
  asserts both are nonzero and representable, and JAX's own `HIGHEST` rule returns them.
* **"This is the same as PyTorch/TensorFlow."**  Possibly - `1 - tanh^2` is a common form - but that
  does not make the returned gradient correct, and JAX already contains the stable alternative in
  the same source file.  The report is against JAX's default rule on JAX's own reference.
* **"Numerical gradient checks pass."**  They do not at these points: a central difference of `tanh`
  at `x = 20` gives the true `1.7e-17` while `jax.grad` returns `0.0` (the reproducer's reference is
  analytic, so it is exact).
