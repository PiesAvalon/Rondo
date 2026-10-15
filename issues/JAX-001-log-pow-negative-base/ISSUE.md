# `jax.jit` returns `0.0` for `log(power(-1.0, 0.5))` where eager returns `nan`

**Component:** JAX 0.10.2, fused XLA lowering of `stablehlo.power` followed by `stablehlo.log`
**Severity:** silent wrong result (a `nan` becomes a finite `0.0`)
**Status:** present in JAX 0.10.2; reproduces on CPU with and without `jit`-fusion of other ops
**Found with:** Rondo program fuzzing (`artifact/hunt-v2/fuzz_programs_jax.py`), reduced to two steps

---

## Summary

`(-1.0) ** 0.5` is not real, so `log` of it is `nan`.  Evaluating the two operations eagerly gives
`nan`; the same jaxpr/HLO inside `jax.jit` gives `0.0`:

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
x = jnp.asarray([-1.0], dtype=jnp.float64)

jnp.log(jnp.power(x, 0.5))                 # [nan]   (eager, per-op XLA dispatch)
jax.jit(lambda t: jnp.log(jnp.power(t, 0.5)))(x)   # [0.0]   <-- wrong
```

The jaxpr and the lowered HLO contain exactly the expected pair of operations - there is no
algebraic simplification at either level:

```text
{ lambda ; a:f64[1]. let b:f64[1] = pow a 0.5:f64[]; c:f64[1] = log b in (c,) }
  %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
  %1 = stablehlo.power %arg0, %0 : tensor<1xf64>
  %2 = stablehlo.log %1 : tensor<1xf64>
```

so the difference comes from how the fused program *evaluates* `power` for a negative base, not from
a rewrite.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| expression at `x = -1.0` | eager | `jit` | |
|---|---|---|---|
| `log(power(x, 0.5))` | `nan` | **`0.0`** | divergence |
| `log(x ** 0.5)` (Python operator) | `nan` | **`0.0`** | same path |
| `log(lax.pow(x, 0.5))` | `nan` | **`0.0`** | same primitive |
| `log(sqrt(x))` | `nan` | `nan` | agrees - it is `power`, not the composition |
| `power(x, 0.5)` alone | `nan` | `nan` | agrees - the rewrite is on the *composite* |
| `log(abs(power(x, 0.5)))` | `nan` | `nan` | agrees once a consumer other than `log` is inserted |
| `log(power(x, 2.0))` | `0.0` | `0.0` | control (integer exponent) |
| `0.5 * log(power(x, 2.0))` | `0.0` | `0.0` | control (integer exponent) |

Both failing forms and both agreeing forms were reproduced identically in two independent processes,
so the result is stable; `x` as a 0-dim scalar, a 1-element vector and a longer vector all show it.

Only **fractional** exponents diverge; integer exponents take a different lowering and agree:

| `log(power(-1.0, c))` | eager | `jit` | |
|---|---|---|---|
| `c = 0.5` | `nan` | **`0.0`** | divergence |
| `c = 1e-8` | `nan` | **`0.0`** | divergence |
| `c = 2.0` | `0.0` | `0.0` | agrees |
| `c = 3.0` | `nan` | `nan` | agrees |
| `c = -1.0` | `nan` | `nan` | agrees |

The jitted value is not arbitrary: it equals `c * log(|x|)`, which is what the simplifier produces
after discarding the (undefined) negative base:

| expression | eager | `jit` | `c * log(abs(x))` |
|---|---|---|---|
| `log(power(-1.0, 0.5))` | `nan` | `0.0` | `0.0` |
| `log(power(-2.0, 0.5))` | `nan` | `0.34657359027997264` | `0.34657359027997264` |
| `log(power(-2.0, 1e-8))` | `nan` | `6.931471805599453e-09` | `6.931471805599453e-09` |

## Root cause

The fold is an **XLA algebraic-simplifier rewrite of the composite**, not a change to `power`:

* `power(x, 0.5)` on its own keeps the `nan` in both modes, so `power` is not re-lowered;
  the simplifier is matching `log(power(A, B))` and replacing it with `B * log(abs(A))`,
  an identity that holds only for `A > 0`.
* The rewrite runs *after* the HLO that `jax.jit(...).lower(...)` shows (that text is unoptimized,
  and it still contains `stablehlo.power` followed by `stablehlo.log`), which is why reading the
  lowered program alone suggests, incorrectly, that no rewrite is involved.
* Passing `XLA_FLAGS=--xla_disable_hlo_passes=algsimp` makes the divergence disappear while leaving
  the integer-exponent rows unchanged, which pins it to the algebraic simplifier:

  ```text
  $ XLA_FLAGS=--xla_disable_hlo_passes=algsimp python repro.py --replays 1
  # log(power(x, 0.5)) at x=-1.0: eager=nan  jit=nan   (bug gone)
  ```

The fix therefore belongs next to the other *unsound-fold* gates in `algebraic_simplifier.cc`
(`openxla/xla#45560` and `openxla/xla#48186` gate `ln(exp(A)) => A`, `sqrt(A*A) => |A|`,
`exp(A)*exp(B)`, `exp(A)/exp(B)`, `A/exp(B)` and `pow(exp(A), B)` behind `enable_fast_math`),
and this rewrite is currently **not** in any of those lists (see the novelty assessment).


## Impact

`log(x ** p)` and `log(pow(x, p))` are ordinary spellings (log-likelihoods, power laws, Box-Cox
transforms, `log(sqrt(...))` written as `log(x**0.5)`).  A `nan` that silently becomes `0.0` is
worse than a `nan`: it is a plausible-looking finite value that will not be caught by
`isfinite`-style checks, and it changes the result of anything downstream.  It also means the same
expression has two different values depending on whether it is jitted, which breaks the usual
assumption that `jit` is semantics-preserving.

## Novelty assessment

* Tracker searches in `jax-ml/jax` for `log pow nan`, `pow negative base jit`, `jnp.power nan jit`
  and `log power simplification` return no matching report.
* The XLA algebraic simplifier is being hardened one unsound identity at a time, each time gated
  behind `enable_fast_math`.  The known, already-gated set (checked 2026-09-11) is:

  | identity | upstream gate |
  |---|---|
  | `ln(exp(A)) => A` | `openxla/xla#45560` (fixes #42732) |
  | `sqrt(A*A) => \|A\|` | `openxla/xla#45560` |
  | `exp(A)*exp(B) => exp(A+B)` | `openxla/xla#48186` (PR #48266, from `tensorflow#123169`) |
  | `exp(A)/exp(B) => exp(A-B)` | `openxla/xla#48186` |
  | `A/exp(B) => A*exp(-B)` | `openxla/xla#48186` |
  | `pow(exp(A), B) => exp(A*B)` | `openxla/xla#48186` |
  | `A - A`, `A / A`, `0 * A` | gated earlier (cited in the `#45560` discussion) |
  | f64 `A/sqrt(B) => A*rsqrt(B)` | `openxla/xla#40844` |

  `log(power(A, B)) => B * log(abs(A))` appears in none of them, and unlike the `#45560`/`#48186`
  rows it is *not* a rewrite of an inverse function pair (`log(exp(..))`) but of `log(pow(..))`.
  A sweep of that family (`artifact/hunt-v2/tools_unsound_folds.py`, 17x17 special-value grid)
  confirms this rewrite is **live** in JAX 0.10.2 while every other live member of the family is
  already attributed upstream - and that the mirror identity `exp(log(A)) => A` is not applied at all.
* The JAX documentation states that `jnp.power` follows NumPy semantics for negative bases with
  fractional exponents (result `nan` in the real dtype), and eager JAX does return `nan` here, so
  the jitted result is the outlier rather than a documented convention.


## Files

* `repro.py` — self-checking reproducer (the divergence, the `sqrt`/`abs`/control rows, two replays,
  CPU and CUDA if available, `--scan`)
* `evidence.txt` — raw observations plus the jaxpr and HLO excerpt
* `collect_env.txt` — environment
* The rewrite is **XLA-level, not JAX-specific**: the same fold is live in TensorFlow 2.21.0's
  XLA path (`tf.function(f, jit_compile=True)`) and reproduces `nan -> 0.0` identically
  (`artifact/hunt-v2/tools_unsound_folds.py --fw tf`).  That matches how the neighbouring gates
  are written - `xla#45560` and `xla#48186` are XLA reports that cite JAX and TF-XLA symptoms -
  so the fix belongs in `algebraic_simplifier.cc` rather than in JAX's own `power` rule.
* A companion sweep of the same family (`tools_unsound_folds.py`) attributes every *other* live
  unsound rewrite in this build to an existing gate, including the two rows that look novel at
  first sight: `exp(A)*exp(-A) => 1` is the gated `exp(A)*exp(B) => exp(A+B)` rule with `B = -A`
  (confirmed by the rewrite firing for independent operands too:
  `exp(1e38)*exp(-3.4e38) -> exp(1e38-3.4e38) = 0.0` instead of `nan`), and `ln(exp(A)) => A`
  is `xla#45560`.  `log(pow(A,B))` is the only live member with no upstream report.
