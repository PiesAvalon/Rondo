# Grappler's constant folding rewrites `x + 0.0` to `x`, so `-0.0` stays `-0.0` instead of the IEEE `+0.0` (and XLA does the same independently)

**Component:** TensorFlow 2.21.0, CPU.  The GraphDef optimiser path: `tf.config.optimizer.set_experimental_options({"constant_folding": ...})`, materialised through `tf.function`; and the XLA path, `tf.function(..., jit_compile=True)`.  `float32` and `float64`.
**Severity:** silent wrong value, compile-mode dependent.  `(-0.0) + (+0.0)` is `+0.0` in IEEE 754-2019 (and in eager TensorFlow and NumPy) but comes back as `-0.0` from an optimised graph.  The bit pattern reaches the user, and every sign-sensitive consumer takes the opposite branch: `tf.math.reciprocal(x + 0.0)` is `+inf` eagerly and `-inf` under `tf.function`/XLA, `tf.math.negative(x + 0.0)` flips from `-0.0` to `+0.0`, `tf.math.sign(x + 0.0)` is `+0.0` eagerly and `-0.0` under XLA.  A value comparison cannot see it (`+0.0 == -0.0`).
**Status:** present in TensorFlow 2.21.0 (CPU), both in `tf.function` and under `jit_compile=True`.  Grappler's constant-folding pass is responsible for the graph-mode half; XLA's algebraic simplifier performs the same rewrite and survives every Grappler switch.
**Found with:** the wave-272 TF control-flow / graph-special-value hunt (Rondo Reference/Fresh/Persistent oracles: eager reference, `tf.function` persistent callable, freshly XLA-compiled callable, with every lane's *input bits* verified), minimised to the single-element witness below; certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 257 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf

x = np.array([-0.0])                      # bits verified as -0.0 in every lane below

# eager reference (also NumPy, also IEEE 754-2019)
t = tf.convert_to_tensor(x)
t + tf.constant(np.array([0.0]))          # +0.0

# the same addition inside a graph
f = lambda x: x + tf.constant(np.array([0.0]))
tf.function(f)(x)                         # -0.0    <- Grappler constant folding
tf.function(f, jit_compile=True)(x)       # -0.0    <- XLA algebraic simplifier
```

IEEE 754-2019: the sum of two zeros is `+0.0` when their signs **differ** and keeps the sign when
they are the **same**, so `(-0.0) + (+0.0) = +0.0` while `(-0.0) + (-0.0) = -0.0`.

| program (input `-0.0`) | eager | `tf.function` | XLA |
|---|---|---|---|
| `x + 0.0` | `+0.0` | **`-0.0`** | **`-0.0`** |
| `0.0 + x` | `+0.0` | **`-0.0`** | **`-0.0`** |
| `(x + 0.0) * 1.0` | `+0.0` | **`-0.0`** | **`-0.0`** |
| `x + tf.zeros_like(x)` | `+0.0` | **`-0.0`** | **`-0.0`** |
| `(x + 0.0) + 0.0` | `+0.0` | **`-0.0`** | **`-0.0`** |
| `x + (-0.0)` (control) | `-0.0` | `-0.0` | `-0.0` |
| `x - 0.0`, `x * 1.0`, `x + x`, `-x`, `abs(x)` (controls) | equal in all three lanes | | |

The defect holds for every width tested (`n = 1, 2, 3, 4, 8, 16, 31, 32, 64, 100`) and in mixed
tensors only the negative zeros flip: `[-0.0, 1.0, -0.0, 0.0] + 0.0` is
`[+0.0, 1.0, +0.0, 0.0]` in eager but `[-0.0, 1.0, -0.0, 0.0]` in the graph lanes.

## Root cause

The traced GraphDef contains the addition itself (`AddV2(x, Const(0))`); the rewrite happens in the
optimiser.  Attribution was done by disabling optimiser groups one per process, with the option set
*before* the first trace (the option has no effect once a function has been traced):

| option | `tf.function` | XLA |
|---|---|---|
| (default) | **`-0.0`** | **`-0.0`** |
| `{"constant_folding": False}` | `+0.0` | **`-0.0`** |
| `{"disable_meta_optimizer": True}` | `+0.0` | **`-0.0`** |
| `{"arithmetic_optimization": False}` | **`-0.0`** | **`-0.0`** |

`arithmetic_optimization`, `function_optimization`, `dependency_optimization`, `remapping`,
`layout_optimizer`, `implementation_selector`, `disable_model_pruning`, `memory_optimization`,
`shape_optimization`, `loop_optimization` and `scoped_allocator_optimization` were each disabled in
turn and changed nothing, so the graph-mode half is **Grappler's constant-folding pass**, whose
`SimplifyArithmeticOperations` forwards one input of an `Add` with a zero constant
(`tensorflow/core/grappler/optimizers/constant_folding.cc`, the `ReplaceOperationWithIdentity`
calls for the additive identity).  That rewrite is valid for the real numbers but not for IEEE
signed zeros.

The XLA half is unaffected by every Grappler switch, so it is XLA's own algebraic simplifier doing
the same `x + 0 → x` rewrite on HLO.

## Impact

* **Silent and comparison-invisible.**  The returned bits differ while `+0.0 == -0.0` is true, so a
  test that compares values passes; only a `signbit` inspection or serialisation sees it.
* **The sign of an infinity flips**: `tf.math.reciprocal(x + 0.0)` is `+inf` eagerly and `-inf`
  under `tf.function`/XLA.  Branch cuts (`sqrt`, `log`, `atan2`, complex phase) and `copysign`
  consumers inherit the flip.
* **Mode-dependent results for identical code**: the same line of Python returns different bits
  depending on whether the surrounding function is traced or executed eagerly, so a model can train
  eagerly and misbehave once wrapped in `tf.function` (or the reverse).
* **Common idiom**: `x + 0.0` / `x + tf.zeros_like(x)` appears in bias initialisation, masking and
  dtype-promotion code, and `tf.zeros_like` is exactly the pattern exercised above.

## Suggested fix

Guard the additive-identity rewrite on the *sign* of the zero constant and the dtype, exactly as the
multiplicative case was fixed for Inf/NaN (tensorflow#119428): only fold `Add(x, 0)` when `0` is a
positive zero **and** the rewrite cannot change a result, e.g. keep the node when the input may be
`-0.0`, or restrict the fold to integral dtypes.  XLA's algebraic simplifier needs the same guard
(its `IsNonNegative`/`IsAllNegativeZero` machinery already exists for the `sqrt(x)*sqrt(x)` case,
tensorflow#70973).  A regression test:

```python
x = np.array([-0.0])
assert np.signbit(tf.function(lambda t: t + 0.0)(x).numpy()[0]) == False   # currently True
```

## Novelty

Searches on the TensorFlow tracker for `constant folding negative zero`, `grappler -0.0`,
`tf.function signed zero`, `ArithmeticOptimizer negative zero`, `SimplifyArithmeticOperations` and
`-0.0 add` return no report of the additive identity rewrite.  The nearest items, all disclosed:

* **tensorflow#119428** (closed 2026-08-19) - *"Grappler constant folding optimizes `x * 0` → `0`,
  suppressing NaN for Inf/NaN inputs"*: the **same pass and the same class of bug** for the
  *multiplicative* identity.  This draft was written after verifying that the multiplicative case is
  fixed in this build (`x * 0.0` with NaN/Inf is preserved by `tf.function` here) while the additive
  identity still breaks IEEE signed zeros - i.e. the earlier fix did not cover this sibling rule.
* **tensorflow#70973** (closed, PR #14478) - *"IsNonNegative(constant(-0.0)) should be false"*: an XLA
  simplifier fix for `sqrt(x)*sqrt(x) → x` and `pow(x, 0.5) → sqrt(x)` at `-0.0`, a different rewrite
  and a different pass, and it explicitly does not cover `x + 0`.
* **tensorflow#122663** (closed) - XLA non-negative inference for *signed integers*: dtype-specific,
  unrelated to signed zeros.
* This campaign's TF-003/TF-015 are eager/`cumsum` signed-zero defects, and PT-020/PT-022 are the
  PyTorch compilers' signed-zero defects; none is the TensorFlow graph optimiser's additive identity.

## Environment

TensorFlow 2.21.0 (CPU), oneDNN enabled, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The difference is only a sign of zero."**  The sign of a zero is part of the IEEE result and is
  observable: it flips the sign of `1/x`, of `log`, of branch cuts and of `copysign`, and it changes
  serialised bytes.  This campaign has filed the same class for PyTorch (`torch.maximum`/`minimum`,
  the Inductor reductions) and TensorFlow (`tf.math.cumsum`, `tf.math.maximum`) signed zeros.
* **"Eager and graph are allowed to differ in optimisations."**  Grappler's rewrites are supposed to
  preserve program semantics; `x + 0.0` is not the identity on `-0.0`, exactly as `x * 0` is not the
  constant `0` on Inf/NaN - the reason tensorflow#119428 was accepted and fixed.
* **"Maybe the input was already `+0.0` in the graph."**  Every lane returns its own `tf.bitcast` of
  the received input, and the repro asserts those bits are `0x8000000000000000` (`-0.0`) before
  accepting any divergence; the controls (`x * 1.0`, `x + x`, `x - 0.0`) also show the graph lane
  receiving `-0.0`.  Along the way this ruled out a first, wrong reading in which eager's documented
  scalar-constant cache (this campaign's TF-011) made the *eager* operand positive.
* **"Is the XLA half just Grappler again?"**  No: `disable_meta_optimizer` restores `+0.0` in
  `tf.function` while the XLA result stays `-0.0`, so two independent implementations apply the
  rewrite.
