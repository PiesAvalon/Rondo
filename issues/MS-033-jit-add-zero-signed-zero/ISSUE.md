# `ms.jit` constant-folds `x + 0.0` to `x`, so a `-0.0` input stays `-0.0` instead of the IEEE `+0.0`

**Component:** MindSpore 2.9.0, CPU.  The graph compiler (`ms.jit`, and `GRAPH_MODE`, at `jit_level` O0, O1 and O2) versus PYNATIVE, for `float32` and `float64`; the reference is PYNATIVE and NumPy.
**Severity:** silent wrong value, compile-mode dependent.  `(-0.0) + (+0.0)` is `+0.0` in IEEE 754-2019, in PYNATIVE and in NumPy, but comes back as `-0.0` from the compiled graph whenever the zero is a **constant** rather than an argument.  The bit pattern reaches the user and every sign-sensitive consumer takes the opposite branch: `ops.reciprocal(x + 0.0)` is `+inf` in PYNATIVE and `-inf` under `ms.jit`.  A value comparison cannot see it (`+0.0 == -0.0`).
**Status:** present in MindSpore 2.9.0 (CPU).  The same addition with the zero supplied as a *function parameter* is correct in both modes (a captured `Tensor(0.0)` is not), which localises the defect to graph constant folding rather than to the `Add` kernel.
**Found with:** the wave-291 MindSpore graph-rewrite audit (identity / inverse / cancellation expressions at special values, PYNATIVE vs `ms.jit`; the only structural divergence in the whole grid), minimised to the single-element witness below; certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 168 checks).

---

## Summary

```python
import numpy as np, mindspore as ms
from mindspore import Tensor

x = Tensor(np.array([-0.0]))            # the reproducer reads the sign bit back in both modes

def f(t):
    return t + 0.0                      # the zero is a Python float -> a graph constant

ms.set_context(mode=ms.PYNATIVE_MODE)
f(x).asnumpy()                          # +0.0     (IEEE, NumPy)

ms.set_context(mode=ms.PYNATIVE_MODE)   # ms.jit switches internally
ms.jit(f)(x).asnumpy()                  # -0.0     <- the graph constant-folds Add(x, 0.0) to x

# every *constant* spelling is affected: 0.0 + x, ops.add(x, 0.0), (x * 1.0) + 0.0 and a captured
# module-level Tensor(0.0); the zero supplied as a *function parameter* is correct in both modes
ms.jit(f_add_zero_arg)(x, Tensor(np.array([0.0])))  # +0.0
```

IEEE 754-2019: the sum of two zeros is `+0.0` when their signs **differ** and keeps the sign when
they are the **same**, so `(-0.0) + (+0.0) = +0.0` while `(-0.0) + (-0.0) = -0.0`.

| program (input `-0.0`) | PYNATIVE | `ms.jit` | NumPy |
|---|---|---|---|
| `x + 0.0` (constant zero) | `+0.0` | **`-0.0`** | `+0.0` |
| `0.0 + x` (commuted) | `+0.0` | **`-0.0`** | `+0.0` |
| `ops.add(x, 0.0)` | `+0.0` | **`-0.0`** | `+0.0` |
| `(x * 1.0) + 0.0` | `+0.0` | **`-0.0`** | `+0.0` |
| `x + captured Tensor(0.0)` | `+0.0` | **`-0.0`** (fp32 lane) | `+0.0` |
| `x + 0.0` (zero as an argument) | `+0.0` | `+0.0` | - |
| `1/(x + 0.0)` | `+inf` | **`-inf`** | `+inf` |
| `x + (-0.0)` (control) | `-0.0` | `-0.0` | `-0.0` |
| `x - 0.0`, `x * 1.0`, `x + x`, `-(-x)` (controls) | `-0.0` | `-0.0` | `-0.0` |
| `-x`, `ops.abs(x)` (controls) | `+0.0` | `+0.0` | `+0.0` |
| `(+0.0) + 0.0` | `+0.0` | `+0.0` | `+0.0` |

The defect reproduces in float32 and float64 for widths 1, 2, 4, 16 and 64, in scalar and vector
shapes, at every `jit_level`, and after 200 clean calls on the same compiled callable (so it is a
property of the translated graph, not of a cold compile).

## Root cause

MindSpore's graph constant folding evaluates `Add(x, const 0.0)` as an algebraic identity and
forwards `x`.  That is valid for the real numbers but not for IEEE signed zeros: the rewrite removes
the operation whose whole result, on a `-0.0` input, is `+0.0`.  The evidence that this is the
folder and not the addition kernel:

* with the zero passed as a **function parameter** (a runtime tensor argument), PYNATIVE and
  `ms.jit` both return the IEEE `+0.0` - the same `Add` node computed by the same runtime;
* with the zero as a **constant** - a Python float, `ops.add(x, 0.0)`, or a *captured* module-level
  `Tensor(0.0)` - only the compiled path is wrong, and the commuted `0.0 + x` and the composed
  `(x * 1.0) + 0.0` are wrong too;
* the input's sign bit is read back through each mode before the comparison (an identity function
  under `ms.jit` returns the `-0.0` bits), so the divergence is in the result, not in the input;
* every other identity/inverse/cancellation expression in the wave-291 grid agrees between the two
  modes, and all the safe controls above agree, so this is not a general graph-mode signed-zero
  difference.

## Impact

* **Silent and comparison-invisible.**  `+0.0 == -0.0` is true, so value-based tests pass; only a
  `signbit` inspection, a serialisation, or a downstream consumer sees it.
* **The sign of an infinity flips**: `1/(x + 0.0)` is `+inf` in PYNATIVE and `-inf` under `ms.jit`.
  Branch cuts (`sqrt`, `log`, `atan2`, complex phase) and `copysign` consumers inherit the flip.
* **Mode-dependent results for identical code**: the same line of Python returns different bits
  depending on whether the surrounding function is compiled, so a model can behave differently in
  PYNATIVE and in graph mode without any error.
* **Common idiom**: `x + 0.0` appears in bias initialisation, masking, dtype-promotion and
  "materialise a value" code, and the constant form is the most common spelling.

## Suggested fix

Do not forward an `Add` with a zero operand when it can change the sign of a zero: either keep the
node, or fold only when the input is provably not a negative zero (e.g. restrict the identity to
integral dtypes, or emit `x + 0.0` for floating point).  The runtime-argument path already behaves
correctly, so the fix is local to the graph constant-folding pass.  A regression test:

```python
x = ms.Tensor(np.array([-0.0]), ms.float32)
assert np.signbit(ms.jit(f)(x).asnumpy()[0]) == False    # currently True
```

## Novelty

Searches on the MindSpore Gitee tracker (`mindspore/mindspore`, API v5 issue search) for
`-0.0`, `signed zero`, `negative zero`, `符号零`, `负零`, `加法 符号`, `常量折叠 零`,
`constant folding`, `add zero graph`, `jit -0.0` and `float sign zero` return no report of the
additive-identity fold.  The nearest items are old and unrelated: **I7S7KP** (closed, 2023,
"MindSpore's basic add and multiply operators have bugs", a generic accuracy report with a zip
attachment, no signed-zero content in the visible body), **I5H5PP** (`Qr` output sign, a different
operator) and **I28LGD**/**I3ZBVE** (constant-folding *features* for PYNATIVE and the Lite
converter).  Nothing describes graph-mode signed-zero behaviour.

This is the same *class* as the campaign's TF-021 (Grappler's constant folding folds `x + 0.0` and
XLA does the same) but a **different framework and a different compiler**: MindSpore's own graph
compiler, with its own constant-folding pass, its own `jit_level` settings and no XLA/LLVM
involvement (`ms.jit` on CPU uses MindSpore's graph IR).  Both are filed as separate drafts; the
cross-framework agreement that `x + 0` is not an identity on IEEE negatives is supporting evidence
for either report, not a duplicate.

## Environment

MindSpore 2.9.0 (CPU), Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64.  Full dump in
`collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The difference is only a sign of zero."**  The sign of a zero is part of the IEEE result and is
  observable: it flips the sign of `1/x`, `log`, branch cuts and `copysign`, and it changes
  serialised bytes.  This campaign has filed the same class for TensorFlow (`cumsum`, `maximum`),
  PyTorch (the Inductor reductions) and MindSpore (`fmod`/`remainder`, `copysign`) signed zeros.
* **"Eager and graph are allowed to differ."**  MindSpore documents graph mode as executing the same
  semantics, and the runtime-argument form of the very same addition is correct in both modes - the
  rewrite, not the mode, is what changes the answer.
* **"Maybe the graph received `+0.0`."**  The reproducer runs an identity function through each mode
  and asserts the round-tripped input's sign bit is set (`-0.0`) before accepting any divergence;
  the safe controls (`x * 1.0`, `x + x`, `x - 0.0`) also show the graph lane receiving `-0.0`.
* **"Is this a duplicate of TF-021?"**  No: different project, different compiler, different
  constant-folding implementation.  The drafts disclose each other.
