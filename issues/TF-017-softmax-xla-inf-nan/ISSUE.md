# `tf.nn.softmax` under XLA returns all-NaN for float32 logits containing `+inf`

**Component:** TensorFlow 2.21.0, CPU.  `tf.nn.softmax` (float32) under
`tf.function(..., jit_compile=True)` (XLA) vs eager and `tf.function` without XLA.
**Severity:** silent wrong values - one `+inf` logit turns **every** element of the affected
softmax row/column into NaN under XLA, while eager, `tf.function`, and the formula TensorFlow
documents for this op all return `NaN` only at the infinite position and exact `0` elsewhere.
**Status:** present in TensorFlow 2.21.0 (CPU); float64 does not show it, `tf.nn.log_softmax` does
not show it, and finite tensors are identical across the three lanes.
**Found with:** the wave-192 Rondo special-value grid for the softmax family (softmax /
log_softmax / logsumexp over `±inf`, all-`-inf` rows, `±inf` mixtures and large-magnitude inputs),
Reference oracle = eager, Fresh oracle = `tf.function(jit_compile=True)`, plus a written-out
reference formula; certified reproducer is `repro.py` in this directory (`--replays 2` ->
`RESULT: 0 failing replay(s).`, 48 checks).

---

## Summary

TensorFlow documents `tf.nn.softmax` as a formula:

```python
def softmax(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)
```

With a `+inf` logit that formula is well defined: the infinite position computes `inf/inf = NaN`, and
every finite position computes `finite/inf = 0`.  Eager and `tf.function` do exactly that; under XLA
the same op returns NaN everywhere in the row:

```python
import tensorflow as tf
x = tf.constant([[float("inf"), 0.0, -1.0, -2.0]])

tf.nn.softmax(x)                                       # [[nan, 0., 0., 0.]]      correct
tf.function(tf.nn.softmax)(x)                          # [[nan, 0., 0., 0.]]      correct
tf.function(tf.nn.softmax, jit_compile=True)(x)        # [[nan, nan, nan, nan]]   <-- wrong

# the documented formula, compiled by the same XLA, gives the eager answer again:
tf.function(lambda z: tf.exp(z) / tf.reduce_sum(tf.exp(z), axis=-1, keepdims=True),
            jit_compile=True)(x)                       # [[nan, 0., 0., 0.]]
```

Measured cases (`eager` / `XLA`):

| input | eager | XLA |
|---|---|---|
| `[inf, 0, -1, -2]` | `[nan, 0, 0, 0]` | **`[nan, nan, nan, nan]`** |
| `[inf, -inf, 0, 1]` | `[nan, 0, 0, 0]` | **`[nan, nan, nan, nan]`** |
| `[0, inf, 1, 2]` | `[0, nan, 0, 0]` | **`[nan, nan, nan, nan]`** |
| `[[1,2,3],[inf,0,-1]]` | `[[.090,.245,.665],[nan,0,0]]` | `[[.090,.245,.665],[nan,nan,nan]]` (row-local) |
| `[[inf,0],[1,2]]`, `axis=0` | `[nan,.119,0,.881]` | `[nan,.119,nan,.881]` (column-local) |
| `[inf] + [0]*127` | `[nan, 0, ...]` | **all NaN** |
| all `+inf` | all NaN | all NaN (agrees) |
| all `-inf` | all NaN | all NaN (agrees) |
| finite values | finite | finite, same within 1e-6 |

So the divergence is **XLA-only**, **float32-only**, **softmax-only** (not `log_softmax`), and
**row/column-local**: other rows of the same tensor are untouched.

## Root cause

The XLA lowering of `tf.nn.softmax` uses the numerically stable max-subtracted form
`exp(x - max) / sum(exp(x - max))`.  With `max = +inf` that becomes `inf - inf = NaN` at the infinite
position and `finite - inf = -inf` (so `exp(...) = 0`) elsewhere; the NaN in the denominator then
propagates to every element of the row.  Eager's float32 kernel (and the formula in the docstring)
computes `exp(x) / sum(exp(x))`, where the finite numerators are divided by `inf` and give `0`.
Writing the documented formula out by hand and compiling it with the same XLA reproduces the eager
result, which shows the arithmetic is available - the difference is the selected softmax lowering.
Note that for **float64** both lanes use the stable form and both return all-NaN, so float32 eager is
the only lane that follows the naive documented formula.

## Impact

A single `+inf` logit - from an overflowing feature, an infinite weight, or a genuinely infinite
value produced upstream - silently converts the whole affected softmax row to NaN **only** when the
model is compiled with `jit_compile=True`.  Attention/classification code that runs correctly in
eager mode and under TF1-style graph execution therefore produces NaN after switching to XLA, and
the failure is data-dependent and silent (no error, no warning).  The same input is also treated
differently for float32 vs float64 within eager itself.

## Suggested fix

Make the two paths agree, in either direction:

* guard the max subtraction when the row maximum is `+inf` (fall back to the documented
  `exp(x) / sum(exp(x))` for those rows), or
* if all-NaN is the intended convention, implement it in eager (and update the docstring, which
  states the `exp/sum` formula, and the float32/float64 split).

A test with one `+inf` logit asserting that XLA and eager agree would pin either choice.

## Controls (all in `repro.py`, 48 checks)

* Finite tensors are identical between eager, `tf.function` and XLA within 1e-6.
* An all-`-inf` row is all-NaN in all three lanes (the framework *does* define that case, which is why
  the `+inf` divergence is not simply "undefined input").
* All-`+inf` and mixed `±inf` rows agree.
* `tf.nn.log_softmax` agrees between eager and XLA (all NaN) for the same inputs.
* The divergence needs float32: for float64 eager and XLA both return all-NaN.
* The documented formula, evaluated eagerly and under XLA, matches eager softmax on every case
  (structurally and within 1e-6), so the reference is not a self-fulfilling one.
* The finite row of a 2-D input that also contains a `+inf` row is unaffected under XLA.

## Novelty

Searched the TensorFlow tracker for `softmax inf XLA`, `tf.nn.softmax jit_compile nan`,
`softmax infinity jit_compile` and `XLA softmax +inf`: the adjacent open reports are
[#126005](https://github.com/tensorflow/tensorflow/issues/126005)
(`sparse_softmax_cross_entropy_with_logits` returns NaN under `jit_compile=True` while eager raises),
[#126893](https://github.com/tensorflow/tensorflow/issues/126893) (`reduce_prod` over an array
containing zero returns NaN under XLA once the array is large enough) and
[#62519](https://github.com/tensorflow/tensorflow/issues/62519) (an XLA softmax/reduce_sum
transposed-output inconsistency).  They are the same *class* (XLA returning NaN where eager does not)
for different ops; none covers `+inf` logits in `tf.nn.softmax`.  The campaign's other TensorFlow
drafts are unrelated (TF-011 eager scalar cache, TF-013/TF-014 complex overflow, TF-015 cumsum signed
zero, TF-016 bool scalar conversion).

## Environment

TensorFlow 2.21.0 (CPU, XLA:CPU via `tf.function(jit_compile=True)`), NumPy 2.5.1, Python 3.12.13,
WSL2 (Linux 6.6.87.2) x86_64.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Softmax with `+inf` logits is undefined."**  TensorFlow defines it: the docstring gives the
  `exp/sum` formula, eager implements it, the `-inf` case is handled consistently in all lanes, and
  the hand-written formula under the *same* XLA produces the eager answer.  The report is about two
  lanes of one framework disagreeing on a value the framework already computes.
* **"All-NaN is the mathematically correct limit."**  That is what torch and JAX return, so it may be
  the preferred convention - but then float32 eager and the docstring are the inconsistent ones, and
  float64 exists as a third behaviour.  Either resolution is fine; the current split is not.
* **"Only degenerate inputs are affected."**  True, and the severity reflects that.  It is still a
  silent, data-dependent NaN that appears only after enabling `jit_compile=True`, which is exactly
  the class of bug a differential oracle should surface; the framework's own eager path is the
  reference that says the user's model was correct before compilation.
