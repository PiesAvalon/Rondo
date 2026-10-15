# Eager mode cannot convert a built-in Python `bool` to `bfloat16`/`complex64`, so `True`/`False` operands fail there while `tf.function` and XLA accept them

**Component:** TensorFlow 2.21.0, CPU: the eager scalar-conversion path (`tf.constant`,
`tf.Tensor.__add__`/`__mul__`/… with a Python `bool`) for `bfloat16` and `complex64`
**Severity:** eager/graph divergence — the same expression raises eagerly and returns a value under
`tf.function`/XLA; `tf.constant(True, tf.bfloat16)` is unusable although every neighbouring dtype
works
**Status:** present in tensorflow 2.21.0 (CPU); NumPy, PyTorch, JAX and MindSpore all accept a Python
bool operand for these dtypes
**Found with:** the wave-186 eager/`tf.function`/XLA dtype-promotion grid, minimised to the
witnesses below; certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

The eager scalar conversion does not know how to turn a built-in Python `bool` into a `bfloat16` (or
`complex64`) tensor, although it handles every other numeric dtype, and although the graph path
converts Python scalars directly to the tensor's dtype:

```python
import numpy as np, tensorflow as tf

tf.constant(True, tf.bfloat16)
# TypeError: Cannot convert True to EagerTensor of dtype bfloat16
tf.constant(True, tf.float16)                          # 1.0        <- works
tf.constant(np.asarray([True, False]), tf.bfloat16)    # [1.0, 0.0] <- works
tf.constant(np.bool_(True), tf.bfloat16)               # 1.0        <- works
tf.cast(tf.constant(True), tf.bfloat16)                # 1.0        <- works (eager, function, XLA)

B = tf.constant([1.0, 2.0], tf.bfloat16)
B + True                                               # raises eagerly
tf.function(lambda: B + True)()                        # bfloat16 [2.0, 3.0]
tf.function(lambda: B + True, jit_compile=True)()      # bfloat16 [2.0, 3.0]
```

## Scope (all with a `True` operand)

| operator | eager (`bfloat16`) | `tf.function` / XLA | eager (`float16` / `float32`) |
|---|---|---|---|
| `+` | `InvalidArgumentError` | `[2.0, 3.0]` | `[2.0, 3.0]` |
| `-` | `TypeError` | `[0.0, 1.0]` | `[0.0, 1.0]` |
| `*` | `TypeError` | `[1.0, 2.0]` | `[1.0, 2.0]` |
| `/` | `TypeError` | `[1.0, 2.0]` | `[1.0, 2.0]` |
| `**` | `TypeError` | `[1.0, 2.0]` | `[1.0, 2.0]` |
| `==` | `TypeError` | `[True, False]` | `[True, False]` |
| `<` | `TypeError` | `[False, False]` | `[False, False]` |

`complex64` behaves the same for `+` (`InvalidArgumentError` eagerly, `[(2+0j), (3+0j)]` under
`tf.function`) and `*`.  `tf.constant(False, tf.bfloat16)` fails identically, and
`tf.constant(True, tf.complex64)` fails as well.  Every other scalar kind works eagerly on
`bfloat16`: `np.bool_(True)`, `np.int8(1)`, `np.uint8(1)`, `1` and `1.0` all give `[2.0, 3.0]`.

## Root cause (pointer)

The eager `PyObject`→`EagerTensor` conversion used for scalar operands and `tf.constant(value,
dtype)` has no `bool`→`bfloat16`/`complex64` entry (it does have `bool`→`float16/32/64`, and it
handles numpy bools through the numpy array path), while the graph tracer converts Python scalars
straight to the operand's dtype.  Adding the missing conversions (or routing the Python bool through
the same `tf.cast` path that already works) removes the divergence; a regression test can assert
`tf.constant(True, tf.bfloat16) == 1.0` and `bfloat16_tensor + True` in both eager and
`tf.function`.

## Impact

Mixed-precision code written against `tf.function` (where `True`/`False` scalars, masks and flags
promote fine) breaks as soon as it runs eagerly — a common situation in unit tests, debugging with
`tf.config.run_functions_eagerly(True)` and eager data pipelines.  `tf.constant(True, tf.bfloat16)`
is also simply unusable, although the equivalent numpy array, numpy scalar and `tf.cast` all work,
which makes the failure look arbitrary.  The value itself is unambiguous (`True` -> `1`, `False` ->
`0`), so no precision judgement is involved.

## Controls (all in `repro.py`)

* `tf.constant(True, tf.float16/float32/float64)` -> `1.0`; `tf.constant(False, tf.float32)` -> `0.0`.
* Numpy bool array -> `bfloat16` `[1.0, 0.0]`; `np.bool_(True)` -> `bfloat16` `1.0`.
* `tf.cast(tf.constant(True), tf.bfloat16)` -> `1.0` eagerly, under `tf.function` and under XLA;
  `tf.cast(bool_array, tf.complex64)` -> `[(1+0j), 0j]`.
* `float16`/`float32` tensors accept `True` eagerly with the correct values.
* `bfloat16` accepts `np.bool_(True)`, `np.int8(1)`, `1` and `1.0` eagerly.

## Novelty

Searched the TensorFlow tracker for `bfloat16 bool scalar eager`, `bfloat16 InvalidArgumentError
True`, `tf.function bool promotion`, `complex64 bool add eager`, `cast bool to bfloat16`,
`bfloat16 bool unsupported`, `Cannot convert True to EagerTensor`, `convert True to EagerTensor of
dtype bfloat16`, `EagerTensor scalar bool bfloat16` and `bfloat16 scalar conversion bool`: no report
exists.  The campaign's TensorFlow drafts (TF-001…015) cover other operators, dtypes and paths, and
none of them mentions `bool` scalar conversion.

## Environment

TensorFlow 2.21.0 CPU wheel, NumPy 2.5.1, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"bool + float tensor is invalid."**  The graph path, `tf.cast` and every other framework accept
  it and produce the obvious `1`/`0` value; TensorFlow itself accepts it for `float16`/`float32` and
  for numpy bools, so the eager path is the outlier rather than the rule being enforced.
* **"It only raises, it does not corrupt data."**  The defect is an eager/graph semantic divergence:
  the identical program works under `tf.function` and fails eagerly (and vice versa for users who
  develop eagerly).  The campaign's other TensorFlow drafts include the same eager-vs-XLA class
  (TF-013, TF-014).
