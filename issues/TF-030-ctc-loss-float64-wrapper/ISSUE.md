# `tf.nn.ctc_loss` rejects float64 logits although the wrapped `CTCLoss` op documents float64 support

**Component:** TensorFlow 2.21.0 (CPU), `tf.nn.ctc_loss` (dense-label path), for `float64` logits.
**Severity:** the documented dtype is unusable.  `tf.raw_ops.CTCLoss` states *"inputs: A `Tensor`. Must
be one of the following types: `float32`, `float64`"*, and the float32 wrapper path is correct, but
passing float64 logits raises an internal dtype-mismatch error instead of computing the loss:

```
InvalidArgumentError: cannot compute Mul as input #1(zero-based) was expected to be a double tensor
but is a float tensor [Op:Mul]
```

for both `logits_time_major=True/False` and with or without `blank_index`.  The failure is an internal
graph error from the Python wrapper (a `Mul` mixing the float64 logits with a float32 constant), not a
validation error, so the user gets no indication that the dtype is the problem.
**Status:** present in 2.21.0 on CPU, independent of the layout and the blank index; the float32 path
agrees with a hand-written CTC forward reference to `1e-7`.
**Found with:** the wave-387 `tf.nn` selector/dtype acceptance probe; certified reproducer `repro.py`
in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 16 checks).

---

## Summary

```python
labels = [[1, 2]]; label_length = [2]; logit_length = [4]
logits = np.random.randn(4, 1, 3)                     # [frames, batch, classes]

tf.nn.ctc_loss(labels, tf.constant(logits.astype(np.float32)), label_length, logit_length)
# <tf.Tensor: shape=(1,), dtype=float32, numpy=array([1.6023118], dtype=float32)>   (matches the reference)

tf.nn.ctc_loss(labels, tf.constant(logits.astype(np.float64)), label_length, logit_length)
# InvalidArgumentError: cannot compute Mul as input #1(zero-based) was expected to be a double
# tensor but is a float tensor [Op:Mul]
```

Controls: the float32 result matches a log-space CTC forward reference to `1e-7` for both layouts;
`tf.raw_ops.CTCLoss` **accepts** the same logits as float64 and returns a float64 loss that agrees with
its own float32 call; the SparseTensor-label wrapper path works in float32 (with `blank_index` given).

## Root cause

`tf.nn.ctc_loss` converts the dense labels to a SparseTensor, appends the blank class and builds the
`CTCLoss` call, but one of the intermediate tensors it multiplies into the logits is created as
float32 (independent of the logits dtype), so the graph fails for float64.  The kernel itself supports
both dtypes (`tf.raw_ops.CTCLoss` documents `float32`/`float64` and accepts float64 here), so the fix
is in the wrapper: build the helper constants with `logits.dtype` (or cast them).

## Suggested fix

In `tensorflow/python/ops/ctc_ops.py::ctc_loss`, create the internal constants/zeros with
`dtype=logits.dtype` (typically the `logit_length`/label-mask arithmetic) so float64 logits flow through;
a regression test is the witness above with `tf.float64`, which must match the float32 result to
`1e-10`.

## Impact

* float64 CTC training or evaluation (used for high-precision loss comparisons, reference
  implementations and cross-framework tests) fails immediately with an unrelated-looking error.
* The error mentions an internal `Mul` and dtypes but not `ctc_loss`, `logits` or "use float32", so the
  workaround (cast to float32) has to be discovered empirically.

## Controls (all in `repro.py`, 16 checks, 0 failing on each of 2 replays)

* **Reference control** - the float32 loss matches a log-space CTC forward implementation for both
  `logits_time_major` settings (`1.6023118` vs `1.6023118`).
* **Dtype control** - the same call with float64 raises `InvalidArgumentError` mentioning the `Mul`
  dtype mismatch, for both layouts and with/without `blank_index`.
* **Kernel control** - `tf.raw_ops.CTCLoss` with float64 inputs returns a float64 loss, and it agrees
  with the raw float32 call to `1e-5`, i.e. the kernel is dtype-capable and the wrapper is at fault.
* **API-documentation control** - the `CTCLoss` op definition lists `float32` and `float64` for
  `inputs` (asserted in the reproducer).
* **Sparse-label control** - the SparseTensor-label wrapper path works in float32 when `blank_index` is
  supplied.

## Novelty

Searched the TensorFlow tracker on 2026-09-14 for `ctc_loss float64`, `CTCLoss double` and
`ctc_loss dtype`: no report of float64 being unusable (`CTCLoss double` surfaces only the unrelated
"MaxPool doesn't accept float64" and TFLite/TPU items; the `ctc_loss` hits concern SparseTensor
behaviour #55328, `num_classes=0` crashes #112638 and a GPU gradient bug #41280).  The campaign has no
prior `tf.nn.ctc_loss` record.

## Environment

```
tensorflow 2.21.0 (CPU), numpy 2.5.1, Python 3.12.13, WSL2 - see collect_env.txt and evidence.txt
```
