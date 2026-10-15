# `tf.image.non_max_suppression` under XLA silently accepts an out-of-range `iou_threshold`

**Component:** TensorFlow 2.21.0, CPU.  `tf.function(..., jit_compile=True)` (XLA:CPU) vs eager and
`tf.function` without XLA.
**Severity:** silent wrong selection.  The CPU kernel documents and enforces
`iou_threshold must be in [0, 1]`; under XLA an out-of-range threshold is applied literally, so a
threshold `> 1` or `NaN` suppresses **nothing** and a negative threshold suppresses **everything**,
without any error.  A `NaN` threshold (easy to produce from upstream arithmetic) silently degrades
detection to "keep all boxes".
**Status:** present in TensorFlow 2.21.0 (CPU); in-range values, `max_output_size = 0` and the
`max_output_size < 0` error agree between the lanes, and the sibling range validations of
`tf.image.adjust_gamma` and `tf.nn.dropout` still raise under XLA.
**Found with:** the wave-197 Rondo invalid-argument grid (axes, shapes, boxes, thresholds, rates),
Reference = eager, Fresh = `tf.function(jit_compile=True)` with the non-XLA `tf.function` as the
control; certified reproducer is `repro.py` in this directory (`--replays 2` ->
`RESULT: 0 failing replay(s).`, 30 checks).

---

## Summary

```python
import tensorflow as tf
boxes = tf.constant([[0.,0.,.5,.5],[.1,.1,.4,.4],[.2,.2,.3,.3],[.6,.6,.9,.9]])
scores = [0.9, 0.8, 0.7, 0.6]

tf.image.non_max_suppression(boxes, scores, 4, iou_threshold=1.5)
# InvalidArgumentError: iou_threshold must be in [0, 1] [Op:NonMaxSuppressionV3]

tf.function(lambda: tf.image.non_max_suppression(boxes, scores, 4, iou_threshold=1.5),
            jit_compile=True)()
# -> <tf.Tensor: [0 1 2 3]>      no error, and nothing was suppressed
```

| `iou_threshold` | `tf.function` (CPU) | `tf.function(jit_compile=True)` |
|---|---|---|
| `0.0` | `[0, 3]` | `[0, 3]` (control) |
| `0.5` / `1.0` | `[0, 1, 2, 3]` | `[0, 1, 2, 3]` (control) |
| `1.5` | raises `InvalidArgumentError` | `[0, 1, 2, 3]` |
| `2.0` | raises | `[0, 1, 2, 3]` |
| `NaN` | raises | `[0, 1, 2, 3]` |
| `+inf` | raises | `[0, 1, 2, 3]` |
| `-0.5` | raises | `[0]` |

The mechanism is visible in the results: the XLA lowering evaluates `iou > threshold` without the
range check, so `> 1`, `NaN` and `inf` never suppress and a negative threshold always does.
`max_output_size = 0` returns `[]` and `max_output_size = -1` raises in **both** lanes, so the XLA
path is not check-free in general - it is specifically the `iou_threshold` range validation that is
missing.

## Root cause

The CPU `NonMaxSuppressionV3` kernel validates `iou_threshold` and raises `InvalidArgumentError`
(the message quoted above is the kernel's own).  The XLA lowering of the op performs the suppression
loop with the comparison but without that argument validation, so the same program compiled with
`jit_compile=True` runs with an unchecked threshold.

## Impact

Non-maximum suppression is the final step of most object detectors, and the threshold is often
computed (e.g. a calibrated or scheduled value, or a tensor reduced from data).  A threshold that
becomes `NaN` or leaves `[0, 1]` through a bug:

* eager / `tf.function`: fails loudly with `iou_threshold must be in [0, 1]`, so the bug is caught;
* `tf.function(jit_compile=True)`: silently returns *all* overlapping boxes (`NaN`, `inf`, `> 1`) or a
  single box (negative), i.e. a completely different detection set, with no NaN and no error to
  notice.  The failure is data-dependent and only appears in the compiled lane.

## Suggested fix

Validate `iou_threshold` in the XLA lowering (raise the same `InvalidArgumentError` as the CPU
kernel) or, if XLA is meant to stay check-free, make the eager path match and document it.  A
`jit_compile=True` test with `iou_threshold=1.5` or `NaN` asserting the exception would pin it.

## Controls (all in `repro.py`, 30 checks)

* In-range thresholds `0.0`, `0.5`, `1.0` produce the same indices in both lanes.
* `max_output_size = 0` -> `[]` in both lanes; `max_output_size = -1` raises in both.
* The contract string `iou_threshold must be in [0, 1]` is asserted from the eager kernel's own
  error, so a future change of the contract would be noticed.
* `tf.image.adjust_gamma(..., -1.0)` and `tf.nn.dropout(..., rate=1.5)` raise in **both** lanes,
  showing that XLA does keep other argument validations - the gap is specific to this op.
* `tf.image.combined_non_max_suppression` rejects the same threshold in both lanes;
  `tf.image.non_max_suppression_with_scores` is not XLA-compilable at all in this build
  ("Detected unsupported operations when trying to compile graph").

## Novelty

Searched the TensorFlow tracker for `non_max_suppression iou_threshold`,
`iou_threshold must be in [0, 1]`, `non_max_suppression jit_compile` and
`XLA NMS validation`: the adjacent reports are
[#117245](https://github.com/tensorflow/tensorflow/issues/117245) (NMS with **empty inputs** crashes
under `jit_compile=True`),
[#119358](https://github.com/tensorflow/tensorflow/issues/119358) (the **padded** variant prioritises
`NaN` scores differently from eager) and
[#105643](https://github.com/tensorflow/tensorflow/issues/105643) (a loop-over-batch compile
failure).  None of them is about the `iou_threshold` range check.  The campaign's other TensorFlow
drafts are unrelated (TF-011..TF-018).

## Environment

TensorFlow 2.21.0 (CPU, XLA:CPU through `tf.function(jit_compile=True)`), NumPy 2.5.1, Python 3.12.13,
WSL2 (Linux 6.6.87.2) x86_64.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The threshold is invalid input."**  It is validated by the CPU kernel and the message documents
  the range; eager and `tf.function` raise.  The defect is that compiling the same program silently
  changes the contract, and that the substitute behaviour (keep everything / keep one box) is a
  plausible detection set rather than a visible failure.
* **"XLA cannot raise at runtime."**  It can: `max_output_size = -1` raises under XLA in this very
  op, `adjust_gamma` and `dropout` range checks raise under XLA, and other XLA kernels raise
  `InvalidArgumentError` for runtime conditions.  The `NonMaxSuppression` lowering simply does not
  carry the check.
* **"Only hand-written invalid thresholds are affected."**  `NaN` is the realistic case - it arrives
  from upstream arithmetic and is accepted by the compiled path as "suppress nothing", which is
  exactly the silent-degradation pattern this report is about.
