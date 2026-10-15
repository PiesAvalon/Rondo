# XLA silently clamps or drops out-of-range `tf.gather` / `tf.nn.embedding_lookup` / `tf.scatter_*` indices (CPU indices are documented to be always validated)

**Component:** TensorFlow 2.21.0, CPU.  `tf.function(..., jit_compile=True)` (XLA:CPU) vs eager and
`tf.function` without XLA.
**Severity:** silent wrong data.  An out-of-range index does not fail; it is **clamped** into
`[0, n-1]` and returns another row's values (or, for `unsorted_segment_*`, silently drops a
contribution), so a miscomputed index corrupts the result only after the model is compiled.
**Status:** present in TensorFlow 2.21.0 (CPU).  The op's own docstring states the contract that XLA
breaks: *"The `validate_indices` argument has no effect. Indices are always validated on CPU and never
validated on GPU."*
**Found with:** the wave-195 Rondo index-bounds grid (indices, `gather`, `gather_nd`,
`embedding_lookup`, segment ops), Reference = eager, Fresh = `tf.function(jit_compile=True)` with the
same graph through `tf.function` as the non-XLA control; certified reproducer is `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 75 checks).

---

## Summary

```python
import tensorflow as tf

v = tf.constant([10.0, 20.0, 30.0])
tf.gather(v, [3])                                        # InvalidArgumentError (eager, and tf.function)
tf.function(lambda: tf.gather(v, [3]), jit_compile=True)()
# -> [30.0]        no error: index 3 was clamped to the last element

tf.function(lambda: tf.gather(v, [-1]), jit_compile=True)()   # -> [10.0]   (clamped to the first)
tf.function(lambda: tf.gather(v, [0, 5, 2]), jit_compile=True)()
# -> [10.0, 30.0, 30.0]    index 5 silently reads element 2
```

Measured with the same graph through `tf.function` without XLA as the control:

| expression | `tf.function` (CPU) | `tf.function(jit_compile=True)` |
|---|---|---|
| `tf.gather(v, [3])` | raises `InvalidArgumentError` | `[30.0]` |
| `tf.gather(v, [4])` | raises | `[30.0]` |
| `tf.gather(v, [-1])` / `[-5]` | raises | `[10.0]` |
| `tf.gather(v, [0, 5, 2])` | raises | `[10.0, 30.0, 30.0]` |
| `tf.gather(w, [3], axis=1)` | raises | `[[2.],[5.],[8.],[11.]]` |
| `tf.gather(w, [[4, 0]])` | raises | `[[[9.,10.,11.],[0.,1.,2.]]]` |
| `tf.nn.embedding_lookup(w, [4])` | raises | row 3 (`[[9.,10.,11.]]`) |
| `tf.nn.embedding_lookup(w, [-1])` | raises | row 0 (`[[0.,1.,2.]]`) |
| `tf.math.unsorted_segment_sum(s, [0,5,0,1], 2)` | raises | `[4., 4.]` (the id-5 contribution is dropped) |
| `tf.scatter_nd([[3]], [9.], [3])` | raises | `[0., 0., 0.]` (the update is dropped) |
| `tf.scatter_nd([[-1]], [9.], [3])` | raises | `[0., 0., 0.]` |
| `tf.tensor_scatter_nd_update(v, [[3]], [9.])` | raises | `v` unchanged (the update is ignored) |
| `tf.tensor_scatter_nd_add(v, [[3]], [9.])` | raises | `v` unchanged |
| in-range variants | value | identical value (control) |

The clamp is to the closed range `[0, n-1]` (too large -> last, negative -> first), which makes the
result finite and plausible: `tf.nn.embedding_lookup` with an off-by-one vocabulary index returns
another token's embedding instead of failing, and `unsorted_segment_sum` with a bad segment id
silently loses that element.

## Root cause

The eager CPU kernels validate indices and raise `InvalidArgumentError`, as the docstring promises
("Indices are always validated on CPU").  The XLA lowerings of `Gather`, `ScatterNd`
and `unsorted_segment_sum` perform the access without that validation: gather clamps the index into
range, scatter silently drops out-of-range updates, and the segment lowering discards the
out-of-range contribution.
`tf.function` without `jit_compile` still runs the validating CPU kernel, so the two TensorFlow lanes
disagree; numpy (`np.take`) and PyTorch (`torch.gather`, `index_select`,
`F.embedding`) both raise for the same access.

## Impact

* Index arithmetic in real models (vocabulary lookups, gathered features, attention indexing,
  `unsorted_segment_*` aggregations) is routinely off by one or uses a stale table size.  Eager TF
  fails loudly; after switching to `jit_compile=True` the model silently consumes another row's data
  and produces plausible but wrong outputs, and the aggregation loses a contribution.
* The failure is data-dependent and silent, so it cannot be caught by a shape or NaN check; it also
  differs between the two TensorFlow execution lanes for the same program.

## Suggested fix

Add the bounds check to the XLA lowering (raise the same `InvalidArgumentError` as the CPU kernel
when an index is outside `[0, dim)`), or - if XLA is intended to stay check-free - make the CPU/eager
path behave the same and update the docstring, which currently promises CPU validation.  A
`jit_compile=True` test asserting the exception for `tf.gather(v, [len(v)])` would pin the behaviour.

## Controls (all in `repro.py`, 75 checks)

* In-range `gather` (axis 0 and 1), `embedding_lookup`, `unsorted_segment_sum`,
  `scatter_nd` (including duplicate indices, which sum) and `tensor_scatter_nd_*` give identical
  values in both lanes.
* The non-XLA `tf.function` lane raises `InvalidArgumentError` for every out-of-range case, so the
  divergence is specific to XLA and not to graph mode.
* Eager `tf.gather` raises the same error before any compilation.
* `numpy.take` raises for the same access (independent reference).
* The docstring sentence that establishes the contract is asserted in the reproducer, so a doc
  change would be noticed.

## Novelty

Two adjacent reports exist for other variants of the same class:
[#112068](https://github.com/tensorflow/tensorflow/issues/112068) ("`tf.gather_nd` silently accepts
out-of-bounds indices under XLA compilation") and
[#118076](https://github.com/tensorflow/tensorflow/issues/118076) ("XLA accepts negative indices in
`tf.gather` with `batch_dims` while eager raises `InvalidArgumentError`").  Neither covers the plain
`tf.gather` (`axis=0` and `axis=1`, 1-D and 2-D indices, positive and negative), `tf.nn.embedding_lookup`
or the `unsorted_segment_sum` and `scatter_nd`/`tensor_scatter_nd_update`/`_add` variants reported
here; the docstring's "always validated on CPU" makes the contract explicit.  The campaign's other TensorFlow drafts are unrelated (TF-011 eager scalar
cache, TF-013/014 complex overflow, TF-015 cumsum signed zero, TF-016 bool scalar conversion, TF-017
softmax `+inf` under XLA).

## Environment

TensorFlow 2.21.0 (CPU, XLA:CPU through `tf.function(jit_compile=True)`), NumPy 2.5.1, Python 3.12.13,
WSL2 (Linux 6.6.87.2) x86_64.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Out-of-range indices are undefined input."**  They are validated on CPU by design and the
  docstring says so; eager and `tf.function` raise.  The defect is that compiling the same program
  silently changes the contract, and that the substitute result is plausible data rather than a
  visible NaN.
* **"XLA cannot raise at runtime."**  It can and does for other conditions in the same build
  (`tf.tile` with negative multiples, `tf.math.segment_sum` with negative ids, unsupported-op
  compilation errors), and the `tf.function` lane raises.  Even if a raising check were impossible,
  returning clamped data silently is the part that needs a decision and a docstring, not an accident.
* **"This is the same as #112068/#118076."**  Those cover `gather_nd` and the `batch_dims` negative
  case; the plain gather/embedding/segment variants here are separate entry points, and the report
  includes the documented CPU contract.  If a maintainer prefers to fold it in, the table above is
  the instance list to attach.
