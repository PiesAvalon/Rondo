# `tf.sparse.add` drops any summed entry whose value is NaN, although the default `threshold=0.0` documents that "everything is kept"

**Component:** TensorFlow 2.21.0, CPU.  `tf.sparse.add(sparse, sparse)` (and its `tf.raw_ops.SparseAdd` binding), eager and under `tf.function`; `float64` and `float32`; rank 1 and rank 2.
**Severity:** silent wrong value.  When two sparse operands share an index and their values sum to **NaN** (`NaN + 1`, `NaN + NaN`, `inf + (-inf)`), the output entry is **removed** instead of keeping NaN, so the dense reading of the result is `0.0`.  A corrupted value becomes a clean zero exactly at the coordinates where two non-finite values meet, and the same computation done densely reports NaN - so sparse and dense code paths disagree, and NaN never reaches a downstream check.
**Status:** present in TensorFlow 2.21.0 (CPU); `tf.sparse.add` is not XLA-compilable in this build ("Detected unsupported operations ... SparseAdd"), so only the eager and `tf.function` lanes exist.
**Found with:** the wave-317/318 sparse-tensor hunt (dense-reference differential over `to_dense`, reductions, transpose, reshape, `sparse_dense_matmul` and `sparse.add` with duplicates, NaN, ±inf and signed zeros; the only divergence that is not a documented sparse convention or an implicit-zero artifact); certified reproducer `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 66 checks).

---

## Summary

```python
import numpy as np, tensorflow as tf

def st(v):
    return tf.sparse.SparseTensor(indices=tf.constant([[0, 0]], tf.int64),
                                  values=tf.constant([v], tf.float64),
                                  dense_shape=tf.constant([1, 1], tf.int64))

a, b = st(float("nan")), st(1.0)
tf.sparse.add(a, b)                               # SparseTensor with ZERO values
tf.sparse.to_dense(tf.sparse.add(a, b))           # [[0.]]        <- the entry is gone
tf.sparse.to_dense(a) + tf.sparse.to_dense(b)     # [[nan]]       <- dense / IEEE reference
# scipy.sparse: (a + b).todense() -> [[nan]]      torch.sparse: (a + b).to_dense() -> [[nan]]
```

| pair at the same index | `tf.sparse.add` | dense reference `to_dense(a)+to_dense(b)` | scipy / torch sparse |
|---|---|---|---|
| `1.0 + 1.0` | `2.0` (kept) | `2.0` | `2.0` |
| `1.0 + (-1.0)` | **`+0.0` (kept)** | `0.0` | `0.0` |
| `0.0 + 0.0`, `-0.0 + -0.0` | kept, sign preserved | same | same |
| `inf + 1.0`, `-inf + 1.0` | `inf` / `-inf` (kept) | same | same |
| **`NaN + 1.0`** | **entry dropped -> dense `0.0`** | **`nan`** | **`nan`** |
| **`NaN + NaN`** | **entry dropped** | **`nan`** | **`nan`** |
| **`inf + (-inf)`** | **entry dropped** | **`nan`** | **`nan`** |

The trigger is the summation path, not NaN as such: an index present in only one operand keeps its
NaN (`values=[nan, 1.0]` for disjoint inputs), the dense-operand lane
(`tf.sparse.add(sparse, dense)`, which returns a dense `Tensor`) keeps NaN, and `to_dense` of the
input keeps NaN.  The same drop reproduces for `float32` and for rank-2 shapes.

## Root cause

`tf.sparse.add`'s documented clipping rule is a **drop** predicate:

> "…callers can specify `threshold`, indicating that if the sum has a magnitude **strictly smaller
> than `threshold`**, its corresponding value and index would then not be included.  In particular,
> `threshold == 0.0` (default) means **everything is kept** and actual thresholding happens only for
> a positive value."  (`tf.sparse.add` docstring, TensorFlow 2.21.0)

The implementation keeps an entry only when the **opposite** test succeeds, `|value| >= threshold`.
Every comparison with NaN is false, so a NaN sum fails the keep test and is pruned - even at the
default threshold, where the documentation promises that nothing is dropped.  The reproducer pins
the predicate from both sides:

| call | result | what it shows |
|---|---|---|
| `add(0.1, 0.1, threshold=0.5)` | dropped | the documented magnitude clipping still works |
| `add(0.1, 0.1, threshold=0.0)` | `0.2` kept | default keeps a small finite value |
| `add(1.0, -1.0, threshold=0.0)` | `+0.0` kept | the rule is `>=`, so an exact zero sum survives |
| `add(NaN, 1.0, threshold=0.0)` | **dropped** | NaN fails `>=` and is pruned |
| `add(NaN, 1.0, threshold=1e-30)` | dropped | the same at any threshold |

## Impact

* **NaN is silently converted into an implicit zero.**  Any sparse accumulation - `A + B`, residual
  and overlap sums, segment bookkeeping, sparse gradient accumulation - erases NaN at exactly the
  coordinates where two non-finite values meet.  A downstream `tf.debugging.check_numerics` on the
  dense reading sees `0.0` and passes.
* **Sparse and dense code paths disagree.**  Computing the same sum through
  `tf.sparse.to_dense(a) + tf.sparse.to_dense(b)` (or through scipy/torch sparse) yields NaN, so a
  model gives different results depending on how the addition is spelled, and the sparse spelling is
  the one that hides the corruption.
* **Silent and shape-preserving**: the output is a valid `SparseTensor` with one fewer entry; nothing
  raises, and no metadata records the removal.

## Suggested fix

Test NaN explicitly instead of relying on the ordering comparison, e.g. keep an entry when
`!(abs(v) < threshold)` (NaN passes) or when `v != v`, or apply the threshold to the two input values
before summing as the documentation describes.  A regression test:

```python
a, b = st(float("nan")), st(1.0)
out = tf.sparse.add(a, b)
assert out.values.shape[0] == 1 and np.isnan(out.values.numpy()[0])   # currently 0 values
```

## Novelty

TensorFlow tracker searches for `sparse add nan`, `SparseAdd nan`, `tf.sparse nan`,
`sparse nan dropped` and `sparse add non-finite` return no report of an entry being dropped.  The
nearest items are unrelated: #49525-#49528 (2021, `tf.raw_ops.SparseAdd` invalid-memory-access fixes),
#99322 ("Add SparseTensor support for math operations") and #50070 (sparse softmax/xent GPU
determinism).  The behaviour of the reference implementations was measured in the same week:
`scipy.sparse` (`(a + b).todense()`) and `torch.sparse_coo_tensor` (`(a + b).to_dense()`) both keep
NaN for all three pairs above, so the drop is TensorFlow-specific and contradicts its own docstring.

## Environment

TensorFlow 2.21.0 (CPU), Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64.  Full dump in
`collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Threshold clipping is documented; dropping the entry is expected."**  The documented rule is a
  *magnitude* comparison (`|sum| < threshold`), and with the default `threshold=0.0` the docstring
  says explicitly that **everything is kept**.  A NaN has no magnitude, but it is not a value the
  user asked to drop; the implementation tests the negation of the documented predicate, which is
  what turns NaN into a drop.
* **"A sparse tensor may omit explicit zeros, so pruning is fine."**  NaN is not zero: the dense
  reference, scipy and torch all return NaN, and the dense-operand lane of the same TensorFlow
  operator returns NaN.  The drop is inconsistent inside TF as well.
* **"Duplicates/non-finite inputs are undefined."**  This is not a duplicate-index input: each input
  has a single entry at the index and both are well-formed and sorted, as the docstring requires.
  `NaN + 1` and `inf + (-inf)` are ordinary IEEE operations with a defined result.
* **"Does it reproduce outside your machine?"**  The predicate is a plain C++ comparison in the
  sparse-add kernel and is dtype- and rank-independent (checked for `float64`/`float32` and rank 1/2);
  the reproducer prints the mechanism checks that pin it.
