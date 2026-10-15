# Negative indices are silently wrapped: the loss family scores a different class and `ops.gather` returns another element

**Component:** MindSpore 2.9.0, CPU, PYNATIVE and GRAPH (`ms.jit`), `float16`/`float32`/`float64`.
**Severity:** silent wrong values.  A **negative** class label - a common data bug (a `-1` "missing
label" sentinel, a signed encoding, an off-by-one) - is not rejected: it is wrapped Python-style to
the end of the class axis, so the loss is computed for a **different class** and stays finite and
plausible.  The upper bound *is* enforced (`label = C` raises), which makes the one-sided check the
defect.
**Status:** present in MindSpore 2.9.0 (CPU); the same behaviour in `ops.cross_entropy`,
`nn.CrossEntropyLoss` and `ops.nll_loss`, in both execution modes and all three float widths;
PyTorch raises `IndexError: Target N is out of bounds.` for every out-of-range label.
**Found with:** the wave-199 Rondo loss/argument grid (label range and dtype grid across the loss
family), Reference = the documented `logsumexp - logit` formula computed in the matching dtype and
PyTorch's `F.cross_entropy` for the error contract, Fresh = `ms.jit`; the certified reproducer is
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 54 checks).

---

## Summary

The `ops.cross_entropy` docstring states the target contract:

> Class indices (int) in the range `[0, C)` where `C` is the number of classes.

The upper bound is checked, the lower bound is not:

```python
import numpy as np, mindspore as ms
from mindspore import Tensor, ops
logits = Tensor(np.asarray([[1.0, 2.0, 3.0]], np.float32))     # C = 3

ops.cross_entropy(logits, Tensor(np.asarray([2],  np.int32)))  # 0.40761   (class 2, correct)
ops.cross_entropy(logits, Tensor(np.asarray([-1], np.int32)))  # 0.40761   <-- silently class 2
ops.cross_entropy(logits, Tensor(np.asarray([-2], np.int32)))  # 1.40761   <-- silently class 1
ops.cross_entropy(logits, Tensor(np.asarray([-3], np.int32)))  # 2.40761   <-- silently class 0
ops.cross_entropy(logits, Tensor(np.asarray([3],  np.int32)))
# RuntimeError: For 'GatherD', the value of 'index' must be in [-3, 3), but got 3
```

| label | MindSpore loss | which class it actually scores |
|---|---|---|
| `0`, `1`, `2` | `2.40761`, `1.40761`, `0.40761` | the labelled class (correct) |
| `-1` | `0.40761` | class `2` |
| `-2` | `1.40761` | class `1` |
| `-3` | `2.40761` | class `0` |
| `3` | raises `RuntimeError` | - (upper bound enforced) |
| `-4` | raises `RuntimeError` | - (beyond the wrapping range) |

The wrap appears in all three entry points and both modes:

* `nn.CrossEntropyLoss()(logits, [-1])` returns the class-2 loss; the Cell's `label = 3` raises the
  same `RuntimeError`.
* `ops.nll_loss(log_probs, [-1])` returns `0.40761`, the value for class 2.
* `ms.jit` reproduces every value bit-for-bit (`eager == jit`), and `float16`/`float64` behave the
  same as `float32` (`0.408203125` / `0.4076059644443806` for `label = -1`), so this is the indexing
  path, not a precision effect.
* With `reduction='none'` the wrapped per-sample losses are returned individually; `ignore_index=-1`
  correctly ignores the label `-1` but does **not** stop `-2` from wrapping to class `C-2`.

PyTorch rejects the whole out-of-range region - `F.cross_entropy(logits, [3])` and every negative
label raise `IndexError: Target N is out of bounds.` - and NumPy-based references are only defined on
`[0, C)`, so the wrapping is MindSpore-specific.

## The same wrap in the underlying op: `ops.gather`

The loss functions are one consumer of the wrapping; `ops.gather` itself behaves the same way, and its
own docstring states the contract that is broken:

> The value of `input_indices` must be in the range of `[0, input_params.shape[axis])`.  **On CPU and
> GPU, an error is raised if an out of bound index is found.**  On Ascend, the results may be
> undefined.

| expression | MindSpore | PyTorch |
|---|---|---|
| `ops.gather([10.,20.,30.], [-1], 0)` | `[30.0]` (last element) | `torch.gather(..., [-1])` raises `RuntimeError: index -1 is out of bounds` |
| `ops.gather(..., [-2], 0)` | `[20.0]` | raises |
| `ops.gather(..., [-3], 0)` | `[10.0]` | raises |
| `ops.gather(..., [3], 0)` | raises `RuntimeError: For 'Gather', the 'input_indices' should be in the range [0, ...)` | raises |
| `ops.gather(matrix, [-1], 1)` | last column (`[3.0, 6.0]`) | raises |

`ms.jit` reproduces the eager wrap for the in-range and negative cases, and the upper bound raises in
both lanes.  NumPy's `np.take` does wrap negative indices, but that is a documented convenience of
`take`/fancy indexing; `torch.gather`/`index_select` reject them, and MindSpore's own docstring
promises an error for an out-of-bound index on CPU.

## Root cause

The loss implementation gathers the logits at the target positions with `GatherD` (the error text
names it) and `ops.gather` uses `Gather`; both kernels accept Python-style negative indices.  The op's validation only rejects an
index `>= C`, so a negative label falls through to the gather and selects the class at `C + label`.
The fix is to validate `label >= 0` (and `< C`) before the gather, in the `cross_entropy`/`nll_loss`
frontend or in the label-gather path, or to mask negative labels out explicitly with the documented
`ignore_index` semantics.

## Impact

A mislabelled sample is the most common data bug in a training pipeline, and a negative label is the
usual sentinel for "unknown/missing".  With this behaviour:

* `label = -1` silently trains or evaluates against the **last class** instead of failing, and the
  resulting loss looks completely plausible (it is a real cross-entropy value for another class);
* the error is one-sided, so a dataset with negative labels passes `label = C`-style sanity checks
  that only test the upper bound;
* every consumer of the loss (`nn.CrossEntropyLoss` modules, `ops.nll_loss`) inherits the wrap, and
  `ms.jit`/GRAPH mode reproduces it exactly, so it is not a mode-specific artefact.

## Suggested fix

Reject negative labels with the same error the upper bound produces (or make the label gather
non-wrapping), and add a test asserting that `label = -1` raises for `ops.cross_entropy`,
`nn.CrossEntropyLoss` and `ops.nll_loss`.  If negative labels are intended to mean "ignore", implement
that explicitly and document it, rather than relying on `GatherD`'s index wrap.

## Controls (all in `repro.py`, 54 checks)

* Valid labels `0..C-1` match the documented `logsumexp(logits) - logits[label]` reference in the
  matching dtype, in both eager and `ms.jit`.
* The out-of-range labels `C`, `C+1` and `-(C+1)` raise `RuntimeError` in both lanes, so the check
  exists and is one-sided.
* `reduction='none'` per-sample values are checked against the reference for both the valid and the
  wrapped label.
* `ignore_index=-1` correctly ignores the label `-1` (loss equals the remaining sample's loss) while
  `-2` still wraps, which shows the wrap is independent of the documented ignore mechanism.
* `nn.CrossEntropyLoss` and `ops.nll_loss` are checked separately with their own expected values.
* `ops.gather` is checked on axis 0 and axis 1 for the negative indices `-1/-2/-3` against the
  corresponding positive index, with the upper bound `3` raising in both lanes and in-range indices
  agreeing between the lanes.
* Both contract sentences of the `ops.gather` docstring (the `[0, shape[axis])` range and the
  promise that "an error is raised if an out of bound index is found" on CPU) are asserted.
* float16/float32/float64 are each compared against a reference computed in the same dtype.
* The docstring sentence that establishes the `[0, C)` contract is asserted, so a documentation
  change would be noticed.

## Novelty

Searched the MindSpore tracker and web for `cross_entropy negative label`, `cross_entropy target
out of range`, `nll_loss negative label`, `CrossEntropyLoss negative target`, `ops.gather negative
index` and the Chinese equivalents (`cross_entropy 负标签`, `gather 负索引`): the hits are the API documentation pages, a `models`
CrossEntropyLoss GPU report unrelated to label ranges, and unrelated PRs - no report of negative
labels being wrapped.  The campaign's other MindSpore drafts are different operators/paths
(MS-021..MS-026 kernels, MS-027 GRAPH dtype promotion); none covers the loss label range.

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64
(PYNATIVE and GRAPH via `ms.jit`).  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Negative indexing is a feature."**  The docstring restricts the target to `[0, C)`, the upper
  bound is enforced with an error, and PyTorch's equivalent raises for every out-of-range label.
  There is no documented negative-label aliasing, and `ignore_index` already exists as the explicit
  mechanism for "skip this sample".
* **"Only a data bug hits this."**  Yes - and the framework's job for an out-of-range label is to
  report it, as it already does for `label = C`.  Silently scoring another class turns a loud data
  error into a wrong model, which is exactly the failure mode a differential oracle should surface.
* **"`ms.jit` differs from eager."**  It does not: both lanes agree exactly, which is why this is
  reported as a runtime/kernel defect rather than a compiler divergence.
