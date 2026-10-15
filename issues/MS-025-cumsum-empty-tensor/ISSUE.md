# Zero-size results abort: `ops.cumsum` on an empty tensor and `ops.repeat_interleave` with all-zero repeats

**Component:** MindSpore 2.9.0, CPU, PYNATIVE and GRAPH: `ops.cumsum` and `ops.repeat_interleave`
(`mint.cumsum` has the same gap through its own unregistered kernel)
**Severity:** valid empty inputs abort with an internal kernel-launch `RuntimeError` instead of
returning the empty tensor, so padding/masking code paths that legitimately produce empty
sequences crash
**Status:** present in MindSpore 2.9.0 (CPU); `ops.cumprod` on the same tensors returns `[]`, and
NumPy, PyTorch, JAX and TensorFlow return the empty array for both
**Found with:** the (corrected) wave-167 runtime/indexing grid (`artifact/hunt-v2/results/w167_*`),
triaged by Rondo's `check_consensus` oracle with NumPy as arbiter; the certified reproducer is
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

```python
import numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

ops.cumsum(ms.Tensor([], ms.float64), 0)
# RuntimeError: Launch kernel failed, name:Default/CumSum-op0
# (cpu kernel log: "Invalid zero value. Please check resize input data.")

ops.cumprod(ms.Tensor([], ms.float64), 0)     # []       <- works
np.cumsum(np.array([])), np.cumprod(np.array([]))   # [], []  <- the semantics
```

Zero total size is the failure condition: the 1-D empty tensor and a `(0, 3)` tensor both raise
(`CumSum-op0`/`op1`), while a `(3, 0)` tensor (zero-length rows) and every non-empty input are
correct.  `ops.cumprod` handles the same empty tensors, and the `mint` layer raises its own
`The kernel CumsumExt unregistered` (the MS-013 family), so both layers have a gap and only the
`ops` one is a kernel-launch failure.  Eager GRAPH mode fails identically.

**The same class hits `ops.repeat_interleave` with an all-zero repeats tensor**: the result is
legitimately empty, but the operator aborts because an internal `Concat` cannot build a zero-size
tensor:

```python
ops.repeat_interleave(ms.Tensor([1.0, 2.0, 3.0]), ms.Tensor([0, 0, 0], ms.int32), 0)
# ValueError: For primitive[Concat], the elements size must be greater than 0, but got 0.
np.repeat([1.0, 2.0, 3.0], [0, 0, 0])     # []   <- valid empty result
```

Non-zero repeats on the same tensors are correct (`[0,1,2] -> [2,3,3]`, `[1,1,1] -> [1,2,3]`), so
only the zero-size result is rejected.

## Impact

`cumsum` is routinely applied to sequences whose length is data-dependent (masked attention,
padding-stripped batches, empty ragged segments), and all-zero repeat counts are the natural result
of masking a batch.  An empty sequence must produce an empty result; instead the process aborts with
an internal kernel message.  The inconsistency with `cumprod` (and with non-zero repeat counts) in
the same build makes it hard to predict which operator is safe on empty input.

## Suggested fix

Short-circuit zero-size inputs in the CumSum CPU kernel dispatch (or compute the output shape and
return early, as `cumprod` evidently does) and add empty-shape cases to the CPU tests for both
`cumsum` and `cumprod`.

## Controls (all in `repro.py`)

* `ops.cumprod` returns `[]` for both empty shapes, and
  `ops.repeat_interleave([1,2,3], [0,1,2])`/`([1,1,1])` are correct where the all-zero repeats
  tensor raises.
* `ops.cumsum` matches NumPy on `[1.0]`, `[1,2,3]`, `[1e308, 1e308]`, `[nan, 1.0]`, `[-0.0, 0.0]`,
  `[3, -1, 4]`.
* A `(3, 0)` tensor and single-element tensors are unaffected, so the defect is specific to a zero
  total size rather than to small shapes.
* Eager GRAPH mode fails the same way; `mint.cumsum` fails through the MS-013 unregistered kernel.

## Novelty

Searched the MindSpore tracker for `cumsum empty`, `empty tensor cumsum`, `cumsum zero size`: no
report exists.  The campaign's other empty/degenerate-shape work (wave 150 reductions, wave 167's
runtime grid) found no other failing case; MS-013 is the separate `mint` `*Ext` registration family.

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Empty tensors are a degenerate input."**  They are valid and common (data-dependent
  sequence lengths), and every other framework - and MindSpore's own `cumprod` - returns an empty
  result.  Aborting with an internal kernel message is a defect, not a documented restriction.
* **"The kernel log says 'Invalid zero value'."**  That is the internal cause, not an API contract:
  there is no documented pre-condition that `cumsum` requires a non-empty tensor.
