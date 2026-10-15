# `ops.unique_consecutive` emits one extra `nan` when the first input element is NaN

**Component:** MindSpore 2.9.0, CPU, PYNATIVE and GRAPH (`ms.jit`), `ops.unique_consecutive` on
`float32`/`float64`
**Severity:** wrong value and wrong shape — the output is one element longer than it should be, with
the leading NaN duplicated, so every downstream consumer sees a bogus extra element (and a length
that exceeds the input)
**Status:** present in MindSpore 2.9.0 (CPU); PyTorch's `unique_consecutive` returns the correct
lengths for the same inputs, and NumPy/PyTorch agree on the adjacent-equality semantics
**Found with:** the wave-173 runtime-selection grid (`artifact/hunt-v2/results/w173_*`), triaged by
Rondo's `check_consensus` oracle with a manual adjacent-equality reference; the certified reproducer
is `repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`unique_consecutive` removes runs of equal adjacent elements.  If the input **starts** with NaN,
MindSpore returns one element too many and repeats that NaN:

```python
import math, numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
T = lambda a: ms.Tensor(np.asarray(a), ms.float64)

ops.unique_consecutive(T([math.nan]))               # [nan, nan]              (input length 1)
ops.unique_consecutive(T([math.nan, 1.0]))          # [nan, nan, 1.0]         (input length 2)
ops.unique_consecutive(T([math.nan, math.nan]))     # [nan, nan, nan]         (input length 2)
ops.unique_consecutive(T([math.nan, 2.0, 3.0]))     # [nan, nan, 2.0, 3.0]    (input length 3)
ops.unique_consecutive(T([math.nan, 1.0, 2, 3]))    # [nan, nan, 1.0, 2.0, 3.0] (input length 4)
```

```text
input                    MindSpore 2.9.0            torch.unique_consecutive
[nan]                    [nan, nan]                 [nan]
[nan, 1.0]               [nan, nan, 1.0]            [nan, 1.0]
[nan, nan]               [nan, nan, nan]            [nan, nan]
[nan, 2.0, 3.0]          [nan, nan, 2.0, 3.0]       [nan, 2.0, 3.0]
[nan, 1.0, 2.0, 3.0]     [nan, nan, 1.0, 2.0, 3.0]  [nan, 1.0, 2.0, 3.0]
```

The overflow of the length is the proof that this is not a NaN-equality convention: no
adjacent-equality rule (or even the "NaN != NaN" rule PyTorch uses) can produce an output longer
than the input.  Inputs whose first element is not NaN are all correct, including ones that contain
NaN later (`[1.0, nan]`, `[2.0, nan, 3.0]`, `[2.0, 3.0, nan]`), `[inf, -inf, inf]`,
`[-0.0, 0.0] -> [-0.0]`, ordinary duplicates (`[1,1] -> [1]`, `[1,2,2,1] -> [1,2,1]`) and the empty
tensor.  float32 shows the same defect, and `ms.jit` (GRAPH) reproduces it.

## Root cause

The kernel initialises its comparison state from a sentinel and writes the first element
unconditionally; with a NaN first element the sentinel comparison is taken as a "change" as well, so
the first element is emitted twice and the output counter is one too large.  The fix is to seed the
state with the first element itself (or suppress the extra emit when the sentinel is NaN) and to
size the output from the number of actual changes.

## Impact

`unique_consecutive` is used to compress run-length-encoded sequences and to deduplicate sorted
inputs before ragged/segment operations.  A leading NaN (a missing or undefined measurement in the
first position) silently produces an output one element too long, which misaligns every subsequent
index, breaks `segment_*`/`unsorted_segment_*` consumers, and adds a spurious NaN to reductions.

## Suggested fix

Initialise the "previous" value with `input[0]` (and handle the initial emit explicitly), or special-
case NaN in the comparison; add a test with a leading NaN asserting `len(output) <= len(input)`.

## Controls (all in `repro.py`)

* Every non-leading-NaN case above matches the adjacent-equality reference, including inputs with
  NaN in other positions and the `[-0.0, 0.0] -> [-0.0]` and `[inf, -inf, inf]` cases.
* The empty tensor returns `[]`.
* float32 and `ms.jit` show the same leading-NaN defect, so it is not a dtype- or trace-specific
  artifact.

## Novelty

Searched the MindSpore tracker for `unique_consecutive`, `unique_consecutive nan`,
`repeat_interleave zero repeats` and `repeat_interleave empty`: no report exists for either
operator.  The campaign's other MindSpore findings are different operators (`ops.unique` of an empty
tensor is folded into MS-021, `ops.cumsum` empty input into MS-025).

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"NaN handling in uniqueness is unspecified."**  This is not a NaN-ordering question: the output
  is *longer than the input*, which no set of equality rules can justify, and PyTorch returns the
  expected lengths for every case.
* **"Only leading NaN is affected."**  Correct - and a leading NaN is a realistic pattern (a missing
  first sample), while the failure is silent and shape-changing.
