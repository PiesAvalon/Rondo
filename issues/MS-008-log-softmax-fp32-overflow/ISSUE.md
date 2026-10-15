# `ops.log_softmax` returns `nan`/`-inf` for float64 logits above `FLT_MAX` while `ops.softmax` is correct

**Component:** MindSpore 2.9.0, `mindspore.ops.log_softmax`
**Severity:** silent wrong result (`nan` where the exact value is `0`, `-inf` where it is finite)
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo fp64 accuracy audit (`artifact/hunt-v2/ms_precision_nn.py`)

---

## Summary

`log_softmax([s, 0, -s])` is `[0, -s, -2s]` up to float64 rounding.  MindSpore is correct up to
`s = 1e38` and returns `[nan, -inf, -inf]` from `s = 1e39` — exactly once the intermediate leaves
the float32 range — while `ops.softmax` on the same input is correct at `1e100`:

```python
import numpy as np, mindspore as ms
from mindspore import ops

x = ms.Tensor([1e39, 0.0, -1e39], ms.float64)
ops.log_softmax(x, -1)   # [nan, -inf, -inf]      (exact: [0, -1e39, -2e39])
ops.softmax(x, -1)       # [1.0, 0.0, 0.0]        (correct)
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| `s` | `ops.log_softmax` | exact | |
|---|---|---|---|
| `1e20` | `[0, -1e20, -2e20]` | same | agrees |
| `1e38` | `[0, -1e38, -2e38]` | same | agrees |
| `1e39` | **`[nan, -inf, -inf]`** | `[0, -1e39, -2e39]` | broken |
| `1e100` | **`[nan, -inf, -inf]`** | `[0, -1e100, -2e100]` | broken |

Below the cliff the same fp32 intermediate shows up as ordinary precision loss: at `s = 1e20` the
returned `-1e20` term carries ~`1e-8` relative error instead of float64's `1e-16`.

Controls: `ops.softmax` is correct at every scale (including `1e100`), `ops.cross_entropy` on the
same large logits returns the correct `0`, and at ordinary magnitudes `log_softmax` matches the
reference to ~5e-6 (fp32-level, not fp64).

## Root cause (pointer)

The stabilised `log_softmax` path forms `exp(x - max(x))` (and/or the log-sum-exp term) in float32
for float64 inputs: once `x` exceeds `FLT_MAX` the subtraction/accumulation overflows and the
result becomes `nan`/`-inf`, whereas the mathematical result `x - logsumexp(x)` is small and
representable.  `ops.softmax` (which only needs the normalised ratios) stays correct because its
intermediate stays bounded — that asymmetry is what makes this a defect rather than a dtype
convention.

## Impact

`log_softmax` is the numerically stable primitive used by cross-entropy-style losses and by any
log-probability computation.  With logits scaled above `3.4e38` (large-magnitude inputs, or a
model that has diverged) float64 code gets `nan` instead of a finite log-probability, which
propagates to every downstream gradient; the fact that `softmax` still looks healthy makes it hard
to localise.

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `log_softmax nan`, `log_softmax precision` and
  `log_softmax inf` return no matching report.
* `ops.softmax` and `ops.cross_entropy` are correct on the same inputs, which rules out a general
  float64 handling issue and points at the `log_softmax` kernel.

## Files

* `repro.py` — self-checking reproducer (scale scan, softmax/cross-entropy controls, magnitude
  check, GRAPH lane)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
