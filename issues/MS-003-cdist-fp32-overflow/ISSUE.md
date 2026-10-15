# `ops.cdist` overflows to `inf` above a distance of ~1.8e19 for float64 inputs — the squared distance is accumulated in float32

**Component:** MindSpore 2.9.0, `mindspore.ops.cdist` (Cdist kernel)
**Severity:** silent wrong result (`inf` instead of a perfectly representable float64 number)
**Status:** present in MindSpore 2.9.0 (CPU, PYNATIVE and GRAPH)
**Found with:** Rondo NN-op differential with NumPy as reference (`artifact/hunt-v2/ms_nn_fuzz.py`)

---

## Summary

`ops.cdist(x1, x2, p)` returns `inf` as soon as the pairwise distance exceeds roughly `1.8e19`, even
for float64 inputs, while MindSpore's own `ops.pdist` returns the correct value for the same data up
to `1e154`, and NumPy/PyTorch return the correct value at every magnitude:

```python
import numpy as np, mindspore as ms
from mindspore import ops

A = ms.Tensor([[0.0, 0.0], [1e20, 0.0]], ms.float64)
ops.pdist(A, 2.0)                      # [1e20]   correct
ops.cdist(A, A, 2.0)                   # [[0, inf], [inf, 0]]   <-- inf instead of 1e20
np.linalg.norm([1e20, 0.0] - [0.0, 0.0])   # 1e20
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

Threshold scan for `d = |a - b|`, float64 (`A = [[0,0],[d,0]]`):

| `d` | `ops.pdist(A)` | `ops.cdist(A, A)[0,1]` | NumPy | |
|---|---|---|---|---|
| `1e18` | `1e18` | `1e18` | `1e18` | agrees |
| `1e19` | `1e19` | `1e19` | `1e19` | agrees |
| `1e20` | `1e20` | **`inf`** | `1e20` | cdist wrong |
| `1e38` | `1e38` | **`inf`** | `1e38` | cdist wrong |
| `1e77` | `1e77` | **`inf`** | `1e77` | cdist wrong |
| `1e150` | `1e150` | **`inf`** | `1e150` | cdist wrong |
| `1e154` | `1e154` | **`inf`** | `1e154` | cdist wrong |
| `1e155` | `inf` | `inf` | `1e155` | both overflow (float64 square limit) |

The overflow is **not** specific to `p = 2`; the L1 distance and the `inf`-norm overflow the same
way, which rules out a square-root overflow and points at the accumulation:

| `p` | `ops.cdist` at `|a - b| ≈ 1e100` | exact value (float64-representable) |
|---|---|---|
| `1.0` | **`inf`** | `2e100` |
| `2.0` | **`inf`** | `1.41e100` |
| `inf` | **`inf`** | `1e100` |

The controlling quantity is the standard float32 range: `3.4e38`.  `(1.8e19)² ≈ 3.4e38`, and every
case above overflows exactly where the *accumulated* value (squared distance, or the L1 sum) leaves
float32, while it is still far inside float64.  For float32 inputs the same threshold is expected
behaviour, and `ops.pdist` does not show it at all.

The reduced precision is directly visible **below** the overflow point too: `ops.cdist` returns
`float32(1e15) = 999999986991104` instead of `1e15`, while `ops.pdist` returns the exact float64
value — so it is a precision defect, and the overflow is only its extreme symptom.

## Root cause (pointer)

The `Cdist` kernel accumulates the pairwise distance in float32 regardless of the input dtype
(float64 inputs are converted down for the accumulation and not converted back with a stable
formula).  The sibling `Pdist` kernel and the reference implementations use a stable,
dtype-preserving accumulation, which is why only `cdist` is affected.  A fix would either keep the
accumulation in the input dtype or use the scaling trick that `nrm2` uses.

This is the same *shape* as **MS-001** (`fmod` computed in float32 in GRAPH mode) and suggests the
reduced-precision-intermediate class is worth auditing across MindSpore kernels.

## Impact

Squared-distance / distance computations with large-magnitude features (un-normalised embeddings,
monetary or physical units, features in the 1e19..1e308 range, or any code that has been scaled up
before differencing) silently become `inf` in float64 code.  Downstream that poisons clustering,
k-NN, RBF kernels, contrastive losses and gradient computation with `nan`/`inf`, and it is
invisible in small-magnitude unit tests.  The inconsistency with `pdist` in the same framework
makes it easy to hit by swapping one for the other.

## Novelty assessment

* Tracker searches in `mindspore-ai/mindspore` for `cdist inf`, `cdist overflow`,
  `cdist float32 precision` and `cdist wrong result` return no matching report.
* The only `cdist` issue is **#361** (open), which reports that broadcastable batch dimensions
  throw an exception on CPU — a shape/API failure, not an overflow in the value path.
* Three independent references agree with each other and disagree with MindSpore: MindSpore's own
  `pdist`, NumPy's `nrm2`-based `norm`, and `torch.cdist` (all return the finite value).

## Files

* `repro.py` — self-checking reproducer (threshold scan, all three `p` values, PYNATIVE + GRAPH,
  `pdist`/NumPy controls, float32 comparison)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
