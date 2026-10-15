# `ops.copysign` ignores the sign of a negative-zero second argument

**Component:** MindSpore 2.9.0 CPU, `mindspore.ops.copysign`, float32 and float64, PYNATIVE and
GRAPH (`ms.jit`)
**Severity:** silent wrong sign.  `copysign(x, y)` is documented to return `|x|` with the sign of
`y`, and a negative zero has a sign; MindSpore returns a **positive** result whenever `y` is `-0.0`:
`copysign(2.5, -0.0) = +2.5` (should be `-2.5`), `copysign(inf, -0.0) = +inf` (should be `-inf`),
`copysign(0.0, -0.0) = +0.0` (should be `-0.0`).  The observable consequence is a flipped sign on an
ordinary value and `1/copysign(0.0, -0.0)` becoming `+inf` instead of `-inf`.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** the cross-framework binary special-case grid of wave 161 (where every other
framework and NumPy/Python transfer the `-0.0` sign correctly), then minimised with the probe shipped
next to this file.

---

## Summary

`copysign` exists precisely to *transfer a sign*, including the sign of a zero:

| call (float64) | MindSpore | NumPy / Python |
|---|---|---|
| `copysign(1.0, -0.0)`  | `1.0`  | `-1.0` |
| `copysign(2.5, -0.0)`  | `2.5`  | `-2.5` |
| `copysign(-2.5, -0.0)` | `2.5`  | `-2.5` |
| `copysign(inf, -0.0)`  | `+inf` | `-inf` |
| `copysign(-inf, -0.0)` | `+inf` | `-inf` |
| `copysign(0.0, -0.0)`  | `+0.0` | `-0.0` |
| `copysign(-0.0, -0.0)` | `+0.0` | `-0.0` |
| `copysign(1.0, 0.0)`   | `1.0`  | `1.0` (control) |
| `copysign(1.0, -1.0)`  | `-1.0` | `-1.0` (control) |
| `copysign(-0.0, 1.0)`  | `+0.0` | `+0.0` (control) |

Only a **negative-zero second argument** fails: every non-zero sign is transferred correctly, and a
negative-zero *first* argument is handled too (`copysign(-0.0, 1.0) = +0.0`).  The defect reproduces
in float32 and float64 and under `ms.jit`.  `mindspore.mint` has no `copysign` in 2.9.0, so this is
about `mindspore.ops.copysign`.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(43 checks per replay, 2 fresh processes.)

```python
import numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

float(ops.copysign(ms.Tensor(np.asarray([2.5], np.float64), ms.float64),
                   ms.Tensor(np.asarray([-0.0], np.float64), ms.float64)).asnumpy()[0])
# 2.5            <-- should be -2.5
float(np.copysign(2.5, -0.0))   # -2.5
```

What the reproducer pins:

| dimension | result |
|---|---|
| cells | eight negative-zero second arguments (`1.0`, `2.5`, `-2.5`, `±inf`, `±0.0`, `1`), each pinned as "the sign is positive (the bug)", as "the sign differs from Python's", and on Python's value |
| dtypes / modes | float32 and float64, PYNATIVE and `ms.jit` |
| consequence | `copysign(0.0, -0.0)` returns `+0.0`, so the reciprocal is `+inf` instead of `-inf` |
| controls | `copysign(1.0, 0.0) = 1.0`, `copysign(1.0, -1.0) = -1.0`, `copysign(-1.0, 1.0) = 1.0`, `copysign(0.0, -1.0) = -0.0`, `copysign(1.0, -2.0) = -1.0`, `copysign(-0.0, 1.0) = +0.0`, `copysign(nan, -0.0) = nan` |

## Root cause (pointer)

The kernel derives the sign from a comparison or a bitwise sign extraction that treats `-0.0` as
non-negative (`y < 0` is false for `-0.0`; `y >= 0` and `y == 0` are true for both zeros), instead of
testing the sign bit (`signbit(y)` / the top bit of the representation).  The fix is to take the sign
from the bit pattern - e.g. copy the sign bit of `y` onto `|x|` - and a regression test is
`ops.copysign(x, -0.0) == -abs(x)` and `copysign(0.0, -0.0)` having a set sign bit.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `copysign negative zero`, `copysign -0.0`,
  `copysign zero`, `ops.copysign` returns **nothing**.
* The campaign's MindSpore sign drafts cover other operators and observables: MS-014 is the sign of a
  `fmod`/`remainder` *zero result* and MS-016 is the second-kind Bessel functions on negative
  arguments; neither touches `copysign`.
* A web search finds only the `mindspore.ops.copysign` API page, which documents "returns a value with
  the magnitude of `x` and the sign of `y`" - exactly the contract that is violated here.
