# `ops.fmod` / `ops.remainder` (and `mint`) return `NaN` for a finite dividend with an infinite divisor

**Component:** MindSpore 2.9.0 CPU, `mindspore.ops.fmod`, `mindspore.ops.remainder` and their
`mindspore.mint` equivalents, float32 and float64, PYNATIVE and GRAPH (`ms.jit`)
**Severity:** silent wrong value. `fmod(1.0, inf)` is `NaN` where C99, Python and NumPy return `1.0`
(the dividend); the floor-convention `remainder` likewise returns `NaN` where NumPy/Python return a
finite value (`1.0` for a positive dividend, `inf` for a negative one).  Any code path that can see
an infinite divisor - a normalised weight, a variance, a scale factor - silently poisons the result
with `NaN`.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** a cross-framework binary special-case grid (wave 161 of the bug hunt), then minimised
to the infinite-divisor cells with the standalone probe shipped next to this file.

---

## Summary

C99 specifies `fmod` completely: *"If `y` is ±Inf and `x` is finite, `fmod(x, y)` returns `x`"*.
MindSpore returns `NaN` for the whole family, in both namespaces and both dtypes:

| call (float64) | `ops` | `mint` | NumPy | Python |
|---|---|---|---|---|
| `fmod(1.0, inf)`   | `nan` | `nan` | `1.0` | `1.0` |
| `fmod(-1.0, inf)`  | `nan` | `nan` | `-1.0` | `-1.0` |
| `fmod(0.5, -inf)`  | `nan` | `nan` | `0.5` | `0.5` |
| `fmod(1e308, inf)` | `nan` | `nan` | `1e308` | `1e308` |
| `remainder(1.0, inf)` | `nan` | `nan` | `1.0` | `1.0` |
| `remainder(-1.0, inf)` | `nan` | `nan` | `inf` (floor convention) | `inf` |
| `remainder(1e308, inf)` | `nan` | `nan` | `1e308` | `1e308` |

The finite-divisor lane is correct, which is what makes this a special-case gap rather than a wrong
algorithm: `fmod(5,3)=2`, `fmod(-5,3)=-2`, `fmod(5,-3)=2`, `fmod(-5,-3)=-2`, and
`remainder(5,3)=2`, `remainder(-5,3)=1`, `remainder(5,-3)=-1`, `remainder(-5,-3)=-2` all match the
floor convention.  `fmod(inf, 1.0) = nan` is also correct (an infinite dividend is a domain error).

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(80 checks per replay, 2 fresh processes.)

```python
import numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x = ms.Tensor(np.asarray([1.0], np.float64), ms.float64)
y = ms.Tensor(np.asarray([np.inf], np.float64), ms.float64)
float(ops.fmod(x, y).asnumpy()[0])       # nan   <-- C99/NumPy/Python: 1.0
float(np.fmod(1.0, np.inf))              # 1.0
float(ops.remainder(x, y).asnumpy()[0])  # nan   <-- NumPy/Python: 1.0
```

What the reproducer pins:

| dimension | result |
|---|---|
| cells | 6 infinite-divisor pairs (`(1, +inf)`, `(-1, +inf)`, `(0.5, -inf)`, `(-0.5, -inf)`, `(1e308, +inf)`, `(1e-320, +inf)`) for both `fmod` and `remainder`, each pinned as `NaN`, as "NumPy is not NaN", and as "NumPy matches Python's convention" |
| namespaces | `mindspore.ops` and `mindspore.mint` both return `NaN` |
| dtypes | float32 and float64 |
| graph lane | `ms.jit(fmod)(1.0, inf)` and `ms.jit(remainder)(1.0, inf)` are `NaN` |
| controls | the eight finite-divisor cells above match the C99/floor values; `fmod(inf, 1.0)` and `fmod(nan, 1.0)` are `NaN` as the standard requires |

## Root cause (pointer)

The kernel computes the remainder through a quotient (`x - trunc(x/y) * y` for `fmod`,
`x - floor(x/y) * y` for `remainder`) without the C99 special case for an infinite divisor: `x / inf`
is `0`, and the subsequent `0 * inf` produces `NaN` before the subtraction.  Both kernels need the
standard guard (return `x` when `y` is infinite and `x` is finite) - the same guard exists in the
finite-divisor path only by accident of the arithmetic.  A regression test is
`ops.fmod(x, inf) == x` and `not isnan(ops.remainder(x, inf))` for `x in {1.0, -1.0, 1e308}`.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `fmod inf`, `fmod infinite`, `remainder inf`,
  `fmod nan` returns **nothing**.
* The campaign's other MindSpore `fmod` drafts cover different input classes: MS-001 is GRAPH-mode
  `fmod` *values* for large `|a/b|`, and MS-014 is the sign of a *zero result*
  (`fmod(-0.0, 1) = +0.0`).  Here the dividend is an ordinary finite number, the divisor is `±inf`,
  and the expected result is the dividend itself.
* Web search finds only the API pages for `mindspore.ops.fmod`/`remainder`, which document the
  element-wise formula and no infinite-divisor behaviour.
