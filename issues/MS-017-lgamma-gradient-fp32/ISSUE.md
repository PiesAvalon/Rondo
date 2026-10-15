# `ops.lgamma`'s gradient is computed in float32 while its forward value is genuine float64

**Component:** MindSpore 2.9.0 CPU, `mindspore.ops.lgamma` differentiated with `mindspore.grad`
(eager `PYNATIVE_MODE` and `GRAPH_MODE` via `ms.jit`), float64 and float32
**Severity:** silent reduced-precision gradient. `ms.grad(ops.lgamma)(x)` should be `digamma(x)`; the
float64 result is the float32 answer rounded into a double - relative error 1.4e-08 to 1.6e-06, and
**every** returned value is exactly float32-representable. The forward value is unaffected
(`ops.lgamma(x)` matches SciPy to 1.9e-16), so only gradient-based code (training losses, MLE of
gamma-family models, higher-order optimisers) loses digits.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** Rondo's `check_dtype_dispatch` property (the float64 observation is exactly
float32-representable while the correctly rounded float64 reference is not, and they differ far
beyond fp64 rounding) applied to the *backward* lane; the standalone probe is shipped next to this
file.

---

## Summary

`lgamma` is the log-gamma function, and its derivative is the digamma function.  MindSpore computes
the forward in float64 but the backward in float32:

| `x` | `ms.grad(lgamma)(x)` (float64) | `digamma(x)` (SciPy 1.17.1) | relative error | float32-representable |
|---|---|---|---|---|
| 0.5   | `-1.9635108709335327`  | `-1.9635100260214235`  | 4.3e-07 | yes |
| 1.0   | `-0.57721590995788574` | `-0.57721566490153287` | 4.2e-07 | yes |
| 1.5   | `0.036489915102720261` | `0.03648997397857652`  | 1.6e-06 | yes |
| 3.0   | `0.92278420925140381`  | `0.92278433509846713`  | 1.4e-07 | yes |
| 17.0  | `2.8035132884979248`   | `2.8035133283274605`   | 1.4e-08 | yes |
| 100.5 | `4.6051740646362305`   | `4.6051743525818454`   | 6.3e-08 | yes |
| 1e-3  | `-1000.5755615234375`  | `-1000.5755719318103`  | 1.0e-08 | yes |
| 1e4   | `9.2102909088134766`   | `9.2102903711428503`   | 5.8e-08 | yes |

The forward lane is the control, and it is correct to the last bits
(`lgamma(0.5) = 0.57236494292470008` vs SciPy `0.57236494292469997`, relative error 1.9e-16; four of
the eight cells are bit-identical).  The float32 lane of the same gradient returns the *same* value,
which is what the float64 lane is carrying:

```python
ms.grad(lgamma)(3.0)                      # 0.92278420925140381
float(np.float32(0.92278420925140381))    # 0.92278420925140381   <-- already float32
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(56 checks per replay, 2 fresh processes.)

```python
import numpy as np, mindspore as ms
from mindspore import ops
import scipy.special as sp
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

def scalar(v):
    return ops.lgamma(v.reshape(1)).reshape(()).astype(ms.float64)

g = float(ms.grad(scalar)(ms.Tensor(np.asarray([0.5], np.float64), ms.float64)))
g                        # -1.9635108709335327
float(sp.digamma(0.5))   # -1.9635100260214235
float(np.float32(g)) == g  # True  <-- the float64 gradient is a float32 value
```

What the reproducer pins:

| dimension | result |
|---|---|
| cells | 8 float64 points (0.5, 1.0, 1.5, 3.0, 17.0, 100.5, 1e-3, 1e4), each pinned as "not the analytic digamma", "relative error above 1e-9", "exactly float32-representable", and "the SciPy reference is not" |
| forward control | `ops.lgamma(x)` matches `scipy.special.gammaln` to 1.9e-16 on the same cells |
| float32 dispatch | every float64 gradient result satisfies `float(np.float32(g)) == g`; the float32 lane returns the identical value for the same `x` |
| graph lane | a retained `ms.jit(ms.grad(lgamma_scalar))` returns the same float32 values (`[-1.96351087, 0.92278421, 2.80351329]`, all float32-representable) |
| controls | `ms.grad` of `digamma`, `exp`, `log`, `sin`, `expm1` and `log1p` at the same points equals the analytic derivative **exactly** (relative error 0.0e+00), so this is not a global gradient mode |

## Root cause (pointer)

The `lgamma` backward kernel evaluates the digamma in single precision and returns it as a double, so
the extra 29 bits are padding rather than precision (the same signature as MS-011's forward).  The
neighbouring `digamma` op shows the lanes are implemented independently: its *forward* is float32
(the already-filed mindspore#400) while its backward is exact.  A fix is to make the `lgamma` grad
kernel use the float64 digamma path (or register a double kernel) and a regression test is
`ms.grad(lgamma)(3.0) == sp.digamma(3.0)` under `pytest.approx(rel=1e-14)` in float64.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `lgamma gradient`, `lgamma backward`,
  `lgamma float64`, `lgamma grad`, `digamma gradient` returns **nothing**.
* Web search finds only implementation tasks - [Gitee I1W0SU](https://gitee.com/mindspore/mindspore/issues/I1W0SU)
  `[MSP]Grad op for LGamma` (adding the gradient op) and
  [Gitee I4XJGL](https://gitee.com/mindspore/mindspore/issues/I4XJGL) (AICPU Lgamma development) -
  and the API documentation for `lgamma`/`digamma`, none of which mentions precision.
* The adjacent filed item is mindspore#400, `digamma`'s float32 *forward* (`digamma(1e-100) = -inf`);
  this report is the `lgamma` *backward*, on an operator whose forward is fp64 - a different kernel
  and a different lane.  The campaign's MS-011 is the float32 forward of `softplus`/`mish`;
  `softplus`'s gradient is also float32, but that is MS-011's operator and is recorded there, while
  `lgamma` is new.
