# MindSpore `ops.softplus` / `ops.mish` compute float64 in float32, and the value changes with the tensor width and with the process call history

**Component:** MindSpore 2.9.0, CPU, `mindspore.ops.softplus` and `mindspore.ops.mish` (PYNATIVE and
GRAPH / `ms.jit`), float64 inputs
**Severity:** silent accuracy loss plus a non-deterministic-looking result. A float64 op documented as
supporting float64 returns a **float32** answer, the answer for the same input changes at `n = 4`, and
it changes again after the fifth call to the same shape in one process.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** the wave-136 MindSpore GRAPH-mode width sweep
(`artifact/hunt-v2/tools_ms_graph_width_probe.py`), which found `softplus` varying with width, then
minimised with `tools_ms_fp64_width_precision.py` and certified with a reproducer.

---

## Summary

`ops.softplus` documents `Supported dtypes: GPU/CPU: float16, float32, float64`, accepts a `Float64`
tensor and returns a `Float64` tensor.  The value it returns is nevertheless a **float32** number:

```python
import numpy as np, mindspore as ms
from mindspore import ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x = ms.Tensor(np.full(4, 1.0, np.float64), ms.float64)
v = float(ops.softplus(x).asnumpy()[0])   # 1.3132610321044922 (or 1.3132611513137817)
float(np.float32(v)) == v                 # True  - a float32 value widened to float64
np.log1p(np.exp(1.0))                     # 1.3132616875182228 - the float64 answer
```

Every value the op produces round-trips through `np.float32` exactly, while the exact float64 value
does not; the error is 4.5-5.5 float32 ulp for `softplus(1.0)`.  `ops.log1p(ops.exp(x))` and
`ops.log(1 + ops.exp(x))` are **exact float64** at the same widths, so the loss is inside the
`Softplus` kernel rather than a global dtype policy.

Two further dependencies make the result a function of more than its input:

| dependency | observation (float64, cold process) |
|---|---|
| tensor width | `softplus(0.0)` is `0.6931471824645996` for `n <= 3` and `0.6931470036506653` for `n >= 4`; `softplus(1.0)` switches `1.3132611513137817 -> 1.3132610321044922`; `mish(1e-8)` switches `5.999999608263806e-09 -> 6.000000052353016e-09` |
| call history | calling the same `(op, shape)` repeatedly returns one value for the first five calls and a different value from the sixth on, permanently; the same happens under `ms.jit` |

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(75 checks per replay, measured in a fresh process; the cold-state width table is taken before any
shape has been called five times.)

| op, value | exact float64 | width <= 3 | width >= 4 |
|---|---|---|---|
| `softplus(0.0)` | `0.6931471805599453` | `0.6931471824645996` | `0.6931470036506653` |
| `softplus(1.0)` | `1.3132616875182228` | `1.3132611513137817` | `1.3132610321044922` |
| `softplus(-1.0)` | `0.31326168751822286` | `0.3132612407207489` | `0.3132612109184265` |
| `mish(1e-8)` | `6.000000032000001e-09` | `5.999999608263806e-09` | `6.000000052353016e-09` |

Call-history sequence for `softplus(1.0)` at `n = 4` in one process:

```text
call 1..5: 1.3132610321044922
call 6..:  1.3132611513137817      # permanent switch, no input changed
```

The same sequence under `ms.jit(ops.softplus)` shows the same switch, so there is no stable compiled
lane to fall back on.  Controls (all exact float64 in the same interpreter, at `n = 4` and `n = 8`):
`log1p(exp(0.0))`, `log(1 + exp(0.0))`, `sigmoid(0.0)`, and `tanh`/`exp`/`erf`/`sigmoid` at `0.5`.

## Root cause (pointer)

`Softplus` and `Mish` are dispatched on the CPU to a kernel that works in float32 for every input
dtype: the returned float64 tensor carries a value that is exactly representable in float32, and the
float32 lane returns the same numbers.  The kernel additionally has (at least) two branches selected
by the number of elements and a warm-up state, so the same mathematical input yields different
float32 results - a `3 -> 4` element boundary is consistent with a scalar-tail path next to a
four-wide vector path.  The float64 registration should either lower to a genuine double kernel or be
removed from the documented dtype list.

## Novelty

* No report was found for `softplus`/`mish` float64 accuracy: keyword searches on
  `mindspore-ai/mindspore` for `softplus float64`, `softplus precision`, `mish float64` return
  nothing, and a web search surfaces only
  [Gitee IBN718](https://gitee.com/mindspore/mindspore/issues/IBN718) - a feature request for
  `mint.nn.functional.softplus` to *support* float64 (a different namespace and a different
  question).
* The known "Double precision" fix campaign
  [Gitee PR !91540](https://gitee.com/mindspore/mindspore/pulls/91540) ("Resolve the precision bugs
  of multiple operators in the Double and Complex data type scenarios", merged 2026-01-19) touches
  `log10`, `log2`, `reciprocal`, `sinh` and `muls` - `softplus` and `mish` are not in its file list,
  and this build still shows the defect.
* This is a different observable from our own MS-009 (`PYNATIVE` arithmetic flushing subnormals after
  warm-up): MS-009 is about subnormal *flush* in `+ - * /`, while here an ordinary finite value of a
  named activation changes both with width and with call history, and the float32 dispatch is
  demonstrated independently by float32-representability.  The shared "warm-up" flavour is noted so a
  maintainer can collapse them if the same switch turns out to be responsible.
