# `ops.expm1` and `ops.log1p` collapse small arguments to a single quantum of 1.0 (or to zero)

**Component:** MindSpore 2.9.0 CPU, `mindspore.ops.expm1` and `mindspore.ops.log1p`, float64 and
float32, PYNATIVE and GRAPH (`ms.jit`)
**Severity:** silent wrong value with 100 % relative error for ordinary inputs. `expm1(1e-16)` returns
`0.0` where the value is `1e-16`; `expm1(1e-30)`, `expm1(-1e-30)`, `expm1(1e-300)`, `log1p(1e-30)` and
the rest of the small-argument range collapse the same way.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** a MindSpore signed-zero/boundary sweep of the unary family against NumPy and SciPy
(`artifact/hunt-v2/tools_ms_signed_zero_probe.py`), minimised to the cancellation cells.

---

## Summary

`expm1(x)` and `log1p(x)` exist precisely because `exp(x) - 1` and `log(1 + x)` lose every significant
digit for small `x`: the true value is `x` itself, and `1e-16` is an ordinary float64 (the smallest
normal double is `2.2e-308`).  MindSpore's kernels evaluate the naive forms, so the result is not `x`
but a single quantum of `1.0` (`2**-53 = 1.11e-16`) or exactly zero:

| call | MindSpore | SciPy 1.17 / NumPy |
|---|---|---|
| `expm1(1e-16)` | `0.0` | `1e-16` |
| `expm1(-1e-16)` | `-1.1102230246251565e-16` | `-1e-16` |
| `expm1(1e-17)`, `expm1(-1e-17)` | `0.0` | `+-1e-17` |
| `expm1(1e-30)`, `expm1(-1e-30)` | `0.0` | `+-1e-30` |
| `expm1(1e-100)`, `expm1(1e-300)` | `0.0` | the value |
| `expm1(-0.0)` | `+0.0` | `-0.0` |
| `log1p(1e-16)` | `0.0` | `1e-16` |
| `log1p(-1e-16)` | `-1.1102230246251565e-16` | `-1e-16` |
| `log1p(+-1e-30)`, `log1p(+-1e-100)`, `log1p(1e-300)` | `0.0` | the value |
| `log1p(-0.0)` | `+0.0` | `-0.0` |
| float32 `expm1(1e-8)` / `log1p(1e-8)` | `0.0` | `9.99999993922529e-09` / `9.999999950000001e-09` |
| float16 `expm1(1e-4)` / `log1p(1e-4)` | `0.0` | `1.0002e-04` |
| float16 `expm1(6e-5)` / `log1p(6e-5)` | `0.0` | `6.0022e-05` |
| float16 `expm1(6e-8)` / `log1p(6e-8)` | `0.0` | `5.9605e-08` |

The threshold is exactly where the intermediate rounds to `1.0`: about `1.1e-16` in float64 and `6e-8`
in float32.  A second facet shows at `1e-8` in **float64**: the returned
`9.99999993922529e-09` is exactly float32-representable while the true `1.0000000050000001e-08` is not
(Rondo's `check_dtype_dispatch` fires; relative error `1.1e-08`), so that lane is carrying float32
precision as well.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(90 checks per replay.)

```python
import numpy as np, mindspore as ms
from mindspore import ops
import scipy.special as sp
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x = ms.Tensor(np.asarray([1e-16], np.float64), ms.float64)
float(ops.expm1(x).asnumpy()[0])   # 0.0     <-- should be 1e-16
float(sp.expm1(1e-16))             # 1e-16
1.0 / float(ops.expm1(x).asnumpy()[0])  # inf  <-- the reciprocal is destroyed
```

What the reproducer pins:

| dimension | result |
|---|---|
| cells | 9 `expm1` and 7 `log1p` small arguments, each pinned on both sides plus the SciPy value and a "relative error above 1e-3" assertion |
| quantisation | every result is `0.0` or `+-2**-53`, never the value |
| sign | `expm1(-0.0)` and `log1p(-0.0)` are `+0.0`, SciPy returns `-0.0` |
| float64 lane | `expm1(1e-8)` is float32-representable, SciPy's is not, relative error `1.1e-08` |
| float32 lane | `expm1(1e-8)` and `log1p(1e-8)` are already `0.0` while `np.expm1(np.float32(1e-8))` is not |
| graph lane | `ms.jit` returns `0.0` for `1e-16` and `1e-30` in both functions |
| controls | `expm1(+-0.5)`, `expm1(1.0)`, `expm1(-0.999999)`, `log1p(+-0.5)`, `log1p(1.0)` match SciPy exactly, and `ops.exp`/`ops.log` are fp64-accurate |

The collapse also reaches the **float16** lane (found by the corrected wave-169 narrow-precision
grid): `expm1`/`log1p` of `1e-4`, `6e-5` (the fp16 minimum normal) and `6e-8` are all `0.0`,
although those arguments and results are fp16-representable (NumPy's float16 lane returns
`1.0002e-04`, `6.0022e-05`, `5.9605e-08`), and the `-0.0` arguments lose their zero sign there as
well; float16 `expm1(0.5)`, `expm1(1.0)`, `log1p(0.5)` and `log1p(1.0)` match NumPy.

## Root cause (pointer)

The CPU kernels compute `exp(x) - 1` and `log(1 + x)`.  When `|x|` is below the ulp of one, `exp(x)`
rounds to exactly `1.0` (so the subtraction gives `0.0`) or to `1 - ulp` (so it gives `-2**-53`), and
`log(1 + x)` loses the same bits; the float64 lane additionally carries float32 precision.  A dedicated
`expm1`/`log1p` (a `log1p`-based series for `|x| < 2**-4`, the platform `expm1`/`log1p`, or the
`log1p`/`expm1` XLA lowerings) fixes both, and a regression test is
`ops.expm1(x) == sp.expm1(x)` for `x` in `{1e-8, 1e-16, 1e-30, -1e-30}`.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `expm1`, `log1p`, `expm1 precision`,
  `log1p precision`, `expm1 zero` and `expm1 float64` returns **nothing**.
* A web search surfaces only [Gitee I64TA3](https://gitee.com/mindspore/mindspore/issues/I64TA3) -
  `Expm1` wrong for **complex128 with dim 0 on Ascend** - a different dtype and platform, and
  [Gitee PR !4878](https://gitee.com/mindspore/mindspore/pulls/4878) "Add custom op interface to
  replace expm1 and log", an implementation task, not a bug report.
* The campaign's other MindSpore drafts cover different operators (fmod GRAPH values, `polygamma` at
  infinity, `softplus`/`mish` dispatch, `mint` kernel registration), so this is a new site.
