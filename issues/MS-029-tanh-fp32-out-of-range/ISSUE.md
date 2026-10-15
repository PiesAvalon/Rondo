# `ops.tanh` in float32 leaves its codomain (`1.00000978` at `x = 5`) and its gradient flips sign

**Component:** MindSpore 2.9.0, CPU, `float32`: `ops.tanh`, `ops.Tanh()`, `nn.Tanh()` (and therefore
every activation layer that uses them), in `PYNATIVE_MODE`, `GRAPH_MODE` and `ms.jit`.  `ops.grad` of
the same op.
**Severity:** silent wrong values and a wrong-sign gradient.  `tanh` is bounded by `[-1, 1]`; the
float32 kernel returns **1.00000978** at `x = 5.0` and saturates to exactly `1.0` for `|x| >= 5.005`,
where a correctly-rounded float32 `tanh` only reaches `1.0` at `|x| >= 10.0`.  Because the backward
pass multiplies by `1 - tanh(x)^2`, the gradient becomes **negative** at `x = 4.975` and `x = 5.0`
(`-1.95504199e-05` where the exact derivative is `+1.81583231e-04`) and **exactly zero** for every
`|x| >= 5.005` (where the exact derivative is still `1.8e-04` at `5.005` and `8.2e-09` at `10.0`) -
so a trained float32 network gets an anti-gradient or no gradient at all in a completely ordinary
activation range.
**Status:** present in MindSpore 2.9.0 on CPU float32.  `float64` is accurate (relative error
`~1e-8` at `x = 10`, `~3e-7` at `12`), so the defect is in the float32 kernel/backward rule.
**Found with:** the wave-207 activation-tail grid (MindSpore Reference vs Fresh oracle: `ops.tanh`
forward and `ops.grad` against the cancellation-free `sech^2` reference and NumPy's correctly
rounded float32 `tanh`); certified reproducer `repro.py` in this directory (`--replays 2` ->
`RESULT: 0 failing replay(s).`, 71 checks).

---

## Summary

```python
import numpy as np, mindspore as ms
from mindspore import ops, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

ops.tanh(Tensor([5.0], ms.float32))          # 1.00000978   <- outside [-1, 1]
ops.tanh(Tensor([5.005], ms.float32))        # 1.0          <- saturated
np.tanh(np.float32(5.0))                     # 0.9999092    (correctly rounded)
np.tanh(np.float32(10.0))                    # 1.0          (first float32 input that reaches 1.0)

ops.grad(lambda z: ops.sum(ops.tanh(z)))(Tensor([5.0], ms.float32))
                                             # -1.95504199e-05   exact: +1.81583231e-04
ops.grad(lambda z: ops.sum(ops.tanh(z)))(Tensor([8.0], ms.float32))
                                             #  0.0              exact: +4.50140598e-07
```

Measured (CPU, MindSpore 2.9.0):

| x | `ops.tanh` float32 | `ops.grad` float32 | exact `sech^2` | NumPy float32 `tanh` |
|---|---|---|---|---|
| 4.5 | 0.999795198 | 4.09561180e-04 | 4.93517399e-04 | 0.999753237 |
| 4.75 | 0.999916494 | 1.67005244e-04 | 2.99362502e-04 | 0.999850333 |
| 4.9 | 0.999974489 | 5.10209247e-05 | 2.21781801e-04 | 0.999889135 |
| 4.95 | 0.999992371 | 1.52587309e-05 | 2.00678590e-04 | 0.999899685 |
| **4.975** | **1.000001073** | **-2.14576835e-06** | 1.90892314e-04 | 0.999904573 |
| **5.0** | **1.000009775** | **-1.95504199e-05** | 1.81583231e-04 | 0.999909222 |
| 5.005 | 1.0 | 0.0 | 1.79776610e-04 | 0.999910116 |
| 5.25 | 1.0 | 0.0 | 1.10139732e-04 | 0.999944925 |
| 6.0 | 1.0 | 0.0 | 2.45765474e-05 | 0.999987721 |
| 8.0 | 1.0 | 0.0 | 4.50140598e-07 | 0.999999762 |
| 10.0 | 1.0 | 0.0 | 8.24461446e-09 | 1.0 |

Pinned thresholds (0.005 grid): the first input whose float32 `tanh` exceeds 1 is `x = 4.975`; the
first that equals 1 is `x = 5.005`; the maximum value over the grid is **1.0000097751617432** at
`x = 5.0`; NumPy's correctly-rounded float32 `tanh` is still below 1 at `x = 9.0` and only reaches
1 at `x = 10.0`.  The float32 forward function is therefore **not monotone** at the defect
(`tanh(5.0) = 1.0000098 > tanh(5.005) = 1.0`).

The same kernel is reached through `ops.Tanh()`, `nn.Tanh()`, vector inputs, `ms.jit` and
`GRAPH_MODE`; `float64` matches the exact derivative to `1e-8` relative at `x = 10`.

## Root cause

`tanh`'s backward is `grad * (1 - tanh(x)^2)`, i.e. it is computed from the **forward** value, so the
forward kernel's error is amplified into a gradient error: once the kernel returns a value above 1
the factor `1 - tanh(x)^2` is negative, and once it returns exactly 1 the factor is exactly zero.
The underlying defect is the float32 CPU `tanh` kernel itself, which is neither correctly rounded
nor range-safe - it overshoots the codomain and saturates about five units too early.  `float64`
behaves correctly, and NumPy/Torch float32 `tanh` stay below 1 until `x ~ 9-10`, which is what the
IEEE-754 spacing near 1.0 (`2^-24 = 5.96e-08`) implies: correctly rounded `tanh(x)` reaches 1.0 when
`1 - tanh(x) < 2^-25`, i.e. around `x = 8.7` (NumPy reaches it at `x = 10.0` on this grid).

## Impact

* **Wrong-sign gradients in float32**, the default dtype for most MindSpore training: at `|x|` between
  4.975 and 5.005 an ascent step moves the weights the wrong way, and at `|x| >= 5.005` the gradient
  vanishes entirely.  Saturating activations are exactly where a saturating unit still needs a small
  gradient, and `|x| >= 5` is an ordinary pre-activation magnitude.
* **Values outside the documented range**: `ops.tanh` is documented as `(e^x - e^-x)/(e^x + e^-x)`
  with output in `(-1, 1)`.  Downstream code that relies on `|tanh| <= 1` (a `log(1 - tanh)` loss,
  `arctanh` of the output, probability rescaling, clipping-free reconstruction) gets a negative
  argument or an out-of-range value: `1 - tanh(5.0) = -9.8e-06` and `log` of that is `NaN`.
* The `float32` and `float64` lanes disagree qualitatively (accurate vs wrong sign), so a model
  behaves differently when its dtype is changed.

## Suggested fix

Use a correctly-rounded float32 `tanh` for the CPU kernel (clamping the result into `[-1, 1]` would
remove the sign flip but not the early saturation), and derive the backward from the forward value
only if it cannot exceed 1 in magnitude - alternatively compute the derivative as
`4 * sigmoid(2x) * sigmoid(-2x)`, which is the same expression without the cancellation.  A
regression test is self-contained:

```python
v = ms.Tensor([5.0], ms.float32)
assert float(ops.tanh(v).asnumpy()[0]) <= 1.0            # currently 1.0000097751617432
assert float(ops.grad(lambda z: ops.sum(ops.tanh(z)))(v).asnumpy()[0]) > 0.0   # currently -1.955e-05
```

## Controls (all in `repro.py`, 71 checks, 0 failing on each of 2 replays)

* The float64 lane is checked against the cancellation-free exact reference at `x = 5, 8, 10, 12`
  (relative error `<= 1e-6`), and the float64 forward value is checked against NumPy, so the defect
  is specific to the float32 kernel.
* NumPy's correctly-rounded float32 `tanh` is checked at the same inputs: it stays below 1 wherever
  MindSpore exceeds it, is still below 1 at `x = 9`, and only reaches 1 at `x = 10`.
* The exact derivative is the float64 `4 e^{-2|x|}/(1+e^{-2|x|})^2`, checked positive at every case;
  the relative error of the zero-gradient rows is asserted to be exactly 1.0.
* Both other entry points (`ops.Tanh()`, `nn.Tanh()`), a vector input, `ms.jit` and `GRAPH_MODE` are
  checked to show the same values, so this is the op's kernel and not one API wrapper.
* `float16` is included as a **control, not a defect**: its spacing near 1.0 is `4.9e-04`, so
  `tanh(5.0)` legitimately rounds to 1.0 there.
* Downstream impact is asserted (`1 - tanh(5.0) < 0`, `clip_by_value` silently rewriting the value).

## Novelty

MindSpore's tracker lives on Gitee; searching it (`mindspore/mindspore` issues for `tanh`,
`tanh 精度`, plus web search) returns:

* [IBNGQ1](https://gitee.com/mindspore/mindspore/issues/IBNGQ1) - "`mint.nn.functional.tanh` has a
  large float32 precision error on **910B**" (closed 2025-07-09): a different API (`mint`), a
  different backend (Ascend), and no out-of-range value or gradient sign;
* [I7OZ6C](https://gitee.com/mindspore/mindspore/issues/I7OZ6C) - "`tanh` dynamic test case has a
  precision problem on CPU" (closed 2023, marked DONE, body contains no reproducer, no numbers and
  no out-of-range observation; the issue is flagged as content-restricted on Gitee);
* [PR !84868](https://gitee.com/mindspore/mindspore/pulls/84868) - "insert cast for reduced types in
  `tanhgrad`" (a dtype-cast fix for the backward, not the float32 forward kernel);
* the campaign's own MS-009 (subnormal flush), MS-011 (float64 activations dispatched to the float32
  kernel) and MS-021 (complex binary ops) cover different behaviours, and none of them reports a
  `tanh` output outside `[-1, 1]`.

No item describes the observable reported here: a float32 `tanh` above 1, saturation at
`|x| >= 5.005`, and the resulting negative/zero gradient in the current 2.9.0 release.  If a
maintainer considers the closed 2023 CI case to cover it, the report is still the only reproducible
statement of the defect with thresholds and an exact reference.

## Environment

MindSpore 2.9.0, CPU, Python 3.12.13, NumPy 2.5.3, WSL2 (Linux 6.6.87.2) x86_64, `PYNATIVE_MODE` and
`GRAPH_MODE`.  Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Float32 tolerance near saturation is expected to be poor."**  Poor accuracy is not the same as
  leaving the codomain: `tanh` cannot exceed 1 for any real input, and the returned `1.00000978`
  makes the analytic derivative negative.  The correctly-rounded float32 value at the same point is
  `0.999909222`, which is `1.1e-04` below 1 - far above the `6e-08` float32 spacing, so the error is
  a kernel defect, not rounding.
* **"The derivative really is ~0 at large `|x|`."**  At `x = 5` the exact derivative is
  `1.8e-04` - five orders of magnitude above zero - and the returned value is *negative*, so the
  failure is qualitative (wrong sign) before it becomes zero.
* **"Use float64."**  float32 is the default dtype for training in MindSpore, and the campaign
  reports the float64 lane as correct; the point is that the float32 lane is wrong.
* **"This is the same as the other frameworks' tanh tail."**  The tail class (zero derivative at
  `|x| >= 20`) also exists in PyTorch/JAX/TF and is already reported there.  This report is a
  different observable: MindSpore's **forward** value leaves `[-1, 1]` at `x = 5`, saturates at
  `x = 5.005`, and the gradient changes sign - none of which happens in the other frameworks at
  those inputs.
