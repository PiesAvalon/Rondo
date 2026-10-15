# `mindspore.mint.nn.functional` activations raise `RuntimeError: The kernel ... unregistered` on CPU

**Component:** MindSpore 2.9.0 CPU (pip build), `mindspore.mint.nn.functional`, PYNATIVE mode
**Severity:** the torch-compatible activation API is unusable for nine activation functions: the call
returns a lazy `Tensor` and materialising it raises a kernel-registration error, while the identical
`mindspore.ops` call on the same input in the same process returns the correct value.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** a scan of the untested `mindspore.mint` namespace
(`artifact/hunt-v2/tools_ms_mint_probe.py`), minimised to one call per process.

---

## Summary

```python
import numpy as np, mindspore as ms
import mindspore.mint.nn.functional as F
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor(np.asarray([-1.0, 0.5], np.float32), ms.float32)

F.softplus(x)              # returns a Tensor without raising ...
F.softplus(x).asnumpy()    # RuntimeError: The kernel SoftplusExt unregistered.
ms.ops.softplus(x)         # 0.31326124 ... correct, in the same process
```

The error text and call stack are unambiguous:

```text
RuntimeError: The kernel SoftplusExt unregistered.
mindspore/ccsrc/pynative/utils/pyboost/pyboost_utils.cc:865 SelectKernel
```

Affected activations (float32, the dtype MindSpore documents as supported):

| `mint.nn.functional` op | reported kernel | `ms.ops` counterpart |
|---|---|---|
| `softplus` | `SoftplusExt` | correct |
| `leaky_relu` | `LeakyReLUExt` | correct |
| `elu` | `EluExt` | correct |
| `gelu` | `GeluExt` | correct |
| `relu6` | `Hardtanh` | correct |
| `hardtanh` | `Hardtanh` | correct |
| `selu` (fp32) | `SeLUExt` | correct |
| `mish` (fp32) | `MishExt` | correct |
| `logsigmoid` (fp32) | `LogSigmoid` | correct |

`relu`, `silu`, `hardswish`, `hardsigmoid` and `glu` work through the same layer, so this is a
missing-kernel set, not a broken module.  Two further `mint` operators hit the same gap and are part of
the same report - `mint.frac` (`The kernel Frac unregistered`) and `mint.median` (`The kernel
MedianExt unregistered`), plus `mint.sort`, which returns a `(values, indices)` tuple whose
materialisation raises `The kernel SortExt unregistered`.

The failure also **poisons the rest of the process**: in a fresh interpreter, calling `mint.sort(x)`
and then an unrelated `mint.abs(x)` makes the *second* call raise the stale `SortExt` error -
`SelectKernel` leaves the pyboost state unusable.  Any harness that compares several mint ops in one
process will therefore attribute the failure to the wrong operator (this cost the wave two false
tables before each op got its own process with the control measured first).  For float64, `selu`, `mish` and `logsigmoid` instead raise
`TypeError: ... supports types: Float16, Float32, BFloat16` (a documented dtype limit), and
`softshrink`/`hardshrink` reject float64 in **both** layers - so those rows are dtype limitations
rather than this defect.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(54 checks per replay; every op is measured in its **own process**, with the `ms.ops` counterpart
measured first in that same process as an in-process control.)

What the reproducer pins, per op:

| dimension | result |
|---|---|
| call vs materialise | `F.<op>(x)` returns a `Tensor`; `.asnumpy()` raises `RuntimeError` naming the kernel |
| control in the same process | `ms.ops.<op>(x).asnumpy()` succeeds and gives the expected value (`softplus(-1) = 0.313261`, `softplus(0.5) = 0.974077`, `leaky_relu(-1) = -0.2`, `gelu(-1) = -0.158655`) |
| working controls | `relu`, `silu`, `hardswish`, `hardsigmoid`, `glu` succeed in both layers and agree to fp32 precision |
| determinism | 20 fresh processes and two full replays produce the same error for `softplus` |

## Root cause (pointer)

The `mint.nn.functional` activations dispatch through the pyboost/op-plugin path to `*Ext` kernels
(`SoftplusExt`, `LeakyReLUExt`, `EluExt`, `GeluExt`, `MishExt`, `SeLUExt`, and the classic-named
`Hardtanh`/`LogSigmoid`), and those kernels are **not registered in the CPU build**:
`SelectKernel` in `mindspore/ccsrc/pynative/utils/pyboost/pyboost_utils.cc` fails and the error
surfaces only when the lazy tensor is materialised.  The classic `mindspore.ops` implementations of the
same functions are present and correct in the same process, so the fix is to register (or fall back to)
the equivalent kernels for the mint layer.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `Ext unregistered`, `kernel unregistered`,
  `mint leaky_relu` and `mint.nn.functional` returns exactly one hit:
  [#286](https://github.com/mindspore-ai/mindspore/issues/286) - "官网案例--快速入门报错
  RuntimeError: the kernel ArgMaxExt unregistered", a **documented example** that failed the same way
  in a 2.3.0 daily build and was **closed as completed** (fixed).  That confirms the class is treated
  as a bug and is fixable; no issue covers the activation family.
* The float64 `TypeError` rows (`selu`, `mish`, `logsigmoid` are Float16/Float32/BFloat16-only) are
  consistent with the docstrings and are recorded as limitations, not drafted.
