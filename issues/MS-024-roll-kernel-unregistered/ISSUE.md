# Core operators with no CPU kernel: `ops.roll` and `ops.take` raise `The kernel ... unregistered` for every input

**Component:** MindSpore 2.9.0, CPU, PYNATIVE and GRAPH: `ops.roll`, `mint.roll`, the
`Tensor.roll` method, and `ops.take`
**Severity:** the operator is unusable — a documented, exported API raises on every input, so any
model or preprocessing pipeline that rolls an array cannot run on CPU
**Status:** present in MindSpore 2.9.0 (CPU); NumPy, PyTorch, JAX and TensorFlow roll correctly
**Found with:** the wave-167 runtime/indexing grid (`artifact/hunt-v2/results/w167_*`), triaged by
Rondo's `check_consensus` oracle with NumPy as arbiter; the certified reproducer is `repro.py` in
this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

Every call to `ops.roll` fails with an unregistered-kernel error, for every dtype, shape, shift and
axis, in PYNATIVE and GRAPH:

```python
import numpy as np, mindspore as ms
from mindspore import ops, mint
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

x = ms.Tensor([1.0, 2.0, 3.0])
ops.roll(x, 1, 0)        # RuntimeError: The kernel Roll unregistered.
ops.roll(x, 0, 0)        # RuntimeError: The kernel Roll unregistered.
mint.roll(x, 1, 0)       # RuntimeError: The kernel Roll unregistered.
x.roll(1, 0)             # RuntimeError: The kernel Roll unregistered.
```

Checked combinations (all raise identically): `float64`/`float32`/`int32` 1-D, 2-D with
`axis=0` and `axis=(0,1)`, negative shifts, zero shifts, flattened (`axis=None`) rolling, a tensor
shift, `mint.roll` and the `Tensor.roll` method; eager GRAPH mode fails the same way.

The API itself is fully present - `ops.roll`/`mint.roll` are callable, exported and documented with
`shift`/`axis` - and every neighbouring op works on the same tensors (`ops.flip`,
`ops.cumsum`, `ops.reverse`), so the wheel simply has no Roll kernel registered for the CPU device.

**`ops.take` has the same gap.**  Its signature is `take(input, index)` (two positional arguments);
with a valid `int32`/`int64` index it raises `RuntimeError: The kernel Take unregistered.` for
`float64`, `float32` and `int32` inputs, 1-D and 2-D, in PYNATIVE.  NumPy, PyTorch, JAX and
TensorFlow all gather correctly.

## Impact

`roll` is a standard array-manipulation primitive (cyclic shifts in signal processing, feature
engineering, rolling-window construction, positional encodings).  On MindSpore CPU it is not merely
slow or wrong: it aborts with an internal `RuntimeError`, with no documented workaround (`ops.flip`
plus slicing only covers reversal, not an arbitrary cyclic shift).

## Suggested fix

Register the Roll CPU kernel (or gate the API so that the missing support is reported as an
unsupported-device error rather than an unregistered-kernel `RuntimeError`), and add a CPU unit
test for `ops.roll` with 1-D/2-D inputs and positive/negative shifts.

## Controls (all in `repro.py`)

* `numpy.roll` on the same arrays and shifts returns the correct values, and `numpy.take`
  gathers the same elements.
* `ops.flip`, `ops.cumsum` and `ops.reverse` work on the same tensors in the same process.
* The `Roll` symbols are exported and callable, so this is a kernel-registration failure rather
  than a missing Python API.

## Novelty

Searched the MindSpore tracker for `roll`, `ops.roll`, `kernel Roll`, `roll unregistered`: no report
exists.  The campaign's other MindSpore registration findings are the `mint` `*Ext` kernels
(MS-013, activations and `frac`/`median`/`sort`); `ops.roll` is a core operator and fails for every
input rather than only in the `mint` layer.

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Roll is not supported on CPU."**  If that is intended it must be reported as an unsupported
  device (as MindSpore does for Ascend-only ops); today the user-visible behaviour is an internal
  kernel-selection `RuntimeError`, and the operator is exported as if supported.
* **"Use `ops.flip`/slicing instead."**  Those express reversal, not a cyclic shift by an arbitrary
  amount, so they are not a workaround for the primitive.
