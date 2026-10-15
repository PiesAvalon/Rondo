# PYNATIVE element-wise arithmetic silently flushes subnormals to zero after the first few operations in a process

**Component:** MindSpore 2.9.0, CPU, PYNATIVE mode — `+`, `-`, `*`, `/` on `float32` (and `float64`)
tensors.  The compiled (`ms.jit`) path is unaffected, and `NumPy` in the same process is unaffected.
**Severity:** silent wrong result — a representable subnormal becomes `0.0` (flush-to-zero), and it
is *state dependent*: the same expression with the same inputs returns a different value depending on
how many tensor operations the process has already executed.
**Status:** present in MindSpore 2.9.0 (CPU); found with the Rondo three-oracle sweep
(`artifact/hunt-v2/tools_ms_graph_oracle.py`), which compares PYNATIVE (Reference) against a fresh
`ms.jit` (Fresh) and a retained callable (Persistent).

---

## Summary

`float32(1e-38)` is a subnormal number: it is representable (the smallest `float32` subnormal is
`1.4e-45`), and NumPy, the fp64 reference and MindSpore's own first few evaluations all return
`1e-38` for `1e-38 + (-0.0)`.  After a handful of operations in the same process, MindSpore's eager
path starts returning `0.0` for the very same expression, and keeps doing so:

```python
import numpy as np, mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

def expr():
    a = ms.Tensor(np.asarray([1e-38] * 4, np.float32))
    b = ms.Tensor(np.asarray([-0.0] * 4, np.float32))
    return float((a + b).asnumpy()[0])

for k in range(6):
    print(k, repr(expr()))
```

```text
0 9.999999350456404e-39
1 9.999999350456404e-39
2 9.999999350456404e-39
3 9.999999350456404e-39
4 9.999999350456404e-39
5 0.0                    <-- the same expression, same inputs, now wrong
6 0.0
...
```

Both values cannot be right; `9.999999350456404e-39` is the correctly-rounded result and
`0.0` is a flush-to-zero of a representable number.  Nothing in the program changed the inputs,
the dtype, the context or the expression.

The state also switches **back**, so the same program can return first one value and then the other
and then the first again.  A 20-call window in one of the replay processes looked like this:

```text
ok ok ok ok ok ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ok ok ok ok ok
```

and calling `ms.jit` on the same expression resets the eager side to the correct behaviour for a
while (which is why the two oracles have to be compared factually rather than interleaved).  The
reproducer therefore asserts the invariant that cannot be argued with - the identical call returns
two different values within one process - instead of any particular sequence.

## Reproduction

`python repro.py --replays 4` → `RESULT: 0 failing check(s).` (each replay runs in its own
process; 4/4 replays passed)

The sequence above is not specific to `add`, to this shape, or to `float32`:

| probe (fresh process, 4-element tensors) | before warm-up | after warm-up | correct value |
|---|---|---|---|
| `1e-38 + (-0.0)` (result subnormal) | `9.999999350456404e-39` | **`0.0`** | `9.999999350456404e-39` |
| `1e-38 / 1e-5` (result subnormal) | `9.999999319067318e-34` | **`0.0`** | `9.999999319067318e-34` |
| `1e-38 * 1e-38` (result underflows to 0 anyway) | `0.0` | `0.0` | `0.0` |
| `1e-38 * 1e10` (subnormal *input*, normal result) | `9.999999429857661e-29` | `0.0` (later) | `9.999999429857661e-29` |
| `float64`: `1e-320 + (-0.0)` | `1e-320` | `0.0` | `1e-320` |

Two distinct stages are visible on a 4-element workload (the exact call index depends on how many
MindSpore operations the process performed earlier, which is itself part of the problem):

* after a handful of operations, subnormal **results** are flushed to zero — the classic FTZ
  behaviour (`1e-38 + (-0.0)` and `1e-38 / 1e-5` both become `0.0`);
* slightly later, subnormal **inputs** are also read as zero — DAZ (`1e-38 * 1e10` becomes `0.0`
  even though its result `1e-28` is an entirely normal number).

## What it is *not*

* **Not the process floating-point environment.**  Pure NumPy subnormal arithmetic in the same
  process, before and after the change, keeps returning the subnormal — for 1-element, 64-element and
  1024-element operands (`numpy` would be affected by MXCSR FTZ/DAZ, so MXCSR was not touched):

  ```text
  after ms add #3    numpy n=1:9.999999e-39  n=64:9.999999e-39  n=1024:9.999999e-39   ms eager add:0.000000e+00
  ```

  The flush therefore lives inside MindSpore's own kernels/dispatch, not in the CPU mode.
* **Not the compile step.**  `ms.jit` of the same expression on the same subnormal inputs keeps
  returning the correct subnormal while eager has already flushed it, so in one process the two
  Rondo oracles disagree — which is exactly how this was found.
* **Not an input-tensor mutation.**  The input tensors still hold `1e-38` when the output comes back
  as `0.0` (asserted in the reproducer).

## Root cause (pointer)

The installed binaries carry the FTZ/DAZ machinery: `libmindspore_ops_cpu.so`, `libakg.so` and the
GPU `libakg.so` all contain the strings `FTZ` and `denormal`, and MindSpore has deliberately enabled
denormals-zero in its SSE kernels before ([gitee PR 67564, "setting DAZ FTZ flags"](https://gitee.com/mindspore/mindspore/pulls/67564.patch),
which adds `_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)` to the conv/matmul intrinsics).
The state dependence points at the eager dispatch: the first evaluations of an op use one kernel,
and after warm-up/autotuning the op is dispatched to an FTZ/DAZ-enabled (AKG-generated) kernel which
is then used for the rest of the process.  The report deliberately stops at this boundary; a
maintainer can confirm it by dumping the selected CPU kernel for `Add` before and after warm-up.

## Impact

Subnormal values are ordinary data: probabilities that underflow gradually, gradients of very small
activations, and physics/scientific inputs with wide dynamic range.  The failure mode is the worst
kind — a plausible finite `0.0` instead of a tiny non-zero number, so `isfinite`-style checks do not
catch it, and it silently changes downstream results.  Because the switch depends on how many
operations the process has already performed, the same script can produce different results in
different runs or after any unrelated change earlier in the program, which makes the behaviour appear
non-deterministic and is very hard to bisect from user code.

## Novelty assessment

* Searches on the MindSpore GitHub mirror (`mindspore-ai/mindspore`) for `denormal`, `subnormal`,
  `flush to zero`, `FTZ`, `denormals are zero` return no matching report (the only `subnormal` hit is
  `#371 ops.signbit returns False for negative zero`, unrelated).
* Searches on the primary Gitee tracker (`mindspore/mindspore`) for `denormal`, `FTZ`,
  `flush zero`, `非规格化`, `次正规` return no matching report.
* The only relevant upstream artefact is the intentional DAZ/FTZ enablement inside conv/matmul SSE
  intrinsics (PR 67564).  That PR is about specific matmul/conv kernels; it does not cover
  element-wise `+ - * /`, does not mention fp64, and does not describe the warm-up-dependent switch
  reported here.  If a maintainer considers subnormal flushing acceptable for those kernels, the
  element-wise case is still a bug because it is silent, applies to fp64 as well, and changes
  behaviour mid-process.

## Files

* `repro.py` — self-checking reproducer (flip detection, NumPy control in the same process, DAZ vs
  FTZ stages, fp64 lane, eager-vs-jit disagreement, one process per replay)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment

## Additional evidence: what does *not* trigger the switch

A cross-framework sweep (`artifact/hunt-v2/tools_denormal_state_probe.py`, one warmer per fresh
process, every sample encoded as IEEE bits) narrowed the trigger.  In MindSpore 2.9.0, CPU, PYNATIVE:

| warmer (fresh process, before/after probes are subnormal) | flips to `+0.0`? |
|---|---|
| 200 iterations of `(a + b) * 0.5` on 4-element tensors | **yes** (result probe and subnormal-input probe) |
| 200 iterations of `relu` | no |
| 200 iterations of `exp` | no |
| 200 iterations of `softmax` | no |
| 200 iterations of 64x64 `matmul` | no |
| 100 iterations of `Conv2d` | no |
| an `ms.jit` call | no |
| NumPy arithmetic in the same process | never (stays subnormal) |

So the state change is tied to the **element-wise eager add/multiply path**, not to "any operation",
and the pre-warm-up state is stable rather than racy: the same probe stays subnormal for a whole
process when only non-element-wise ops run.  The same tool run under PyTorch and JAX does not flip
the probe with any of these warmers, and under TensorFlow every probe is already flushed at baseline
by the documented `port::ScopedFlushDenormal` policy.
