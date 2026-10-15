# GRAPH mode: a bfloat16 constant in a complex expression fails to compile (`infer_value_for_Cast`)

**Component:** MindSpore 2.9.0, CPU, GRAPH mode (`ms.jit`), `bfloat16` x `complex64`/`complex128`;
PYNATIVE mode is unaffected
**Severity:** hard compile failure of an otherwise valid graph — the same expression works eagerly and
works in GRAPH mode when the bfloat16 operand arrives as an input, so a whole `ms.jit` function is
uncompilable purely because a bfloat16 value is a captured constant
**Status:** present in MindSpore 2.9.0 (CPU); `np.can_cast(bfloat16, complex64, 'unsafe')` is `False`
in the numpy shipped with MindSpore, and `infer_value_for_Cast` folds the constant through numpy
**Found with:** the wave-187 Rondo graph/constant-folding grid — the Fresh oracle (recompile in a new
process) and the Persistent oracle disagree with the Reference (eager) oracle on the same expression,
and the divergence survived two process-isolated replays; certified reproducer is `repro.py` in this
directory (`--replays 2` -> `RESULT: 0 failing replay(s).`, 33 checks)

---

## Summary

`complex64 + bfloat16` is supported: PYNATIVE mode computes it, and `ms.jit` computes it whenever the
bfloat16 value is a graph **input**.  If the bfloat16 value is baked into the graph instead — a
captured global/closure constant, or a `Tensor(..., ms.bfloat16)` built inside the traced function —
compilation of the whole graph fails:

```python
import numpy as np, mindspore as ms
from mindspore import Tensor, ops
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

X = Tensor(np.asarray([1.0, 2.0], np.complex64), ms.complex64)
K = Tensor(np.asarray([1.0, 2.0], np.float32), ms.bfloat16)

def f(x):
    return x + K                      # K is a captured constant -> graph constant

f(X)                                  # [2.+0.j 4.+0.j]        (PYNATIVE: correct)
ms.jit(f)(X)
# TypeError: Cannot cast array data from dtype(bfloat16) to dtype('complex64')
#            according to the rule 'unsafe'

# in-body construction fails the same way
def g(x):
    return x + Tensor(np.asarray([1.0, 2.0], np.float32), ms.bfloat16)
ms.jit(g)(X)                          # same TypeError

# scalar constant variant
def h(x):
    return x + Tensor(True, ms.bfloat16)
ms.jit(h)(X)                          # TypeError: Cannot cast data from bfloat16 to complex64.

# but the identical dtypes work when bfloat16 is an argument
def two(x, y):
    return x + y
ms.jit(two)(X, K)                     # [2.+0.j 4.+0.j]        (GRAPH: correct)
```

The failure is not a kernel limitation: `ms.jit` executes `complex + bfloat16` correctly above, and
`ops.add`, `mint.add` and `ops.cast(..., ms.complex64)` all show the same split between a constant
and an argument.  Only the **bfloat16** constant is affected — captured `float16`, `float32`,
`float64`, `int8/32/64` and `complex64` constants all promote to complex64 without error, and a
captured bfloat16 constant with a `float32` input also works.

## Root cause

`infer_value_for_Cast` in
`mindspore/ops/operations/manually_defined/ops_def.py` implements the constant fold of a `Cast` as

```python
np_dst_type = mstype._dtype_to_nptype(dst_type)
...
value = Tensor_(x.asnumpy().astype(np_dst_type), dtype=dst_type)
```

When the promotion of `complex64 + bfloat16` is lowered, the bfloat16 constant is cast to the
promoted type `complex64`, and numpy refuses it:

```python
>>> B = ms.Tensor(np.asarray([1., 2.], np.float32), ms.bfloat16)
>>> B.asnumpy().dtype
dtype('bfloat16')                                     # MindSpore's own numpy bf16 dtype
>>> np.can_cast(B.asnumpy().dtype, np.float32,   'unsafe')
True
>>> np.can_cast(B.asnumpy().dtype, np.complex64, 'unsafe')
False
>>> B.asnumpy().astype(np.complex64)
TypeError: Cannot cast array data from dtype(bfloat16) to dtype('complex64') according to the rule 'unsafe'
```

So the float -> float path of the fold succeeds, the float -> complex path raises, and because the
Rondo `Cast` fold runs during static analysis the exception aborts compilation of the entire graph.
(The scalar case reports MindSpore's own wording, `Cannot cast data from bfloat16 to complex64.`)

## Impact

Any `ms.jit` function or `nn.Cell` in GRAPH mode that mixes a complex tensor with a bfloat16
**constant** cannot be compiled at all: bf16 precision configurations where a weight/scale is frozen
into the graph (`Tensor(np.asarray(...), ms.bfloat16)`), module-level bf16 constants, and complex
signal-processing / complex-valued network code.  The workaround is to route the constant through
float32 (`ops.cast(k, ms.float32)`), which is easy to miss because the identical expression works in
PYNATIVE mode and works in GRAPH mode with the bf16 value as an input, so the failure surfaces only
after moving the value into the graph (e.g. when freezing a configuration value).

## Suggested fix

Handle the bfloat16 source in `infer_value_for_Cast` by going through a numpy-friendly intermediate,
e.g.

```python
arr = x.asnumpy()
if arr.dtype.name == "bfloat16" and np.issubdtype(np_dst_type, np.complexfloating):
    arr = arr.astype(np.float32)
value = Tensor_(arr.astype(np_dst_type), dtype=dst_type)
```

or register the bfloat16 -> complex64/complex128 cast rule for MindSpore's numpy bfloat16 dtype, and
add a graph-mode test that promotes a captured bfloat16 constant against a complex input.

## Controls (all in `repro.py`, 33 checks)

* `complex64 + bfloat16` and `bfloat16 + complex64` as **graph inputs** compile and are correct, at
  both complex64 and complex128 — proof that the graph path and the kernel support the dtype mix.
* Captured `float16`/`float32`/`float64`/`int8`/`int32`/`int64`/`complex64` constants + complex64
  input compile and are correct; a captured bfloat16 constant + float32 input compiles and is correct.
* Eager results for every witness equal the numpy reference (`[2+0j, 4+0j]`, and `[2+0j, 3+0j]` for
  the bool scalar).
* `np.can_cast(bfloat16, float32)` is `True` while `np.can_cast(bfloat16, complex64)` is `False`, and
  `astype(np.float32)` succeeds where `astype(np.complex64)` raises — the exact root-cause split.
* The float32 workaround compiles and is correct in GRAPH mode.
* Two process-isolated replays of the certified reproducer give `RESULT: 0 failing check(s) out of 33`
  each; `verify_drafts.py --replays 2 --only MS-027` reports PASS 2/2.

## Novelty

Searched the MindSpore tracker and web for `Cannot cast array data from dtype(bfloat16)`,
`Cannot cast data from bfloat16 to complex64`, `infer_value_for_Cast bfloat16`, `bfloat16 complex64
add` and `bfloat16 complex64 graph mode`: the reports that surface concern different operators and
different modes (`mint.ones_like`/`mint.triu` complex-exclusion in PYNATIVE, `unique_consecutive`,
`cumsum` empty input), not constant folding of a bfloat16 `Cast` under GRAPH mode.  The nearest
tracker item is the 2022 Gitee report [I5YRO1 `[CT][MS][Cast] complex dtype has some
problem`](https://gitee.com/mindspore/mindspore/issues/I5YRO1), which is about `Cast` and complex
dtypes in general, is marked DONE/closed, and contains no bfloat16-constant or dtype-promotion
description; it does not cover this failure.  The campaign's
other MindSpore findings are unrelated operators (MS-013 unregistered `mint` kernels, MS-021 stale
buffers, MS-022 `ldexp` range, MS-023 `logaddexp` infinities, MS-024 `roll` kernel, MS-025 empty
`cumsum`, MS-026 `unique_consecutive` leading NaN).  This is the first finding in the corpus about
dtype promotion of a *constant* in GRAPH mode.

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.  Full dump in
`collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"bfloat16 + complex is an unusual dtype mix."**  It is supported by the framework: eager computes
  it and GRAPH computes it when bfloat16 is an input.  The bug is not that the mix is rejected, it is
  that the *same* computation is rejected only when the bf16 value is a constant — a distinction
  users cannot see in their source semantics.
* **"This is a numpy limitation, not a MindSpore bug."**  MindSpore registers the bfloat16 numpy dtype
  and owns the `infer_value_for_Cast` fold that relies on it; the framework already treats
  bfloat16 -> float32 as a valid numpy cast, so the complex target only needs the same one-line
  intermediate cast.  Either way the effect is a compile failure of valid user code.
* **"Use `ops.cast(k, ms.float32)`."**  That is the workaround, and the draft documents it, but
  requiring users to know which side of a promotion triggers a numpy cast is not a reasonable
  contract.
* **"Only constants are affected, so it is a corner case."**  Constants are exactly what graph
  freezing produces, and the failure is a hard error on the whole graph rather than a degraded value.
