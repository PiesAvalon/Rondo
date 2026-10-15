# `mindspore.jvp` / `ops.jvp` rejects a single-element tuple or list of inputs with the internal error `Type Join Failed: Abstract type AbstractTuple cannot join with AbstractTensor`

**Component:** MindSpore 2.9.0 CPU, `mindspore.jvp` and `mindspore.ops.jvp`, PYNATIVE_MODE and
GRAPH_MODE, float32 and float64, plain function and `nn.Cell`
**Severity:** the documented input form for a one-argument function cannot be used at all.  The call
fails with an internal graph-compiler error instead of a result or a user-facing `TypeError`; the same
computation with an unwrapped tensor works and returns the correct derivative, and the docstring's
two-input tuple form works.
**Status:** present in MindSpore 2.9.0 (CPU)
**Found with:** a forward-mode-vs-reverse-mode consistency sweep of the MindSpore autodiff lane (the
lane that produced MS-017); the standalone container probe ships next to this file.

---

## Summary

`jvp(fn, inputs, v)` documents `inputs` as `Union[Tensor, tuple[Tensor], list[Tensor]]`.  For a
function of one argument the natural spelling - and the one MindSpore's own example uses for two
arguments - is a one-element tuple.  MindSpore rejects every container form of length 1:

| call (PYNATIVE, fp32) | result |
|---|---|
| `jvp(f_cube, x, v)` (unwrapped tensor) | works, `jvp = 6.75` (`3x²` at `x = 1.5`) |
| `jvp(f_cube, (x,), (v,))` | `TypeError: Type Join Failed: Abstract type AbstractTuple cannot join with AbstractTensor` |
| `jvp(f_cube, [x], [v])` | same error |
| `ops.jvp(f_cube, (x,), (v,))` | same error |
| `jvp(f_cube, (x64,), (v64,))` (float64) | same error |
| `jvp(f_aux, (x,), (v,), has_aux=True)` | same error |
| `jvp(CubeNet(), (x,), (v,))` (`nn.Cell`) | same error |
| `jvp(f_two, (x, y), (v, v))` (the docstring example) | works: `[4, 13, 28, 49]` |

GRAPH_MODE behaves identically.  The error is raised by the graph compiler, not by argument
validation: the message contains an abstract-value diff and the internal node name
(`@grad_grad_single`, comparing `AbstractTuple{element[0]: AbstractTensor...}` with
`AbstractTensor(...)`), so the caller has no way to tell that the input *format* was the problem:

```text
Type Join Failed: Abstract type AbstractTuple cannot join with AbstractTensor.

----------------------------------------------------
This:
AbstractTuple{ element[0]: AbstractTensor(shape: (1), ...) }
, other: AbstractTensor(shape: (1), ...).
Please check the node: @grad_grad_single_16:CNode_17{...}
```

That the container is the only problem is shown by the controls: unwrapping the same tensor returns
the correct `6.75`, the docstring's two-input tuple example still produces the documented
`[[4, 13], [28, 49]]`, and `ms.grad` on the same function and shape works in fp32 and fp64.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(21 checks per replay, 2 fresh processes.)

```python
import numpy as np, mindspore as ms
from mindspore import jvp, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

def f(a):
    return a ** 3

x = Tensor(np.asarray([1.5], np.float32))
v = Tensor(np.asarray([1.0], np.float32))

jvp(f, x, v)          # works: (3.375, 6.75)
jvp(f, (x,), (v,))    # TypeError: Type Join Failed ... AbstractTuple cannot join with AbstractTensor
```

What the reproducer pins:

| dimension | result |
|---|---|
| container forms | `(x,)` and `[x]` both raise the internal type-join `TypeError` in PYNATIVE and GRAPH mode, fp32 and fp64 |
| entry points | `mindspore.jvp` and `mindspore.ops.jvp`, with a plain function and with an `nn.Cell`, with `has_aux=True` and `False` |
| message | the message starts with `Type Join Failed` (the compiler's error, not a documented input-format `TypeError`) |
| controls | the unwrapped form returns `(3.375, 6.75)`; the documented two-input tuple example returns `[[4, 13], [28, 49]]`; `ms.grad` on the same function returns `6.75` in fp32 and fp64 |

## Root cause (pointer)

The length-1 sequence is not unwrapped before tracing: `jvp` keeps the container as an `AbstractTuple`
while the single-parameter function expects an `AbstractTensor`, and the graph compiler's type join
fails inside `@grad_grad_single`.  JAX spells the single-input case exactly this way
(`jax.jvp(f, (x,), (v,))`), and MindSpore's own example uses tuples for the multi-input case, so a
one-argument function written consistently fails.  A fix is to normalise a length-1 `tuple`/`list` to
its single element before building the forward-mode graph (and to report a user-facing error for
genuine arity mismatches); a regression test is `jvp(f, (x,), (v,))` equal to `jvp(f, x, v)` for a
one-argument `f`, in both modes.

## Novelty

* `gh search issues --repo mindspore-ai/mindspore` for `jvp`, `ops.jvp`, `jacobian vector product`,
  `forward mode automatic differentiation` and the error text `Type Join Failed AbstractTuple`
  returns **nothing**.
* A web search surfaces the `mindspore.jvp` API pages
  ([r2.7.2 docs](https://www.mindspore.cn/docs/en/r2.7.2/api_python/mindspore/mindspore.jvp.html)),
  the [Gitee `mindspore.jvp.rst` source](https://gitee.com/mindspore/mindspore/blob/master/docs/api/api_python/mindspore/mindspore.jvp.rst)
  and a generic Gitee report about `Type Join Failed` CRITICAL logs not raising (I77HVV) - none is
  about the container form of `jvp` inputs.
* The campaign's other MindSpore drafts are about operator kernels or `mint` registration; this is
  the first one about the public autodiff API's input handling, and it is distinct from MS-017 (a
  wrong `lgamma` gradient value).
