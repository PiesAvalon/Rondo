# `[inductor]` the `aten._weight_norm_interface` decomposition normalises over **all** dims for 1-D inputs — `weight_norm(..., name="bias")` silently returns wrong values and `inf`/`nan` gradients under `torch.compile`

**Component:** `torch.compile` → Inductor, via the decomposition registered in
`torch/_decomp/decompositions.py:5380`
**Severity:** **silent wrong results + non-finite gradients**
**Status:** present in `torch 2.12.1+cu130`; `backend="aot_eager"` and `backend="eager"` match eager
**Found with:** Rondo reference oracle over the reparameterisation utilities (`weight_norm`, `parametrize`, `spectral_norm`, `checkpoint`)

---

## Summary

`torch/_decomp/decompositions.py` replaces the ATen kernel `aten._weight_norm_interface` with

```python
@register_decomposition(torch.ops.aten._weight_norm_interface)
def _weight_norm_interface(v, g, dim=0):
    keep_dim = tuple(i for i in range(len(v.shape)) if i != dim)
    norm_dtype = torch.float if g.dtype == torch.bfloat16 else None
    norm = v.norm(2, keep_dim, keepdim=True, dtype=norm_dtype)
    return v * (g / norm.to(g.dtype)), norm
```

For a **1-D** `v` with `dim=0`, `keep_dim` is the **empty tuple**, and
`Tensor.norm(2, dim=(), keepdim=True)` reduces over **all** dims instead of over none. Eager's
ATen kernel treats "all dims except 0" of a 1-D tensor as the element-wise absolute value
(`torch.norm_except_dim(v, 2, 0) == v.abs()`), so the two disagree:

```
v = [ 0.14473324, 0.12190525, 0.01815160]   # (the magnitude vector g equals |v| at init)
eager    torch._weight_norm(v, g, 0)  ->  [ 0.14473324, 0.12190525, 0.01815160]   (v itself)
compiled torch._weight_norm(v, g, 0)  ->  [ 0.11019301, 0.07817399, 0.00173319]   (= v*g/||v||)
```

`v * (g / ||v||)` is exactly the wrong formula the decomposition computes, which is what makes
the root cause unambiguous.

## Reproduction results

`python repro.py --replays 2` (in this directory): 20 checks, 0 failures, identical in two
separate processes.

Triggers (all diverge; `float64`, `float32` and `bfloat16` all affected):

| case | eager | compiled |
|---|---|---|
| `torch._weight_norm(v_1d, g_1d, 0)` | `[0.1447, 0.1219, 0.0182]` | `[0.1102, 0.0782, 0.0017]` |
| `torch._weight_norm_interface(v_1d, g_1d, 0)[0]` | same | same |
| `torch._weight_norm(v_1d.float(), g_1d.float(), 0)` | `[0.1447, 0.1219, 0.0182]` | `[0.1102, 0.0782, 0.0017]` |
| `torch._weight_norm(v_1d.bfloat16(), g_1d.bfloat16(), 0)` | `[0.1445, 0.1221, 0.0182]` | `[0.1104, 0.0781, 0.0017]` |
| compiled value `== v * g / ||v||` (bit-exact, confirms the formula) | — | yes |
| `nn.utils.parametrizations.weight_norm(nn.Linear(4, 3), name="bias")(x)` | — | max diff `0.0437` |
| gradients of that module | finite | **`inf` / `nan` in both parametrisation tensors** |

Controls (all correct — this is what isolates the defect):

| case | result |
|---|---|
| `torch._weight_norm(v_1d, g_1d, -1)` | matches (`dim=-1` path is unaffected) |
| `torch.norm_except_dim(v_1d, 2, 0)` compiled | matches |
| the explicit `v * (g / torch.norm_except_dim(v, 2, 0))` compiled | matches |
| 2-D `v`, `dim=0` and `dim=1` | match |
| `backend="aot_eager"`, `backend="eager"` | match |

The explicit-expression control is the important one: it shows the *components* compile
correctly and only the decomposition's dim arithmetic is wrong.

## Root cause

`keep_dim` must be the dims **other than** `dim`. When `v.dim() == 1` and `dim == 0` that set is
empty, and `Tensor.norm` interprets an empty `dim` as "reduce everything" rather than "reduce
nothing". ATen's own kernel (`aten/src/ATen/native/WeightNorm.cpp`) computes this case as
`norm_except_dim`, which yields `|v|` element-wise for 1-D input. The decomposition therefore does
not reproduce the op it replaces.

## Suggested fix

Handle the empty-`keep_dim` case explicitly in the decomposition, e.g.

```python
if len(keep_dim) == 0:
    norm = v.abs()
else:
    norm = v.norm(2, keep_dim, keepdim=True, dtype=norm_dtype)
```

(or reuse `torch.norm_except_dim`'s semantics). A regression test is three lines: compare
`torch._weight_norm(v, g, 0)` with `torch.compile(lambda: torch._weight_norm(v, g, 0))()` for a
1-D `v`, and assert `torch.isfinite(grad).all()` for a `weight_norm(..., name="bias")` module.

## Related upstream work (novelty assessment)

* #112086 (`[export] aten._weight_norm_interface doesn't have decomposition`, closed) is why the
  decomposition exists; it says nothing about its correctness.
* #114577 (`torch.norm_except_dim() operator: Cannot call sizes() on tensor with symbolic
  sizes/strides`, closed) is a symbolic-shape crash in a different code path.
* Tracker searches for `weight_norm torch.compile wrong result`, `norm_except_dim torch.compile`,
  `_weight_norm_interface decomposition` and `weight_norm bias compile mismatch` return no
  report of this wrong-result/`nan`-gradient behaviour.

## Impact

`torch.nn.utils.weight_norm` and `torch.nn.utils.parametrizations.weight_norm` are the standard
weight-normalisation utilities. Applying them to a **1-D** parameter (`name="bias"`, or any 1-D
weight) and compiling the model silently changes the forward result and produces `inf`/`nan`
gradients during training — the worst kind of failure for a training loop. `torch.compile`'s
contract is to preserve eager semantics, and here the compiled path is not even self-consistent:
the explicit equivalent expression compiles correctly.

## Files

* `repro.py` — 4 triggers + formula check + 3 expression controls + module forward/gradient checks
* `collect_env.txt` — environment
