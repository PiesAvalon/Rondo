# `[aten]` `_weight_norm_interface` never validates the size of `g` — out-of-bounds read, garbage output and heap corruption instead of a shape error

**Component:** ATen C++ kernel `aten::_weight_norm_interface` (used by `torch._weight_norm` and by
`torch.nn.utils.weight_norm`'s forward pre-hook)
**Severity:** memory safety — out-of-bounds read; glibc reports heap corruption and aborts
**Status:** present in `torch 2.12.1+cu130` (CPU); found while investigating `RONDO-PT-013`
**Found with:** Rondo reference oracle on the weight-norm utilities (`weight_norm`, `parametrize`)

---

## Summary

`_weight_norm_interface(v, g, dim)` assumes `g` has exactly the size implied by
`(v.shape, dim)`: it computes a `norm` tensor of `g.numel()` elements and then produces an output
of `v`'s shape. There is no shape check anywhere on that path:

```python
import torch
v = torch.ones(4)
g = torch.full((3,), 2.0)          # one element short
out, norm = torch._weight_norm_interface(v, g, 0)
# no error.  out.shape == (4,), norm.shape == (3,)
# out[-1] is whatever follows `g` in memory
```

The trailing element of `out` is uninitialised memory: it differs between processes
(`4.38e-41` vs `3.47e-41` in two runs of the reproducer), and with a different heap layout the
process aborts with `free(): invalid next size (fast)`. An *oversize* `g` is accepted too — the
result is then computed from the first `v.numel()` entries while `norm` keeps `g`'s size.

## Reproduction results

`python repro.py` (in this directory): 5 checks, 0 failures.

| case | observed |
|---|---|
| `_weight_norm_interface(v(4), g(3), 0)` | no error; `out` (4,), `norm` (3,), `out[-1]` = uninitialised memory |
| `_weight_norm_interface(v(4), g(5), 0)` | no error; `out` (4,), `norm` (5,) |
| `_weight_norm(v(4), g(3), 0)` | no error; same as above |
| control: `_weight_norm(v(4), g(4), 0)` | matches `v * (g / norm_except_dim(v, 2, 0))` |
| control: out-of-range `dim` | clean `IndexError` |

Two further processes that called the kernel with a too-small `g` aborted at interpreter exit with
glibc's `free(): invalid next size (fast)` after printing the result, i.e. the out-of-bounds access
had already corrupted the allocator's metadata.

## Reachability

* `torch._weight_norm` / `torch._weight_norm_interface` are the fused implementation that
  third-party code calls directly, and they are the op behind
  `torch.nn.utils.parametrizations._WeightNorm.forward`.
* `torch.nn.utils.weight_norm` installs a forward pre-hook that calls
  `torch._weight_norm(module.<name>_v, module.<name>_g, dim)` on every forward. Replacing
  `<name>_v` with a parameter of a different size (a legal `nn.Parameter` assignment) sends a
  mismatched pair into the kernel; the out-of-bounds read happens before anything downstream can
  raise a shape error.

The size of `g` is fully determined by `(v.shape, dim)` (`torch.norm_except_dim(v, 2, dim)`),
so the check is cheap and exact.

## Root cause

`aten/src/ATen/native/WeightNorm.cpp` iterates the `g` tensor to derive the norm without checking
`g.numel()` against the expected reduced size, and the output tensor is allocated with `v`'s shape.
When `g` is smaller, the read walks past the end of the tensor's storage.

## Suggested fix

Validate the `g` size at the top of the kernel and raise a normal shape error, e.g.

```cpp
auto expected = norm_except_dim(v, 2, dim).numel();
TORCH_CHECK(g.numel() == expected,
            "_weight_norm_interface: expected g with ", expected,
            " elements for v", v.sizes(), " and dim=", dim, ", got ", g.numel());
```

A regression test is two lines: `torch._weight_norm(torch.ones(4), torch.ones(3), 0)` must raise,
and the same with a correctly sized `g` must match `v * (g / norm_except_dim(v, 2, 0))`.

## Related upstream work (novelty assessment)

* `RONDO-PT-013` (own finding) is a *wrong result* in the **decomposition** of the same op; this
  report is about the **ATen kernel** accepting a malformed `g`. They were found together but need
  separate fixes.
* #112086 (`[export] aten._weight_norm_interface doesn't have decomposition`, closed) and #114577
  (`torch.norm_except_dim() ... symbolic sizes`, closed) are unrelated.
* Tracker searches for `_weight_norm_interface shape check`, `weight_norm g size crash`,
  `_weight_norm heap corruption` return no matching report.

## Impact

A malformed `g` produces silently wrong values and an out-of-bounds read; depending on the
allocator state the process dies with heap corruption. Callers reach the kernel through the
public weight-norm utilities, so the failure mode is "silent garbage or allocator abort" where a
one-line shape check would produce a clean, actionable error.

## Files

* `repro.py` — 2 controls + 3 no-validation checks + a per-allocation evidence dump
* `collect_env.txt` — environment
