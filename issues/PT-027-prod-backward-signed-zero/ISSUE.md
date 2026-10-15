# `torch.prod`'s backward drops the sign of zero for inputs with more than one zero

**Component:** PyTorch 2.12.1+cu130, CPU.  Eager reverse mode, `torch.autograd.grad(x.prod(), x)` and
`Tensor.backward()` (the no-`dim` overload); `float32` and `float64`.
**Severity:** wrong gradient values.  `d prod / d x_i = prod_{j != i} x_j`, and IEEE-754 fixes the
sign of every zero product, so the gradient of a product with zero factors has well-defined `-0.0`
entries.  For an input with **more than one zero** eager PyTorch returns a tensor of **positive**
zeros instead.  Sign-sensitive consumers (`1 / grad`, `signbit`, `copysign`, division by a gradient,
masks built from gradient signs) see the opposite sign, and the same operator on the same input
returns different bits depending on whether a graph is built.
**Status:** present in `torch 2.12.1+cu130` (CPU) in both widths.  The eager result differs from the
exact partials in 5 of 12 sign/zero patterns; `create_graph=True`, the `prod(dim=...)` overload,
`torch.func.grad`, `torch.func.jacrev` and the same backward under `torch.compile` match the exact
partials in **all** 12 (`torch.func.jacfwd` loses the sign in 9 - that forward-mode defect is tracked
separately as this campaign's PT-019).
**Found with:** the wave-202 Rondo autodiff oracle (mode-pair and transform-pair differential testing
of `prod`'s gradient: eager backward vs `torch.func.jacrev`/`grad` vs the exact float64 product of
the other factors, plus a `create_graph` lane).  Certified reproducer: `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 235 checks).

---

## Summary

```python
import torch
x = torch.tensor([-0.0, -0.0], requires_grad=True)

torch.autograd.grad(x.prod(), x)[0]                      # [+0.0, +0.0]   <-- both signs lost
torch.autograd.grad(x.prod(), x, create_graph=True)[0]   # [-0.0, -0.0]   correct, same op + input
torch.autograd.grad(x.prod(0), x)[0]                     # [-0.0, -0.0]   correct
torch.func.grad(lambda t: t.prod())(x)                   # [-0.0, -0.0]   correct
torch.func.jacrev(lambda t: t.prod())(x)                 # [-0.0, -0.0]   correct
torch.compile(lambda t: torch.autograd.grad(t.prod(), t)[0])(x.clone())
                                                         # [-0.0, -0.0]   correct

# the exact partials:  d prod / d x_i = prod_{j != i} x_j, evaluated in float64
[-0.0, -0.0]
```

The gradient of `x.prod()` for `x = [1.0, -0.0, -0.0, 1.0]` (`Tensor.backward()`), with the exact
partials and every other lane for comparison:

| lane | gradient |
|---|---|
| exact `prod_{j != i} x_j` (fp64) | `[-0.0, -0.0, -0.0, +0.0]` |
| **eager `autograd.grad` (default)** | **`[+0.0, +0.0, +0.0, +0.0]`** |
| **`Tensor.backward()`** | **`[+0.0, +0.0, +0.0, +0.0]`** |
| `autograd.grad(..., create_graph=True)` | `[-0.0, -0.0, -0.0, +0.0]` |
| `autograd.grad(x.prod(0), x)` | `[-0.0, -0.0, -0.0, +0.0]` |
| `torch.func.grad` | `[-0.0, -0.0, -0.0, +0.0]` |
| `torch.func.jacrev` | `[-0.0, -0.0, -0.0, +0.0]` |
| `torch.compile` (same `autograd.grad`) | `[-0.0, -0.0, -0.0, +0.0]` |
| `torch.func.jacfwd` | `[+0.0, +0.0, +0.0, +0.0]` (forward mode; PT-019) |

Measured over twelve zero/sign patterns (`float32` shown; `float64` identical):

| input | exact partials | eager `autograd.grad` | `jacrev` / `func.grad` / `dim` / `create_graph` / compiled |
|---|---|---|---|
| `[-0.0, -0.0]` | `[-0.0, -0.0]` | `[+0.0, +0.0]` | exact |
| `[-0.0, 0.0]` | `[+0.0, -0.0]` | `[+0.0, +0.0]` | exact |
| `[-0.0, 0.0, 0.8]` | `[+0.0, -0.0, -0.0]` | `[+0.0, +0.0, +0.0]` | exact |
| `[-0.0, 0.0, 0.8, -1.3]` | `[-0.0, +0.0, +0.0, -0.0]` | `[+0.0, +0.0, +0.0, +0.0]` | exact |
| `[-0.0] * 4` | `[-0.0] * 4` | `[+0.0] * 4` | exact |
| `[-0.0, -0.0, -0.0]` | `[+0.0, +0.0, +0.0]` | `[+0.0, +0.0, +0.0]` (agree) | exact |
| `[-0.0, -1.0]` (one zero) | `[-1.0, -0.0]` | exact | exact |
| `[-0.0, 1.0, 1.0]` (one zero) | `[+1.0, -0.0, -0.0]` | exact | exact |
| no zeros (control) | `[2.08, -1.56, 3.9, -2.4]` | exact | exact |
| all `+0.0` (control) | `[+0.0] * 3` | exact | exact |

Every non-zero magnitude is correct in every lane and the two controls agree everywhere, so this is
purely the sign of zero.  The pattern is exactly structural: **the defect appears if and only if the
input contains more than one zero**, and in that case the eager gradient is literally a tensor of
`+0.0` (checked element-wise by the reproducer).

## Root cause

`prod_backward` (`torch/csrc/autograd/FunctionsManual.cpp`) has three branches:

```cpp
Tensor zero_idx = (input == 0).nonzero();
if (zero_idx.sym_numel() == 0) {
    return grad * (result / input).conj();
} else if (!at::GradMode::is_enabled() && zero_idx.sym_size(0) > 1) {
    return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);   // <-- loses the signs
} else {
    return prod_safe_zeros_backward(grad, input.contiguous().view(-1), 0).view_as(input);
}
```

* With **no** zeros the `result / input` formula is exact.
* With **one** zero, `prod_safe_zeros_backward` is used; it computes
  `cumprod(exclusive, normal) * cumprod(exclusive, reversed)`, which is exactly `prod_{j != i} x_j`
  and carries the signs - hence the one-zero rows above are correct.
* With **more than one zero** and grad mode disabled, the early-out returns `at::zeros_like(input)`.
  Backward passes that do not build a graph (`create_graph=False`, i.e. `torch.autograd.grad`'s
  default and `Tensor.backward()`) run with grad mode disabled, so this branch is taken.
  Mathematically every partial really is zero *in magnitude* when there are two or more zeros - the
  shortcut is a legitimate optimization numerically, it just discards the sign bits of that zero.

The `create_graph=True` row above is the proof: the branch condition flips, the *same* op on the
*same* input returns the sign-carrying product, and `prod(dim=...)`'s overload (which has no such
shortcut) is exact as well.

## Impact

* `1 / grad` at those positions is `+inf` with the default eager gradient and `-inf` with the exact
  derivative (the reproducer asserts both).  Anyone using the gradient of a product in a
  sign-sensitive way - dividing by it, `sign`/`copysign` masks, comparing against `-0.0`, log-|grad|
  chains, Jacobian analysis - gets the opposite sign.
* `torch.compile` and `torch.func` transform the *same* model differently from eager: the compiled
  backward and every functional lane are exact, so a model and its compiled/transformed copy see
  gradients of opposite sign at these points.
* Silent: the magnitudes are correct, so numerical gradient checks with any tolerance pass and second
  order is unaffected (`create_graph=True` is exact).

## Suggested fix

Do not return `at::zeros_like` for a signed floating type in the `zero_idx.sym_size(0) > 1` branch:
either always use `prod_safe_zeros_backward` (it is already sign-exact and only marginally more
expensive), or, if the shortcut must stay, return zeros carrying the sign of the product of the other
factors, e.g. multiply a positive-zero mask by `sign(prod_safe_zeros_backward(...))`.  A regression
test is easy and needs no tolerance:

```python
x = torch.tensor([-0.0, -0.0], requires_grad=True)
assert torch.signbit(torch.autograd.grad(x.prod(), x)[0]).all()   # currently fails
```

## Controls (all in `repro.py`, 235 checks, 0 failing on each of 2 replays)

* Two controls agree in every lane: an input with no zeros, and an all-`+0.0` input (where `zeros_like`
  happens to be right), so only *negative* zero signs differ.
* Non-zero magnitudes are checked to be identical between the eager gradient and the exact partials
  in every case; where lanes differ, the wrong entries are checked to be `+0.0` in positions whose
  exact value is `-0.0`.
* The exact reference is the float64 product of the other factors, cross-checked by five independent
  lanes of the framework that all agree with it (`create_graph=True`, `prod(dim=0)`,
  `torch.func.grad`, `torch.func.jacrev`, `torch.compile`).
* The structural signature (eager gradient == all `+0.0` iff more than one zero) is asserted
  per-case, so the claim is tied to the mechanism rather than to one example.
* `float32` and `float64` are both covered, and `Tensor.backward()` is checked in addition to
  `torch.autograd.grad`.
* `jacfwd`'s sign loss is observed but deliberately left to PT-019, which is the forward-mode report;
  this draft's novelty claim is the eager shortcut.

## Novelty

Searched the PyTorch tracker for `prod backward signed zero`, `prod gradient -0.0`, `prod -0.0`,
`prod_backward negative zero`, `zeros_like prod grad sign` and `torch.prod grad sign`.  The adjacent
reports are [#995](https://github.com/pytorch/pytorch/issues/995) (the backward of `-prod(x)` missing
the sign, 2017), the "slow but safe formula" change
[#81617](https://github.com/pytorch/pytorch/commit/84c8a9f88e15b6d94407e8d1839db1ea8eb6ac6a), the
double-backward fix for 2+ zeros (`cf08f8f`), the `cumprod` zero-element crash fix (#32070) and the
macOS `masked.cumprod` NaN report (#109166): none of them is about the **sign of zero** in the
gradient of `prod`, and none observes that `create_graph` toggles the result.  This campaign's other
signed-zero drafts cover different operators and paths - PT-012/PT-020 `min`/`max`/`clamp` width,
PT-019 `jacfwd`/`jvp`, PT-021 gelu/softshrink layout, PT-022 the compiled unary flush, PT-024 complex
addition, PT-025 float8 `abs` - and only PT-019 touches `prod`, from the forward-mode side.

## Environment

PyTorch 2.12.1+cu130 (CPU execution; Inductor codegen for the compiled lane), NumPy 2.5.3, Python
3.12.13, WSL2 (Linux 6.6.87.2) x86_64.  Full dump in `collect_env.txt`; transcripts in
`evidence.txt`.

---

## Anticipated objections and responses

* **"The gradient really is zero, so this is cosmetic."**  The magnitude is zero but the value is not
  signless: IEEE-754 multiplication fixes it, `1 / grad` returns the opposite infinity, `signbit`
  flips, and PyTorch itself returns the exact signed zeros through five other lanes of the same
  operator (`create_graph=True`, `prod(dim=...)`, `torch.func.grad`, `jacrev`, Inductor).  The
  framework is internally inconsistent, and eager is the outlier.
* **"`zeros_like` is the intended fast path for a zero gradient."**  Then the intent is
  magnitude-based; the report is that the fast path must not be allowed to change the *sign* of a
  floating zero, which is observable.  The fix is cheap (the safe formula is already implemented and
  used in the adjacent branch).
* **"Numerical gradient checks pass."**  Yes, and the reproducer says so explicitly: the magnitudes
  are identical, only signs of zeros differ.  The claim is deliberately narrow.
* **"Overlaps your own PT-019."**  PT-019 is `torch.func` forward mode (`jvp`/`jacfwd`) on other
  functors; the eager reverse-mode shortcut is a different code path (`prod_backward`'s `zeros_like`
  early-out), reproducible without `torch.func` at all, and PT-027 keeps only a cross-reference to
  the forward-mode case.
