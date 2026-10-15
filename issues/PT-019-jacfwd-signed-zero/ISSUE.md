# `torch.func` **forward mode** (`jvp` and `jacfwd`) loses the **sign of zero** where the exact derivative is `-0.0`, while reverse mode (`jacrev` / eager backward) is correct

**Component:** `torch.func` — forward-mode differentiation (`jvp`, `jacfwd`), PyTorch 2.12.1
**Severity:** wrong bit-level result (a `-0.0` derivative becomes `+0.0`); silent, and it changes
`1/f'`, `signbit`, `copysign` and comparisons against `-0.0`
**Status:** present in `torch 2.12.1+cu130` (CPU and CUDA)
**Found with:** Rondo `torch.func` transform-pair probe (`artifact/hunt-v2/tools_torch_func_combos.py`)

---

## Summary

For derivatives whose exact value is a signed zero, both forward-mode entry points
(`torch.func.jvp` and `torch.func.jacfwd`) return `+0.0` while `torch.func.jacrev` and eager
autograd return the correct `-0.0`.  Four independent cases, all stable across runs (CPU and CUDA):

```python
import torch
x = torch.tensor([-0.0], dtype=torch.float64)
f = lambda t: (t * t).sum()

torch.func.jacrev(lambda t: f(t).reshape(1))(x.clone())   # tensor([[-0.]])   correct (2x = -0.0)
torch.func.jacfwd(lambda t: f(t).reshape(1))(x.clone())   # tensor([[ 0.]])   <-- sign lost
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| function | point | exact derivative | `jacrev` | `jacfwd` |
|---|---|---|---|---|
| `x²` | `x = -0.0` | `2x = -0.0` | `-0.0` | **`+0.0`** |
| `log(x)` | `x = -inf` | `1/x = -0.0` | `-0.0` | **`+0.0`** |
| `sqrt(\|x\|)` | `x = -inf` | `sign(x)/(2√\|x\|) = -0.0` | `-0.0` | **`+0.0`** |
| `1/x` | `x = +inf` | `-1/x² = -0.0` | `-0.0` | **`+0.0`** |
| control `x³` | `x = -0.0` | `3x² = +0.0` | `+0.0` | `+0.0` |
| control `exp(x)` | `x = -inf` | `+0.0` | `+0.0` | `+0.0` |
| control `x²`, `log`, `exp`, `sin` | non-zero points | non-zero | agree | agree |

Both forward-mode paths are affected and reverse mode is not:

| case | `jvp(f, (x,), (1.0,))` | `jacfwd(f)` | `jacrev(f)` | eager `.backward()` |
|---|---|---|---|---|
| `x²` at `-0.0` | **`+0.0`** | **`+0.0`** | `-0.0` | `-0.0` |
| `log(x)` at `-inf` | **`+0.0`** | **`+0.0`** | `-0.0` | `-0.0` |
| `1/x` at `+inf` | **`+0.0`** | **`+0.0`** | `-0.0` | `-0.0` |

The behavior is identical on CPU and CUDA and reproduces in two independent processes.

## Root cause (pointer)

Forward-mode rules in `torch.func` evaluate the derivative as a product of the tangent and a local
partial derivative; several of them spell the result as `tangent * something` or `0 * x`, and a
product that multiplies an explicit `0` (or a zero tangent) normalises the sign to `+0.0`, whereas
the reverse-mode formulas keep the sign of the surviving factor.  Reverse mode is correct on all
four cases, so the fix belongs in the forward-mode derivative formulas for `mul`, `div`, `log` and
`sqrt`.  An open question for the report: with a **0-dim** input tensor the same comparison did not
show the divergence in an earlier campaign, so the trigger may require a 1-element 1-D tensor —
worth stating in the issue so a maintainer can reproduce exactly.

## Second-order scope: `torch.func.hessian` inherits it, and reverse-over-reverse is not exempt

`torch.func.hessian` is documented as "forward-over-reverse (`jacfwd(jacrev(func))`) … a good default
for good performance", so it inherits the forward-mode sign loss on the *second* derivative.
`hessian_sign.py` (33 checks, 2/2 replays, `RESULT: 0 failing check(s)`) evaluates the diagonal of
the Hessian through six strategies and compares the **sign of the zero** against the analytic second
derivative computed elementwise in float64 (`-2·tanh(x)·(1-tanh²x)`, `6x`):

| lane (`f = Σ tanh x`, exact signs at `+0.0, -0.0, +inf, -inf`) | signs |
|---|---|
| exact (analytic) | `-, +, -, +` |
| `torch.func.hessian` | `+, +, +, +` — **wrong at `+0.0` and `+inf`** |
| `torch.func.jacfwd(jacrev(f))` | `+, +, +, +` (identical to `hessian`, as documented) |
| `torch.func.jacfwd(grad(f))` | `+, +, +, +` |
| `torch.func.jacrev(jacrev(f))` | `+, +, -, +` — wrong at `+0.0` |
| `torch.func.jacrev(grad(f))` | `-, +, -, +` — correct |
| `torch.autograd.functional.hessian` | `-, +, -, +` — correct |

So a correct path exists in the same library, and the two public Hessian entry points disagree with
each other.  The qualifier to the title's "reverse mode is correct": reverse-*over*-reverse is not
exempt either.  For `f = Σ x³` at `x = -0.0` the exact `6x` is `-0.0`; `jacrev(jacrev(f))` returns
`+0.0` while `torch.func.hessian`, `jacfwd(jacrev)`, `jacrev(grad)` and
`torch.autograd.functional.hessian` all return the correct `-0.0`.

This is the same root cause as the first-order report (forward mode does not preserve the sign of a
zero derivative), so one fix should cover both; it is included here because `torch.func.hessian` is
the API most users call and its default strategy is the affected one.

## Impact

Forward-mode Jacobians are used for sensitivity analysis, implicit differentiation and
mixed-mode Hessians.  A sign flip on a zero derivative is individually small, but it is
*observable*: `1 / f'` flips between `+inf` and `-inf`, `signbit`/`copysign` disagree between
`jacfwd` and `jacrev` for the same function and point, and code that branches on the sign of a
zero derivative takes the other branch.  It is also silent — the values compare equal under `==`.

## Novelty assessment

* Tracker searches in `pytorch/pytorch` for `jacrev jacfwd sign`, `jacfwd signed zero`,
  `torch.func jacobian sign of zero` and `signed zero jacobian` return no matching report.
* **#189799** (open) is the known systemic *Inductor* signed-zero issue and does not involve
  `torch.func`; this divergence is between two eager `torch.func` modes and reproduces without any
  compiler.
* The discriminator that makes this a forward-mode defect rather than a signed-zero convention is
  that **both** forward-mode entry points (`jvp`, `jacfwd`) lose the sign while **both** reverse-mode
  paths (`jacrev`, eager `.backward()`) keep it, on the same function and point.

## Files

* `repro.py` — self-checking reproducer (jacrev/jacfwd/jvp comparison, controls for non-zero
  derivatives and for the `+0.0` cases, CPU and CUDA, `--scan`)
* `hessian_sign.py` — second-order scope: the Hessian diagonal through six strategies against the
  analytic signed-zero table (33 checks, `--replays 2`)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
