# `[inductor]` `maximum` / `minimum` / `fmax` / `fmin` / `clamp` return the wrong **sign of zero**

**Component:** `torch.compile` → Inductor codegen (`backend="inductor"` only)
**Severity:** wrong bit-level result; observable downstream as `1/x` returning `-inf` instead of `inf`
**Status:** present in `torch 2.12.1+cu130`, CPU and CUDA; `aot_eager` and `eager` backends are correct
**Found with:** Rondo reference oracle over signed-zero / NaN semantics (`hunt.py` triage vocabulary)

---

## Summary

When the two operands of an element-wise min/max compare equal but have different zero signs,
eager returns one specific operand while Inductor's compiled kernel returns the other, so the
**sign bit** of the result differs. The difference is not cosmetic: it propagates through
division, `signbit`, `copysign` and `atan2`.

```python
import torch
a = torch.tensor([0.0, -0.0])
b = torch.tensor([-0.0, 0.0])

torch.maximum(a, b)                          # tensor([ 0.0000, -0.0000])
torch.compile(lambda x, y: torch.maximum(x, y))(a, b)
                                             # tensor([-0.0000,  0.0000])   <-- sign flipped

1.0 / torch.maximum(a, b)                    # eager:    tensor([ inf, -inf])
1.0 / torch.compile(lambda x, y: torch.maximum(x, y))(a, b)
                                             # compiled: tensor([-inf,  inf])
```

## Reproduction results

`python repro.py --device cpu` and `--device cuda` (in this directory): 15 checks, 0 failures on
both devices.

Triggers on CPU (all 10 differ from eager in the sign of zero):

| expression | eager | compiled |
|---|---|---|
| `maximum(a, b)` | `[0.0, -0.0]` | `[-0.0, 0.0]` |
| `minimum(a, b)` | `[0.0, -0.0]` | `[-0.0, 0.0]` |
| `fmax(a, b)` | `[0.0, -0.0]` | `[-0.0, 0.0]` |
| `fmin(a, b)` | `[0.0, -0.0]` | `[-0.0, 0.0]` |
| `torch.max(a, b)` | `[0.0, -0.0]` | `[-0.0, 0.0]` |
| `torch.min(a, b)` | `[0.0, -0.0]` | `[-0.0, 0.0]` |
| `clamp_min(a, 0.0)` | `[0.0, -0.0]` | `[0.0, 0.0]` (sign lost) |
| `clamp_max(a, 0.0)` | `[0.0, -0.0]` | `[0.0, 0.0]` (sign lost) |
| `1.0 / maximum(a, b)` | `[inf, -inf]` | `[-inf, inf]` |
| `1.0 / clamp_min(a, 0.0)` | `[inf, -inf]` | `[inf, inf]` |

On CUDA every case except `clamp_min` also differs (CUDA eager itself returns `+0.0` for
`maximum`, but Inductor still returns `-0.0` for the first element, so the compiled result again
disagrees with eager on the same device).

Controls (correct, which is why this is a codegen issue and not the reference):

| case | result |
|---|---|
| `maximum(1.0, 1.0)` (equal, non-zero) | identical |
| `maximum(a, b) + 0.0` (sign normalised by the add) | identical |
| `copysign(maximum(a, b), -1.0)` (sign overwritten) | identical |
| `torch.compile(..., backend="aot_eager")` | identical to eager |
| `torch.compile(..., backend="eager")` | identical to eager |

The last two rows show that the compiled-but-eager-op path is fine, so the defect is in the
Inductor kernel that replaces the op.

## Root cause

Eager's element-wise min/max returns the *first* operand when the comparison is not strictly
true (`a < b ? b : a` style, giving `maximum(0.0, -0.0) == 0.0`). Inductor lowers these ops to
`tl.maximum` / `fmax`-style primitives, whose result for equal-but-signed-zero operands is the
other operand (and `clamp_min`/`clamp_max` are lowered to `maximum`/`minimum`, which drops the
sign of the negative zero). The compiled kernel therefore disagrees with both eager CPU and
eager CUDA on the sign bit.

## Suggested fix

Lower `maximum`/`minimum` (and the two-argument forms of `max`/`min`, and `clamp_min`/
`clamp_max`) so that the `+0.0`/`-0.0` case matches eager, e.g. by comparing with `==` and
selecting the first operand when the values are equal (`where(a == b, a, max_prim)`), or by
using the same primitive eager uses. A regression test is two lines: compare
`torch.signbit(torch.maximum(a, b))` for `a = [0.0, -0.0]`, `b = [-0.0, 0.0]` between eager and
`torch.compile(...)`.

## Related upstream work (novelty assessment)

* Tracker searches for `maximum signed zero torch.compile`, `clamp_min negative zero`,
  `fmax signed zero inductor`, `torch.maximum -0.0 wrong` and `sign of zero inductor codegen`
  return no matching report.
* This is a different failure mode from the other Rondo findings: it is not a guard and not a
  dtype/calling-convention issue, it is a value-level codegen difference in Inductor.

## Impact

`1 / max(x, eps)`-style normalisation, `copysign`/`signbit` logic, and any downstream division by
a min/max result can flip `inf`/`-inf` (or `NaN` in later arithmetic) when the operand is exactly
`-0.0` — for example a tensor produced by `-x` for `x == 0.0`, or by `x * -1.0`. The result is
silent and deterministic: no error, no warning, no graph break.

## Files

* `repro.py` — 10 triggers (device-aware expectations) + 3 controls + 2 backend controls
* `collect_env.txt` — environment

## Anticipated objections and responses

* **"Sign of zero is a bit-level detail, not worth fixing."**  It is silent and it propagates:
  `1 / maximum([0.0, -0.0], [-0.0, 0.0])` returns `[-inf, inf]` instead of `[inf, -inf]`, i.e. the
  sign flips the infinity.  The systemic Inductor signed-zero issue (**#189799**, open) shows
  maintainers treat this class as a defect, and the draft isolates the min/max/clamp lowering.
* **"This is a precision difference."**  It is not: the values are exactly equal under `==`, only
  the sign bit differs, and the repro reports the sign bits explicitly.
* **"Which side is correct?"**  IEEE-754 `maximum(+0, -0)` is `+0`; eager returns `+0` and the
  compiled kernel returns `-0`, so the compiled side is the outlier (and `aot_eager` agrees with
  eager in the repro).
