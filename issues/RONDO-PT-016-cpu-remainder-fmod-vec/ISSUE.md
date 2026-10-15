> **UPDATE 2026-09-11 (novelty re-check): already tracked upstream.** `torch.fmod`/`torch.remainder` returning wrong values on the vectorized CPU path is issue **#193753** (open) and is being fixed by PR **#193763** (open).  The root cause is the same (`Vectorized<T>::fmod`/Sleef undefined for large `|a/b|`, and `CppVecOverrides.remainder = a - floordiv(a,b)*b`).  **Do not submit this draft as a new issue**; it is kept as evidence for that discussion (it adds the non-NaN symptoms `remainder(1e10, 1e-20) -> 0`, `remainder(1e38, 1e-38) -> -inf` and the infinite-divisor triggers `remainder(0.5, inf) -> nan`).

# `[inductor][cpu]` the **vectorized** `remainder` / `fmod` lowering is wrong for large `|a/b|` — compiled CPU returns `0` / `-inf` / `nan` where eager returns the exact value

**Component:** `torch.compile` → Inductor CPU codegen (`CppVecOverrides.remainder`, `codegen/cpp.py:1520`)
**Severity:** **silent wrong results** for finite inputs (wrong finite value, `-inf` or `nan`)
**Status:** present in `torch 2.12.1+cu130`; `aot_eager` / `eager` backends are correct
**Found with:** Rondo edge-pair differential oracle (`special_edge_fuzz.py --binary`)

---

## Summary

The scalar and vectorized lowerings of `remainder` disagree:

```python
# OpOverrides.remainder (scalar), codegen/common.py:970
r = mod(a, b)
return where((r != 0) & (signbit(r) != signbit(b)), r + b, r)

# CppVecOverrides.remainder (vectorized), codegen/cpp.py:1520
return a - floordiv(a, b) * b
```

The vector form computes the quotient first, so it loses the exact result (or overflows) when
`|a / b|` is large.  `fmod`'s vector form (`Vectorized<float>::fmod`, SLEEF) fails on the same
inputs.  Only the *vectorized* path is affected, so a one-element tensor is correct and any tensor
with **two or more elements** is wrong.

```python
import torch
x = torch.full((4,), 1e10)      # note: 4 elements, not 1
y = torch.full((4,), 1e-20)
torch.remainder(x, y)                                   # [2.641808081341232e-21, ...]  exact
torch.compile(lambda a, b: torch.remainder(a, b))(x, y) # [0.0, 0.0, 0.0, 0.0]           wrong
```

## Reproduction results

`python repro.py --replays 2` (in this directory): 26 checks, 0 failures, identical in two runs.

Triggers (all with `n = 4`):

| expression | eager | compiled |
|---|---|---|
| `remainder(1e10, 1e-20)` f32 | `2.64181e-21` | `0` |
| `remainder(65504, 6e-8)` f32 | `6.71535e-09` | `0` |
| `remainder(3.4e38, 1.4e-45)` f32 | `0` | `-inf` |
| `fmod(3.4e38, 1.4e-45)` f32 | `0` | `nan` |
| `remainder(1e38, 1e-38)` f32 | `9.63916e-39` | `-inf` |
| `fmod(1e38, 1e-38)` f32 | `9.63916e-39` | `nan` |
| `remainder(1e300, 1e-300)` f64 | `4.89155e-301` | `-inf` |
| `fmod(1e300, 1e-300)` f64 | `4.89155e-301` | `nan` |
| `remainder(1.0, inf)` f32 | `1.0` | `nan` |

Controls (all correct, which is what isolates the defect):

| control | result |
|---|---|
| the same inputs with **`n = 1`** (scalar lowering) | match — this is why a one-element repro looks fine |
| `remainder(5.0, 2.0)`, `fmod(5.0, 2.0)`, `remainder(1.0, 1e30)`, `remainder(1e38, 3.4e38)` | match |
| `floor_divide(3.4e38, 1e-45)`, `div(..., rounding_mode="floor")` | match (both `inf`, as eager) |
| `backend="aot_eager"`, `backend="eager"` | match |

`TORCH_LOGS=output_code` for the failing case (`evidence.txt`) shows exactly the vectorized formula:

```cpp
auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + x0, 4L);
auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + x0, 4L);
auto tmp2 = tmp0 - (div_floor_floating_vec(tmp0, tmp1)) * tmp1;
tmp2.store(out_ptr0 + x0, 4L);
```

## Root cause

`a - floordiv(a, b) * b` is only equivalent to `remainder` when the intermediate
`floordiv(a, b) * b` can be represented exactly:

* `a = 1e10, b = 1e-20`: `a/b ≈ 1e30`, `floor(a/b) * b ≈ 9.999999...e9` which rounds back to `a`
  in float32, so the difference underflows to `0`;
* `a = 1e38, b = 1e-38`: `a/b ≈ 1e76` overflows float32, `floor = inf`, `inf * b = inf`, so
  `a - inf = -inf`;
* `b = inf`: `a/inf = 0`, `0 * inf = nan`, so the result is `nan` (eager returns `a` for `a >= 0`).

The scalar lowering avoids all three by computing `mod` first and adjusting with the sign bits.

## Relationship to pytorch#194345

#194345 (`[inductor][cuda] torch.remainder / torch.fmod wrong results under torch.compile on CUDA
for large |a/b| ratios`, open) reports the CUDA/Triton variant of the same class of failure and
states: *"eager CUDA, eager CPU, and compiled CPU all give the correct result"*.  Its reproducer
uses **one-element tensors** (`torch.tensor([a])`), which take the CPU *scalar* lowering; with
`n >= 2` the CPU result is wrong as well, as shown above.  This is a **different lowering site**
(`CppVecOverrides.remainder` in the C++ codegen, not the Triton lowering), so a CUDA fix would not
address it, and the CPU case needs its own fix and regression test.

## Suggested fix

Make the vectorized lowering use the same numerically safe form as the scalar one — compute
`Vectorized::fmod` (or `mod`) first and add `b` where the sign differs — instead of
`a - floordiv(a, b) * b`; and correspondingly use a safe `fmod` for the vectorized `fmod`.  A
regression test needs a tensor with at least two elements: `torch.remainder(torch.full((4,), 1e10),
torch.full((4,), 1e-20))` must equal the eager result.

## Impact

`remainder`/`fmod` are used for angle wrapping, periodic encoding and normalisation where the
ratio can be huge (large values against small periods).  The failure is silent and returns a
*finite wrong value* in the common case (`0` instead of the true remainder), so it is easy to miss
in a test suite that only uses moderate ratios.

## Files

* `repro.py` — 9 triggers + 15 controls (scalar-path, moderate-ratio, other-division, backend)
* `evidence.txt` — generated C++ for the failing case, the n=1/n>=2 contrast and the #194345 note
* `collect_env.txt` — environment
