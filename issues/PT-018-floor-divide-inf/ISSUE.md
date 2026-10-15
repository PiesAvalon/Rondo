# `[inductor]` `floor_divide` ignores the infinite-divisor correction — compiled returns `-0` / `±inf` where eager, NumPy and Python return `-1` / `nan`

**Component:** `torch.compile` → Inductor, `aten.div` with `rounding_mode="floor"` /
`torch.floor_divide` (CPU and CUDA)
**Severity:** silent wrong result (wrong magnitude and wrong sign)
**Status:** present in `torch 2.12.1+cu130`; `aot_eager` and `eager` are correct
**Found with:** Rondo identity/edge-value differential hunt (`artifact/hunt-v2`)

---

## Summary

`torch.floor_divide(a, b)` follows the Python/NumPy floor-division contract: the quotient is
`floor(a / b)`, corrected so that `a == q*b + r` with `r` having the sign of `b`.  That correction
matters when the true quotient is a signed zero, e.g. `-0.5 // inf == -1`, and when the dividend is
infinite, e.g. `inf // 2.0` is `nan` (NumPy/Python).  Inductor's lowering computes the plain
`floor(a / b)` instead, so the compiled result is `-0.0` (i.e. `0`) and `±inf`:

```python
import torch, numpy as np
a, b = torch.tensor([-0.5]), torch.tensor([float("inf")])
torch.floor_divide(a, b)                                        # tensor([-1.])
torch.compile(torch.floor_divide, fullgraph=True)(a, b)         # tensor([-0.])   <-- wrong
-0.5 // float("inf"), np.floor_divide(np.float32(-0.5), np.float32("inf"))  # -1.0, -1.0
```

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).`

| `floor_divide(a, b)` | eager | compiled | NumPy / Python | comment |
|---|---|---|---|---|
| `(-0.5, inf)` | `-1` | **`-0`** | `-1` / `-1` | signed-zero quotient must floor to `-1` |
| `(0.5, -inf)` | `-1` | **`-0`** | `-1` / `-1` | same |
| `(2.0, -inf)` | `-1` | **`-0`** | `-1` / `-1` | same |
| `(-0.5, -inf)` | `0` | `0` | `0` / `0` | agrees |
| `(0.0, -inf)` | `-0` | `-0` | `-0` / `-0` | agrees |
| `(inf, 2.0)` | `nan` | **`inf`** | `nan` / `nan` | infinite dividend |
| `(-inf, 2.0)` | `nan` | **`-inf`** | `nan` / `nan` | infinite dividend |
| `(inf, inf)` | `nan` | `nan` | `nan` / `nan` | agrees |
| control `(7.0, 3.0)` | `2` | `2` | `2` / `2` | agrees |
| control `(1e38, 1e-38)` | `inf` | `inf` | `inf` / `inf` | agrees |
| control `(1e10, 1e-20)` | `1e30` | `1e30` | `1e30` / `1e30` | agrees |

The divergence reproduces for **float32 and float64**, on **CPU and CUDA**, for **n = 1 and n ≥ 2**
(so it is not a vectorization-only effect), while `backend="aot_eager"` and `backend="eager"` are
correct:

```text
fp32 cpu  n=4 fd(-0.5, inf)  eager=[-1.0,...]  inductor=[-0.0,...]  aot_eager=[-1.0,...]  eager=[-1.0,...]
fp64 cuda n=1 fd(inf, 2.0)   eager=[nan]       inductor=[inf]       aot_eager=[nan]       eager=[nan]
```

## Root cause

Inductor lowers `floor_divide` to the plain `floor(a / b)` (vectorized: `div_floor_floating_vec`,
which is `a - fmod(a, b)` based; scalar: `std::floor(a / b)`), losing the two special cases that
the ATen reference implements:

* when `a / b` is a signed zero with both operands non-zero, the floored quotient must be `-1`,
  not `-0` (this is what makes `-0.5 // inf == -1` and keeps `a == q*b + r` with `r` of the sign
  of `b`);
* when `a` is infinite (and `b` finite), the quotient is `nan`, not `±inf`.

The ATen `div_floor_floating_vec` already carries a workaround for the *infinite divisor* case
(`vec_t::blendv(a - mod, a, (basic_div.abs() == inf) & (a.abs() != inf))`, added by an earlier fix
for the `fmod` overflow), which is why `(-0.5, -inf)` and `(0.0, -inf)` agree, but it has no
equivalent for the signed-zero-quotient case or for an infinite dividend.

## Suggested fix

Make the Inductor lowering share the ATen reference semantics rather than re-deriving
`floor(a/b)`:

* on the scalar path, use the same expression ATen uses for `rounding_mode="floor"`
  (`std::floor` is not equivalent; the sign correction is needed), and
* on the vectorized path, extend the existing `basic_div.abs() == inf` blend with the
  `basic_div == 0 && a != 0 && sign(a) != sign(b)` case (quotient `-0` → `-1`) and with the
  `a.abs() == inf && b.abs() != inf` case (result `nan`).

A regression test can be written directly from the table above, on CPU and CUDA, for `n = 1` and
`n = 4`.

## Impact

`torch.div(..., rounding_mode="floor")` / `//` is used for index arithmetic, bucketing and
quantisation.  A silently wrong `-0` instead of `-1` shifts every bucket by one for values whose
quotient is a signed zero (a whole half-open interval is mapped one bucket lower), and `inf`
instead of `nan` turns a "undefined" result into a huge finite one.

## Novelty assessment

* Tracker searches for `floor_divide inf`, `floor_divide inductor wrong`,
  `div rounding_mode floor inf nan` and `floor_divide compile wrong result` return no matching
  report.
* The two neighbouring items were checked and are distinct:
  * **#193753 / PR #193763** (open) is about `torch.fmod` / `torch.remainder` returning `NaN` on
    the vectorized CPU path.  Its body explicitly leaves `div` with `rounding_mode="floor"` out of
    scope ("`div_floor` can do that because it only needs the quotient"), and neither the issue
    nor the PR mentions `floor_divide` — the divergence reported here is in the quotient path it
    considers fixed.
  * **#129536** (closed) added the infinite-divisor blend that is the workaround this report shows
    to be incomplete for the signed-zero-quotient and infinite-dividend cases.
* The divergence is between Inductor and *eager*, and eager agrees with NumPy and Python, so this
  is a compiler regression rather than a reference-semantics question.

## Files

* `repro.py` — self-checking reproducer (per-case eager/compiled/NumPy/Python table, dtype ×
  device × size matrix, backend controls)
* `evidence.txt` — raw observations
* `collect_env.txt` — environment
