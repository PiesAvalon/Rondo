# `torch.maximum` / `torch.minimum` return the wrong sign of zero, and the sign flips at 16 elements

**Component:** PyTorch 2.12.1+cu130, CPU, eager — `torch.maximum`, `torch.minimum`, and user code
composed from them (clipping / custom ReLU idioms).
**Severity:** silent wrong sign of zero, which propagates to `±inf` through `1/x` (and through
`log`, `copysign`, `atan2(y, ...)`).  The same expression also returns a different sign depending on
how many elements the tensor has.
**Found with:** Rondo's vectorisation-width oracle (`artifact/hunt-v2/tools_vector_width_diff.py`),
which holds the value fixed and varies the tensor width.

---

## Summary

IEEE 754-2019 specifies the result of `maximum`/`minimum` over the signed zeros:
`maximum(+0,-0) = +0`, `maximum(-0,+0) = +0`, `minimum(+0,-0) = -0`, `minimum(-0,+0) = -0`.
PyTorch's CPU implementation returns the *first* operand on such a tie for tensors up to 12 elements
and the *second* operand from 16 elements onwards, so the sign of the result depends on the tensor
size and one of the two operand orders always contradicts the specification:

| expression | n=1 | n=4 | n=8 | n=12 | n=16 | n≥32 | IEEE 754-2019 |
|---|---|---|---|---|---|---|---|
| `maximum(+0, -0)` | `+0.0` | `+0.0` | `+0.0` | `+0.0` | **`-0.0`** | **`-0.0`** | `+0.0` |
| `maximum(-0, +0)` | **`-0.0`** | **`-0.0`** | **`-0.0`** | **`-0.0`** | `+0.0` | `+0.0` | `+0.0` |
| `minimum(+0, -0)` | **`+0.0`** | **`+0.0`** | **`+0.0`** | **`+0.0`** | `-0.0` | `-0.0` | `-0.0` |
| `minimum(-0, +0)` | `-0.0` | `-0.0` | `-0.0` | `-0.0` | **`+0.0`** | **`+0.0`** | `-0.0` |

(bold = contradicts the specification; measured with `torch.set_num_threads(1)`)

```python
import numpy as np, torch
for w in (1, 12, 16, 64):
    x = torch.tensor([0.0] * w); y = torch.tensor([-0.0] * w)
    m = torch.maximum(x, y)[0]
    print(w, repr(float(m)), "signbit", bool(np.signbit(float(m))))
# 1  0.0  signbit False
# 12 0.0  signbit False
# 16 -0.0 signbit True      <-- same expression, different answer
# 64 -0.0 signbit True
```

### The same defect exists in the float64 lane, with the loop boundary at 8

| expression | n=1 | n=4 | n=8 | n=16 | n>=64 | IEEE 754-2019 |
|---|---|---|---|---|---|---|
| `maximum(+0, -0)` | `+0.0` | `+0.0` | **`-0.0`** | **`-0.0`** | **`-0.0`** | `+0.0` |
| `maximum(-0, +0)` | **`-0.0`** | **`-0.0`** | `+0.0` | `+0.0` | `+0.0` | `+0.0` |
| `minimum(+0, -0)` | **`+0.0`** | **`+0.0`** | `-0.0` | `-0.0` | `-0.0` | `-0.0` |
| `minimum(-0, +0)` | `-0.0` | `-0.0` | **`+0.0`** | **`+0.0`** | **`+0.0`** | `-0.0` |

`torch.clamp(-0.0, 0, 6)` stays width independent in fp64 as well (it is a control in the
reproducer, which now asserts both dtypes).

## Why it matters

The sign of zero decides the sign of infinities downstream, so a user-written activation or clip
changes value at the width boundary:

```python
relu_like = lambda t: torch.minimum(torch.maximum(t, 0.0), 6.0)
x = torch.full((16,), -0.0)
1.0 / relu_like(x)[0]     # +inf     (vectorised path)
1.0 / relu_like(torch.full((8,), -0.0))[0]   # -inf     (scalar path)
```

`torch.clamp`/`torch.nn.functional.hardtanh` are width-independent in the same test (they take a
different bound-application path), so this is specifically about `maximum`/`minimum` and about the
composites users write with them - which is how most custom ReLU/clip code is written.  Any model
whose tensor size crosses 16 elements between builds (a batch-size change, a different sequence
length, a different device) can therefore change the sign of a zero, and with it the sign of a
reciprocal or a logarithm.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).` (each replay runs in its own process)

The reproducer asserts:

* the sign table above for all four operand orders and widths 1..128, including the exact boundary
  (widths up to 12 agree with each other, widths from 16 up agree with each other, and the two
  classes disagree);
* that for every operand order exactly one of the two classes contradicts IEEE 754-2019, i.e. the
  two implementations cannot both be right;
* the user-level consequence: `torch.minimum(torch.maximum(x, 0.0), 6.0)` on `x = -0.0` is `-0.0`
  at n<=12 and `+0.0` at n>=16, so `1.0 / composite` is `-inf` vs `+inf`;
* controls that must be width-independent: `maximum(1,-1)=1`, `minimum(1,-1)=-1`,
  `maximum(0.0, 0.0)=+0.0`, `minimum(-0.0, -0.0)=-0.0`, and `clamp(-0.0,0,6)` (measured to be
  width-independent, so it is not part of the bug).

## Impact bounded: the gradient is *not* affected

Because `maximum`/`minimum` are differentiable, a tie-break difference could have leaked into
autograd and silently routed gradients to a different operand.  It does not - measured for both ops
at every width in `{1,2,4,8,12,16,32,64}`:

| tie | d/da, d/db (every width, and under `torch.compile`) |
|---|---|
| `a == b` (any value, incl. `+0` vs `-0`) | `0.5, 0.5` |
| `a == b == nan`, or one operand NaN | `1, 1` |
| `a > b` | `1, 0` |
| `a < b` | `0, 1` |

So the autograd rule uses explicit equality masks and is width independent: the defect in this
report is confined to the *forward* sign of zero.  (That also means the fix does not have to touch
the backward kernels.)

## The same split exists in other element-wise kernels (separate patches)

While measuring this, the same eager scalar-loop vs vectorised-loop split showed up in two further
CPU kernels.  They are listed here because they are the same *mechanism*, but they are different
kernels and would need their own fixes - they are **not** claimed to be fixed by a
maximum/minimum patch, and they are not filed as separate reports:

| kernel | behaviour | boundary |
|---|---|---|
| `torch.nn.functional.gelu` | `gelu(x)` for `x < 0` (e.g. `-0.0`, `-10`, `-100`, `-1e4`, `-1e10`, `-1e20`): scalar path `-0.0`, vectorised path `+0.0` (fp64 keeps the negative sign) | flips at `n=2` |
| `torch.nn.functional.softshrink` | `softshrink(x)` for `x <= -0.5` and for `-0.0`: scalar path `-0.0`, vectorised path `+0.0` | flips at `n=16` |

`gelu` is the more visible of the two, but note that the vectorised GELU kernel is already the
subject of `pytorch#189804` ("Inductor vectorized GELU kernel produces tiny negative value ... where
eager produces ...", closed 2026-07-14), which is *later* than the commit this build was made from
(2026-06-09) - so that one is a pre-fix build rather than a new finding and is deliberately not
reported here.  `softshrink` has no matching report.

## Root cause

ATen's CPU element-wise max/min has a scalar loop and a `at::vec::Vectorized<float>` loop.  The
scalar loop uses `a > b ? a : b` (returning the *second* operand on a tie - which matches NumPy),
while the vectorised loop's `maximum`/`minimum` use the opposite comparison, returning the *first*
operand.  Neither applies IEEE 754-2019's rule for the signed zeros, where the sign of a zero result
is specified rather than left to the tie-break.  This is the same mechanism as the filed TF-003 and
TF-001 in this repository (one operation, two implementations, different answers), in a different
codebase.

## Novelty assessment

* Searches on `pytorch/pytorch` for `maximum negative zero`, `minimum negative zero`,
  `signed zero maximum`, `maximum signed zero` return no matching report.
* NumPy is width-stable here but also uses the "second operand on a tie" rule, so it too deviates
  from IEEE 754-2019 for `maximum(+0,-0)`.  That is why a fix should not simply copy NumPy's
  tie-break: the report is about the width-dependent disagreement, and the specification is the
  target.
* The equivalent TensorFlow defect is filed separately (TF-003) with a different packet width (8),
  because it is a different codebase (Eigen functors) and needs a different patch.

## Files

* `repro.py` - self-checking reproducer (sign table, IEEE verdict per class, composite/reciprocal
  consequence, controls, two process-isolated replays)
* `evidence.txt` - raw observations
* `collect_env.txt` - environment
