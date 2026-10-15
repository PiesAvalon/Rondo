# `F.gelu` and `F.softshrink` return different signs of zero depending on memory layout *and* tensor width

**Component:** PyTorch 2.12.1+cu130, CPU, eager — `torch.nn.functional.gelu` and
`torch.nn.functional.softshrink` (no compiler involved).
**Severity:** silent wrong sign of zero in a widely used activation, which propagates to `±inf`
through `1/x` and changes `signbit`/`copysign` results.  Which sign you get depends on whether the
input tensor is contiguous and on how many elements it has.
**Found with:** Rondo's class-only differentials — first the width oracle
(`tools_torch_special_width.py`), then a new layout oracle (`tools_layout_diff.py`).

---

## Summary

In these two activation kernels the vectorised contiguous loop and the scalar/strided loop disagree
on the sign of a zero result.  An fp64 computation arbitrates, because it takes neither fp32 path,
and it shows that the *vectorised contiguous* path is the wrong one for `gelu` while the
*scalar/strided* path is the wrong one for `softshrink`:

| expression | contiguous `n=64` | strided `n=64` (`base[::2]`) | scalar `n=1` | float64 contiguous | float64 strided |
|---|---|---|---|---|---|
| `gelu(-0.0)` | `+0.0` | `-0.0` | `-0.0` | `-0.0` | `-0.0` |
| `gelu(-10.0)` | `+0.0` | `-0.0` | `-0.0` | `-0.0` | `-0.0` |
| `gelu(-1e38)` | `+0.0` | `-0.0` | `-0.0` | `-0.0` | `-0.0` |
| `softshrink(-0.0)` | `+0.0` | `-0.0` | `-0.0` | `+0.0` | `-0.0` |
| `softshrink(-0.5)` | `+0.0` | `-0.0` | `-0.0` | `+0.0` | `-0.0` |
| `softshrink(-1e-4)` | `+0.0` | `-0.0` | `-0.0` | `+0.0` | `-0.0` |

Arbitration differs between the two ops, and the fp64 column is what settles each one:

* **`gelu`**: fp64 returns `-0.0` in *every* layout, and the fp32 strided and scalar paths agree with
  it.  The fp32 **contiguous** path (`+0.0`) is the wrong one.
* **`softshrink`**: fp64 shows the *same* layout split (`+0.0` contiguous, `-0.0` strided), so fp64
  cannot arbitrate between the paths - but the operation is defined as `0` in the `|x| <= lambda`
  branch, and the fp64 *contiguous* result is `+0.0`.  The **strided/scalar** path is the wrong one
  (and it is wrong in fp64 as well).

```python
import numpy as np, torch
gelu = torch.nn.functional.gelu
base = torch.full((128,), -10.0)
contig = gelu(torch.full((64,), -10.0))[0]     # 0.0   signbit False
strided = gelu(base[::2])[0]                   # -0.0  signbit True
print(repr(float(contig)), bool(np.signbit(float(contig))),
      repr(float(strided)), bool(np.signbit(float(strided))))
# 0.0 False -0.0 True
```

For `gelu(-10.0)` the mathematically justified fp32 result *is* `-0.0`: `gelu(x) = 0.5·x·(1+erf(x/√2))`
and `1 + erf(-10/√2)` rounds to exactly `0.0` in fp32 *and* in fp64, so the product of the negative
`x` with a positive zero is `-0.0`.  The contiguous fp32 path returns `+0.0`, i.e. it drops the sign,
while both the strided fp32 path and fp64 keep it.  For the non-saturating inputs the two paths also
differ in magnitude by ~1e-7 relative (`gelu(-0.5)`: `-0.1542687863` contiguous vs `-0.1542687714`
strided), which is a precision difference between two different implementations and is *not* part of
this report - but it confirms the two paths really are separate code.

Two independent inputs select the paths, which is why the defect is easy to miss:

* **layout**: contiguous (`+0.0`) vs a stride-2 view (`-0.0`) - also visible with `expand` (stride 0),
  which follows the scalar path;
* **width**: `gelu` flips at `n=2` and `softshrink` at `n=16` (fp32; `n=8` in fp64), the same
  scalar-loop/vectorised-loop boundary as PT-020/TF-003.

## Why it matters

`1 / gelu(x)`, `torch.log(gelu(x))`, `torch.sign`/`signbit` and `copysign` all read the sign of that
zero, so the value of a downstream expression changes with the memory layout of the input:

```python
x_contig  = torch.full((64,), -10.0)
x_strided = torch.full((128,), -10.0)[::2]
1.0 / gelu(x_contig)[0]    # inf
1.0 / gelu(x_strided)[0]   # -inf
```

The trigger is not exotic: any `unfold`/`transpose`/`narrow`/slice before the activation (very
common in attention and windowing code) switches the path, and so does running the same layer with a
different batch size.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing check(s).` (each replay runs in its own process)

The reproducer asserts, for both ops and several negative inputs:

* the contiguous and strided paths return **different signs** (this is the bug);
* the fp64 result in **both** layouts, and the documented `softshrink` branch, to say which fp32
  path is wrong in each case (`gelu`: the contiguous path, which disagrees with fp64; `softshrink`:
  the strided path, which disagrees with the `0` the branch is defined to return and with fp64
  contiguous);
* the width trigger (`n=1` vs `n>=2` for `gelu`, `n<=8` vs `n>=16` for `softshrink`) agrees with the
  layout story, i.e. it is the same scalar-vs-vectorised split;
* controls that must not move: `gelu(1.0)`, `gelu(0.0)`, `gelu(10.0)`, `softshrink(-10.0)`,
  `softshrink(1.0)`, `softshrink(0.0)` and `softshrink(0.25)` are identical in both layouts, and
  `gelu(-0.5)` agrees between the layouts to 1e-6 relative (only its sign of zero is at stake).

## The eager path is the affected one: `torch.compile` does not reproduce it

Measured with a strict harness that runs **one (op, layout) combination per process** and repeats it
in three separate processes (`artifact/hunt-v2/tools_compile_strict.py`), so no compile cache or
closure state can leak between measurements:

| combination | eager | `torch.compile` | arbitrated correct value |
|---|---|---|---|
| `gelu(-10.0)`, contiguous | `+0.0` | `-0.0` | `-0.0` (fp64) |
| `gelu(-10.0)`, stride-2 view | `-0.0` | `-0.0` | `-0.0` (fp64) |
| `softshrink(-0.5)`, contiguous | `+0.0` | `+0.0` | `+0.0` (branch/fp64) |
| `softshrink(-0.5)`, stride-2 view | `-0.0` | `+0.0` | `+0.0` (branch/fp64) |

`torch.compile` is therefore consistent across layouts and agrees with the arbitration in every case:
the defect is in the **eager** kernels, and compiling the same expression is a workaround.  That also
cross-validates the arbitration above, since the compiled kernels and fp64 independently agree on
which sign is right.

(An earlier harness of mine reported a *compiled* layout split for these two ops.  That was an
artifact of measuring many combinations inside one process; it is retracted, and the table above is
from the one-process-per-combination harness with 3/3 stable repeats.)

## Root cause (pointer)

Both kernels are computed through `TensorIterator`, which picks a vectorised contiguous loop when it
can and an unrolled scalar loop otherwise; each loop has its own implementation of the branch that
produces a zero (`gelu`'s saturated tail and `softshrink`'s `|x| ≤ λ` branch), and the two take the
sign of that zero from different expressions.  The fp32 vectorised loop for `gelu` and the
scalar/strided loop for `softshrink` are the wrong ones.

## Novelty assessment

* Searches on `pytorch/pytorch` for `gelu negative zero`, `gelu signed zero`, `gelu stride sign`,
  `gelu non-contiguous`, `softshrink signed zero`, `softshrink negative zero`, `sign of zero layout`
  and `contiguous strided signed zero` return no matching report.
* `pytorch#189804` ("Inductor vectorized GELU kernel produces tiny negative value (~-1.65e-7) where
  eager produces ...", closed 2026-07-14) is a different symptom on a different path: it is about the
  *Inductor*-generated kernel returning a small non-zero magnitude, whereas this report is about
  *eager* execution returning the wrong **sign** of a zero, with no compiler involved.  This build's
  commit (2026-06-09) also predates that closure, so it cannot be a regression of it.
* The separate PT-020 (wave 54) covers `torch.maximum`/`minimum`, which is the same
  scalar-loop/vectorised-loop dispatch but different kernels and a different (operand-order) trigger;
  this report was found independently by the new layout oracle.

## Files

* `repro.py` - self-checking reproducer (both ops, layout and width triggers, fp64 arbitration,
  controls, two process-isolated replays)
* `evidence.txt` - raw observations
* `collect_env.txt` - environment
