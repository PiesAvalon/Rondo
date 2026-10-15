# Complex addition loses the sign of zero - eagerly, and loses both zero signs under `torch.compile`

**Component:** PyTorch 2.12.1 (CPU), `torch.add` / the `+` operator / `Tensor.add_` on
`complex64`/`complex128`, eager and `torch.compile(dynamic=False)` (Inductor)
**Severity:** silent wrong sign of zero — a code path that inspects `signbit` (branch cuts, `1/z`,
`angle`, comparisons with `copysign`) takes the opposite branch; under `torch.compile` **both**
components are wrong at once
**Status:** present in torch 2.12.1+cu130 on CPU; NumPy (the IEEE reference) and the componentwise
recombine `torch.complex(a.real + b.real, a.imag + b.imag)` both preserve the signs
**Found with:** the wave-180 `torch.compile` complex probe (Reference vs Fresh oracle: eager vs a
freshly compiled callable), minimised to the sign-bit witnesses below; certified reproducer
`repro.py` in this directory (`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

IEEE-754 addition preserves a zero's sign: `(-0.0) + (-0.0) = -0.0` (and the sum of two zeros with
different signs is `+0.0`).  NumPy follows this for complex operands; PyTorch does not:

```python
import numpy as np, torch
z = torch.tensor([complex(-0.0, -0.0)], dtype=torch.complex128)   # signbits (real, imag) = (T, T)

torch.signbit(torch.view_as_real(z + z))          # -> (False, True)   real +0.0, imag -0.0
torch.signbit(torch.view_as_real(torch.from_numpy(np.asarray(z) + np.asarray(z))))
                                                  # -> (True,  True)   both -0.0  (correct)
torch.signbit(torch.view_as_real(torch.complex(z.real + z.real, z.imag + z.imag)))
                                                  # -> (True,  True)   the componentwise path is right
torch.signbit(torch.view_as_real(torch.compile(lambda a, b: a + b, dynamic=False)(z, z)))
                                                  # -> (False, False)  both signs lost
```

A second eager case loses the other component: with `z = c = complex(+0.0, -0.0)`, eager returns
`+0+0j` where NumPy returns `+0-0j`.

## Expected vs actual (sign bits, `True` = negative)

| operands | NumPy (reference) | eager `+` | `torch.compile` `a + b` (length 4) |
|---|---|---|---|
| `(-0-0j) + (-0-0j)` | `(True, True)` | **`(False, True)`** | **`(False, False)`** |
| `(+0-0j) + (+0-0j)` | `(False, True)` | **`(False, False)`** | `(False, False)` |
| `(-0+0j) + (-0+0j)` | `(True, False)` | `(True, False)` | **`(False, False)`** |
| `(+0+0j) + (+0+0j)` | `(False, False)` | `(False, False)` | `(False, False)` |

The eager loss is length-independent (verified for lengths 1-32) and identical for `complex64` and
`complex128`; `Tensor.add_` and `torch.add` show the same signs as `+`.  Under `torch.compile` both
signs are lost for lengths 1, 4, 16 and 32 with a freshly compiled callable; from roughly length 33
on the real part is still lost and the imaginary part is sometimes preserved (the exact bits past 32
depend on the process history, so the reproducer asserts only that the compiled result differs from
NumPy's `(True, True)` there).

## Scope and controls

* **Real-valued add is correct**: `[-0.0] + [-0.0]` keeps the `-0.0` sign bit.
* **Other complex operators are correct on the same operands**: `sub`, `mul` and `div` match NumPy's
  sign bits for the sampled zero combinations (for example `(0,0)+` for subtraction and
  multiplication, `(1,1)` for division by `1+0j`).
* **The componentwise recombine is correct**: `torch.complex(z.real + c.real, z.imag + c.imag)`
  returns `(True, True)` for the failing input, which is both the reference and a workaround.
* The operands' own sign bits are `(True, True)` before the addition (checked with
  `torch.signbit(torch.view_as_real(...))`), so the loss happens in the addition, not in tensor
  construction or in the test.

## Root cause (pointer)

The complex addition kernel produces the correct values but writes `+0.0` for a component whose
exact IEEE sum is `-0.0`; the componentwise formulation is exact, so the kernel (eager CPU complex
add, and Inductor's complex-add codegen) is computing or folding the zero case differently - likely
a `real`/`imag` split with a normalisation of zero.  `signbit`-level tests for the four zero
combinations would catch it.

## Impact

Signed zeros are semantic in complex arithmetic: `1/(x + 0j)` and `1/(x - 0j)` differ in sign, the
`angle` of `-0.0 ± 0.0j` is `±pi`, and branch-cut-sensitive functions (`sqrt`, `log`, `asin`,
`atanh`) choose the side of the cut from the zero sign.  Code that adds a zero-imaginary constant
(for example `z + complex(0.0, -0.0)`) or that compares `signbit(real)` can therefore silently take
the wrong branch, and the compiled path loses even more of the information than eager does.

## Novelty

Searched the PyTorch tracker for `complex add negative zero`, `complex addition signed zero`,
`complex add -0.0`, `torch complex signed zero` and `complex128 add sign of zero`: no report exists.
The campaign's other signed-zero findings are different operators and paths (`PT-012`
`maximum`/`minimum`/`fmin`/`fmax`/`clamp` in Inductor, `PT-019` forward-mode derivatives of `-0.0`,
`PT-020` `maximum`/`minimum` above the vector width, `PT-021` `gelu`/`softshrink` eager width
dependence, and `PT-023` `set_default_dtype`); none covers complex addition.

## Environment

torch 2.12.1+cu130 (CPU execution), NumPy 2.5.1, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"The sign of a zero is not observable."**  It is: `torch.signbit`, `1/z`, `torch.angle` and the
  branch cuts all read it, and NumPy and the componentwise torch path both keep it, so eager and
  compiled disagree with the reference *and with each other*.
* **"`(-0.0) + (-0.0)` is a corner case."**  It is the canonical way a zero-valued imaginary part
  arises (an added `0-0j` constant, a subtraction that cancels, a product of zeros), and the
  compiled path loses both components, so the corner case is reachable from ordinary code.
* **"This is a known signed-zero family."**  The campaign's other signed-zero drafts are for
  `min`/`max`/`clamp`/activation/JVP paths; complex addition is a different kernel and is not
  mentioned in any of them or in the tracker search above.
