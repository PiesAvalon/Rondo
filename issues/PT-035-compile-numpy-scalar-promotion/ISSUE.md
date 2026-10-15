# `torch.compile` changes type promotion for NumPy scalars and 0-dim `ndarray` operands

**Component:** PyTorch 2.12.1+cu130, `torch.compile` / `torch._dynamo` on CPU.  Reproduces with
`backend="eager"`, `backend="aot_eager"` **and** `backend="inductor"`, so the change happens during
tracing, not in Inductor codegen.
**Severity:** silent dtype **and value** change for the same source expression: a model that mixes
NumPy scalars or 0-dim `ndarray`s into tensor arithmetic returns a different dtype with and without
`torch.compile` (float32 vs float64, int32 vs int64, float16 vs float64), which changes downstream
numerics, doubles/quarters memory traffic and breaks dtype asserts.

```python
import numpy as np, torch
x = torch.tensor([1, -2, 3], dtype=torch.int32)
(x + np.float64(2.5)).dtype                      # eager:    torch.float32
torch.compile(lambda a: a + np.float64(2.5))(x).dtype   # compiled: torch.float64
```

**Status:** present in 2.12.1+cu130 in a fresh process, on every backend, with a cold Dynamo cache.
**Found with:** the wave-407 Rondo dtype-promotion metadata oracle (`torch.compile` vs eager over a
12-dtype x 10-operand grid); certified reproducer `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 39 checks across four fresh-process lanes).

---

## Summary

| expression (1-D tensor op operand) | eager | compiled (all backends) |
|---|---|---|
| `int32 + np.float64(2.5)` | `float32` | **`float64`** |
| `uint8 * np.float64(2.5)` | `float32` | **`float64`** |
| `int32 / np.float64(3.0)` | `float32` | **`float64`** (different bits) |
| `bool + np.int32(2)` | `int64` | **`int32`** |
| `float32 + np.array(2.5)` (0-dim `ndarray`) | `float64` | **`float32`** |
| `float16 + np.array(2.5)` | `float64` | **`float16`** |

Controls that do **not** change dtype under compile (all four lanes agree): `int32 + 2.5`
(Python float -> `float32`), `int32 + torch.tensor(2.5, dtype=float64)` (0-dim tensor -> `float64`),
`float32 + torch.tensor(2.5, dtype=float64)` (-> `float32`), `int32 + np.float32(2.5)` (-> `float32`).

## Which side is documented

PyTorch's own promotion API agrees with **eager**, so the compiled side is the deviation:

* `torch.result_type(torch.tensor([1], dtype=torch.int32), np.float64(2.5))` -> **`torch.float32`**
  (a NumPy scalar is a *wrapped number*, i.e. weak, like a Python scalar);
* `torch.result_type(torch.tensor([1], dtype=torch.int32), torch.tensor(2.5, dtype=torch.float64))`
  -> **`torch.float64`** (a 0-dim *tensor* is a strong operand).

A 0-dim `numpy.ndarray` behaves like a 0-dim tensor in eager (`float32 + np.array(2.5)` -> `float64`),
which the compiled lanes treat as a weak Python scalar instead.

## Root cause

Dynamo represents a NumPy scalar as a 0-dim `NumpyNdarrayVariable` (a `TensorVariable` subclass - see
PR #185815, "Fix Dynamo NumPy scalar identity handling", for the same representation), so the traced
graph promotes it with the **tensor** category (strong).  Eager's binary-op path wraps the NumPy
scalar as a Python number (weak).  For a 0-dim `ndarray` the mapping is inverted: Dynamo unwraps it to
a Python scalar, while eager keeps the float64 tensor category.  Because all three backends agree with
each other and disagree with eager, this is a tracing-fidelity defect, not an Inductor kernel defect.

## Why this is not a precision artifact

* The dtype itself differs (`float32` vs `float64`, `int32` vs `int64`, `float16` vs `float64`), which
  is a metadata contract change, not rounding.
* For `int32 / np.float64(3.0)` the float64 lane also produces different values: widening the eager
  float32 result to float64 does not reproduce the compiled result (`1/3` and `2/3` round differently).
* Python scalars and 0-dim *tensors* keep identical dtypes across all lanes, so the divergence is
  specific to the NumPy operand kinds and not a general compile-vs-eager precision difference.
* `torch.result_type` is the arbiter and sides with eager for the NumPy-scalar case.

## Minimal reproduction

```python
import numpy as np, torch

x = torch.tensor([1, -2, 3], dtype=torch.int32)
f = lambda a: a + np.float64(2.5)
print(f(x).dtype)                       # torch.float32
print(torch.compile(f)(x).dtype)        # torch.float64   <- same expression, different dtype/value

y = torch.tensor([1.5, -2.0], dtype=torch.float32)
g = lambda a: a + np.array(2.5)         # 0-dim ndarray
print(g(y).dtype)                       # torch.float64
print(torch.compile(g)(y).dtype)        # torch.float32   <- reversed direction
```

## Additional manifestations (wave 409 extension, same tracing defect)

Multi-element `ndarray` operands behave like the 0-dim ones - the same swap of the promotion category -
and the extended reproducer now covers them (`--replays 2` -> 51 checks, certified by
`results/w410_PT-035-VERIFICATION.md`):

| expression | eager | compiled (all backends) |
|---|---|---|
| `float32 + np.array([1,2,3], dtype=int32)` | `float64` | **`float32`** |
| `int32 * np.array([1.5,2.5,3.5], dtype=float16)` | `float64` | **`float16`** |
| `bool \| np.array([True,False,True])` | `uint8` | **`bool`** |

The wider grid (`results/w409_torch_numpy_operands.py`) also shows that the same typing difference can
turn a working eager expression into a **compile-time error**: `torch.compile(lambda x: x @ np.arange(9.).reshape(3,3))(float32_tensor)`
raises `RuntimeError: expected m1 and m2 to have the same dtype`, `torch.where(np_bool_array, x, 0)`
raises `TypeError: where() received an invalid combination of arguments`, and `bool_tensor | np.float64_array`
/ `float64_tensor - np.bool_array` raise in one of the compiled lanes although eager evaluates them.
Those cases are archived in `results/w409_torch_numpy_operands.out` rather than asserted by the
reproducer, because the eager lane is the only reference for their dtypes.

## Novelty check

Searched the PyTorch tracker for Dynamo/`torch.compile` NumPy-scalar promotion and dtype reports.  The
only close item is **PR #185815** ("Fix Dynamo NumPy scalar identity handling", closed, not merged),
which is about Python object **identity** of NumPy scalar singletons (`np.bool_(True) is ...`, alias
guards) and explicitly does not touch promotion categories.  No issue or PR was found for the dtype
and value change described here.
