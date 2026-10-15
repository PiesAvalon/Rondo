# `[inductor] float(shape % k == c)` crashes with `AssertionError: ToFloat(False)` in the symbolic value-range analysis

**Component:** `torch.compile` → Inductor bound analysis / `torch/utils/_sympy/value_ranges.py`
**Severity:** crash (compilation failure) — valid eager program cannot be compiled
**Status:** present on `main` (`to_dtype`/`increasing_map` unchanged)
**Found with:** Rondo differential oracle (eager vs `torch.compile(dynamic=True)`) + `torch.sym_*` API fuzzing (`sym_api_fuzz.py`)

---

## Summary

Converting a **modulo-equality on a symbolic size** to a float aborts compilation:

```python
import torch

def f(x):
    return x + float(x.shape[0] % 2 == 0)

torch.compile(f, dynamic=True, fullgraph=True)(torch.randn(6, 3))
# torch._inductor.exc.InductorError: AssertionError: ToFloat(False)
```

Eager returns `x + 1.0`. Under `torch.compile(dynamic=True)` Inductor fails while computing
loop bounds:

```
torch/_inductor/bounds.py:145        get_index
torch/utils/_sympy/value_ranges.py:1142  bound_sympy
torch/utils/_sympy/interp.py:214     sympy_interp
torch/utils/_sympy/interp.py:132     _run_sympy_handler
      return analysis.to_dtype(args[0], torch.float64)
torch/utils/_sympy/value_ranges.py:485   to_dtype
      return ValueRanges.increasing_map(a, ToFloat)
torch/utils/_sympy/value_ranges.py:345   increasing_map
      return ValueRanges(fn(x.lower), fn(x.upper))
torch/utils/_sympy/value_ranges.py:158   ValueRanges.__init__
torch/utils/_sympy/value_ranges.py:67    simple_sympify
      raise AssertionError(e)
torch._inductor.exc.InductorError: AssertionError: ToFloat(False)
```

The comparison `s0 % 2 == 0` has a *boolean*-valued range. `to_dtype` applies `ToFloat`
to each bound of that range, producing the SymPy object `ToFloat(False)`; `simple_sympify`
only accepts numbers and raises, turning a normal expression into a compiler crash.

## Reproduction results

`python repro.py --replays 3` (in this directory):

| trigger | eager | compiled (`dynamic=True`) |
|---|---|---|
| `float(x.shape[0] % 2 == 0)` | ok | **AssertionError: ToFloat(False)** |
| `float(x.shape[0] % 2 != 0)` | ok | **AssertionError: ToFloat(False)** |
| `float(x.shape[0] % 2 == 1)` | ok | **AssertionError: ToFloat(False)** |
| `float(x.shape[0] % 3 == 0)` | ok | **AssertionError: ToFloat(False)** |
| `float(torch.sym_not(x.shape[0] % 2 == 0))` | ok | **AssertionError: ToFloat(False)** |
| `2.0 * float(x.shape[0] % 2 == 0)` | ok | **AssertionError: ToFloat(False)** |
| control: `float(x.shape[0] % 2)` | ok | ok |
| control: `float(x.shape[0] == 6)` | ok | ok |
| control: `float(x.shape[0] > 2)` | ok | ok |
| control: `float(x.shape[0])` | ok | ok |

* 12/12 trigger runs fail (6 triggers × CPU + CUDA), 3/3 replays each, deterministic.
* Every control passes on both devices: the failure needs the **modulo** comparison. Plain
  equality, inequality, ordering and a bare modulo all compile fine, so the trigger is
  specifically a boolean range produced by `PythonMod` inside `Eq`/`Ne`.
* `dynamic=False` specialises the size and does not trigger the bound analysis.

## Root cause

`torch/utils/_sympy/value_ranges.py`:

```python
@staticmethod
def increasing_map(x: ExprIn | ExprVR, fn: ExprFn) -> ExprVR:
    """Increasing: x <= y => f(x) <= f(y)."""
    x = ValueRanges.wrap(x)
    return ValueRanges(fn(x.lower), fn(x.upper))      # fn(x.lower) -> ToFloat(False)

@staticmethod
def to_dtype(a, dtype, src_dtype=None):
    if dtype == torch.float64:
        return ValueRanges.increasing_map(a, ToFloat)  # <-- applied to a bool range
    ...
```

`ValueRanges.__init__` funnels each bound through `simple_sympify`
(`value_ranges.py:67`), which raises for any `sympy.Expr` that is not a number.
`ToFloat(False)` is exactly such an expression, because its argument is the Python
boolean `False` rather than a sympified `sympy.false`/`BooleanTrue`.

Note that `SymPyValueRangeAnalysis` otherwise implements the full `interp.handlers()`
registry, so this path is *not* caught by registry/implementation diffing — `ToFloat` is
dispatched by the special case at `interp.py:131`, which calls `to_dtype` directly.

## Suggested fix

Handle boolean-valued ranges in `to_dtype` (or sympify the boolean before building the
`ToFloat` node):

```python
@staticmethod
def to_dtype(a, dtype, src_dtype=None):
    if dtype == torch.float64:
        if isinstance(a, ValueRanges) and a.is_bool:
            # False -> 0.0, True -> 1.0
            lower = ValueRanges.wrap(a.lower)
            upper = ValueRanges.wrap(a.upper)
            return ValueRanges(ToFloat(BooleanTrue() if upper.upper else BooleanFalse()),
                               ToFloat(BooleanTrue() if lower.lower else BooleanFalse()))
        return ValueRanges.increasing_map(a, ToFloat)
```

A more robust variant would make `simple_sympify` accept Python `bool` by mapping it to
`sympy.S.true`/`sympy.S.false`, which also removes the class of failures where a boolean
leaks into a numeric range.

## Prior art

* The same assertion string appears in **#131897** ("Failed to export module with dynamic
  shapes", 2024, closed): that report triggered it through
  `F.interpolate(..., scale_factor=...)` under `torch.export`. It was resolved on the
  interpolate side; the `value_ranges.to_dtype` code path itself is unchanged on `main`,
  and the direct user-level expression above still fails.
* Searches for `to_dtype ToFloat value_ranges` and `ToFloat bool value range` return no
  other matching report.
* `#187876` ("Fix mark dynamic half bounded range") is unrelated.

## Impact

`float(shape % k == c)` is a natural way to derive a 0/1 mask or a scale factor from a
dynamic dimension (e.g. "is this batch even", "pad to the next multiple"). Such code
compiles under `dynamic=False` and runs eagerly, but crashes with an internal
`AssertionError` under `dynamic=True`, which gives the user no indication of what is
unsupported. The failure is in the *bounds* pass, so it also affects any larger graph whose
index expressions contain such a value.

## Independence from the `ValueRanges.eq`/`ne` boolean fix

The same boolean range that reaches `to_dtype` is produced by `ValueRanges.eq`, and upstream
has a separate PR for that function (#191886, "Fix ValueRanges.eq/ne raising TypeError on
boolean ranges"). Applying that fix's idiom locally — mapping boolean operands to `{0, 1}`
before the disjoint test — does **not** fix this bug:

`python independence.py` output:

```
baseline (unpatched):
  PT-004 float(s % 2 == 0)                     ToFloat-assert
  (s % 2 == 0) == True                         EqNe-TypeError
after applying the #191886 boolean guard to ValueRanges.eq:
  PT-004 float(s % 2 == 0)                     ToFloat-assert   <-- unaffected
  (s % 2 == 0) == True                         ToFloat-assert   <-- eq fix worked, next gap revealed
```

So this report targets a distinct defect (`to_dtype` / `simple_sympify`) that remains after
the known `eq`/`ne` fix lands.

## The suggested fix is verified

`fix_check.py` monkeypatches `SymPyValueRangeAnalysis.to_dtype` with the mapping above and
re-runs the reproducers:

```
--- baseline
    float(s % 2 == 0)          FAIL ToFloat-assert
    bool(s % 2)                FAIL ToFloat-assert
    float(s % 3 == 1)          FAIL ToFloat-assert
    2.0 * float(s % 2 == 0)    FAIL ToFloat-assert
    control float(s % 2)       ok same=True
    control float(s > 2)       ok same=True
--- with suggested to_dtype fix
    float(s % 2 == 0)          ok same=True
    bool(s % 2)                ok same=True
    float(s % 3 == 1)          ok same=True
    2.0 * float(s % 2 == 0)    ok same=True
    control float(s % 2)       ok same=True
    control float(s > 2)       ok same=True
```

All four crashes are resolved and produce results matching eager, while the two controls are
unaffected — so the defect is confined to the boolean branch of `to_dtype`, and the fix
outlined above is sufficient for the reported cases.
