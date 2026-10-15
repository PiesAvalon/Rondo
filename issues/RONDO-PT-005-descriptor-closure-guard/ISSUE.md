# `[dynamo]` `torch.compile` silently returns another function's result when two closures share a code object and capture different `torch.Tensor` method descriptors

> **See also:** `findings/DYNAMO-GUARD-MODEL.md` places this report alongside the other three guard-model findings (`RONDO-PT-005/006/007/008`), lists everything that was verified as *correctly* guarded, and sketches a single fix.


**Component:** `torch.compile` (Dynamo) — guard emission for closure cells
**Severity:** **silent wrong results** (not a crash); deterministic and reproducible
**Status:** present in `torch 2.12.1+cu130`; the guard dump below shows the missing guard
**Found with:** Rondo OpInfo differential campaign (`opinfo_campaign.py`) + `verify_candidates.py` process-isolated confirmation

---

## Summary

`torch.compile` does not emit any guard for a closure cell that holds a `torch.Tensor`
C method descriptor. Because Dynamo keys its cache on the code object, a second function
created by the *same factory* — same code object, different captured operation — is treated
as a cache hit and **executes the first function's compiled graph**, returning wrong values
with no warning or error.

```python
import torch

def make(op):
    def fn(x, y):
        return op(x, y)
    return fn

add = make(torch.Tensor.__add__)
mul = make(torch.Tensor.__mul__)

x = torch.tensor(6.0)
y = torch.tensor(3.0)
```

```
eager     add(x, y) = 9.0     mul(x, y) = 18.0
compiled  add(x, y) = 9.0     mul(x, y) = 9.0     <-- silently wrong
```

`torch.compile(mul)` returns `9.0` — the result of `add`. This is completely silent: no
graph break, no warning, no error, and `mul` is still "compiled".

## Reproduction results

`python repro.py --replays 3` (in this directory):

| case | eager B | compiled B | result |
|---|---|---|---|
| `Tensor.__add__` then `Tensor.__mul__` | 18.0 | **9.0** | 3/3 replays wrong, CPU + CUDA |
| `Tensor.__sub__` then `Tensor.__mul__` | 18.0 | **3.0** | 3/3 replays wrong, CPU + CUDA |
| `Tensor.abs` then `Tensor.neg` | `[1.5, -2.0, 3.0]` | **`[1.5, 2.0, 3.0]`** | 3/3 replays wrong, CPU + CUDA |
| control: `torch.add` then `torch.mul` (builtins) | 18.0 | 18.0 | ok |
| control: two top-level `def`s (distinct code objects) | 18.0 | 18.0 | ok |
| control: `torch._dynamo.reset()` between the two compiles | 18.0 | 18.0 | ok |

The bug also fires with `dynamic=False`, and with the operation supplied as a *default
argument* instead of a closure cell.

## Mechanism: the closure cell is never guarded

`TORCH_LOGS=guards` for `torch.compile(make(torch.Tensor.__add__))` prints the full guard
list (`guard_dump.txt`):

```
+- GuardManager: source=L['x'], ... type=<class 'torch.Tensor'>
| +- TENSOR_MATCH: check_tensor(L['x'], Tensor, ..., size=[], stride=[])
| +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False
| +- NO_TENSOR_ALIASING: check_no_aliasing(L['x'], L['y'])
+- GuardManager: source=L['y'], ... type=<class 'torch.Tensor'>
| +- TENSOR_MATCH: check_tensor(L['y'], Tensor, ..., size=[], stride=[])
| +- NO_HASATTR: hasattr(L['y'], '_dynamo_dynamic_indices') == False
| +- NO_TENSOR_ALIASING
```

There is **no guard referencing the closure cell `op`**. The only inputs guarded are the
tensor arguments, which are identical for `add` and `mul`, so the second `torch.compile`
finds the first entry's guards satisfied and reuses its graph.

`error_on_recompile` confirms it is a pure cache hit:

```python
torch._dynamo.config.error_on_recompile = True
torch._dynamo.reset()
torch.compile(make(torch.Tensor.__add__), dynamic=True)(x, y)   # compiles
torch.compile(make(torch.Tensor.__mul__), dynamic=True)(x, y)   # NO RecompileError -> cache hit
# control:
torch.compile(make(torch.add), dynamic=True)(x, y)
torch.compile(make(torch.mul), dynamic=True)(x, y)              # raises RecompileError -> recompiled
```

## Scope

`scope_fuzz.py` runs the same template over 39 captured-value kinds: it builds two functions
from one factory (one code object, different captured value), compiles the first, then
compiles the second, and checks the result. Eager A and eager B are asserted to differ so
that graph reuse is observable.

| captured value kind | result |
|---|---|
| `torch.Tensor` **method descriptors** — `__add__`, `__sub__`, `__mul__`, `__pow__`, `__rpow__`, `__lt__`, `__gt__`, `add`, `mul`, `abs`, `neg`, `sum`, `mean`, `numel`, `dim`, `size`, `stride` | **10 cases wrong** |
| `torch` builtin functions (`torch.add`, `torch.mul`, `torch.neg`, `torch.exp`) | correct |
| Python functions / lambdas (`F.relu`, `F.tanh`, `operator.add`, `math.sin`) | correct |
| bound method wrappers (`x.__add__`, `str.upper`) | correct |
| captured **tensors** (full/arange/ones/scalar/views) | correct |
| `torch.dtype` values, `torch.Size`, shape tuples, `int`/`float` scalars, `slice` objects, `dim` ints, `nn.Module` instances, `functools.partial` objects | correct |

The decisive property is the Python type of the captured object:
`type(torch.Tensor.__add__)` is `types.MethodDescriptorType` and is never guarded, while
`type(torch.add)` is `builtin_function_or_method` and `type(x.__add__)` is also
`builtin_function_or_method` — both of which are guarded correctly.

### Five ways to hit it

1. **Closure cell** — the factory pattern above.
2. **Default argument** — `def fn(x, y, _v=torch.Tensor.__add__): return _v(x, y)`.
3. **`nn.Module` attribute**:

```python
class M(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
    def forward(self, x, y):
        return self.op(x, y)

torch.compile(M(torch.Tensor.__add__), dynamic=True)(x, y)   # 9.0
torch.compile(M(torch.Tensor.__mul__), dynamic=True)(x, y)   # 9.0  <-- must be 18.0
```

`guard_dump_module_attr.txt` shows the guards for that module: `TENSOR_MATCH` on `x` and
`y`, a `TYPE_MATCH` on `self`'s type, and **nothing at all on `self.op`**.

4. **Module global rebound after compilation** (the monkeypatch / config-swap pattern) —
   `rebind_fuzz.py`:

```python
namespace = {}
exec("def program(x):\n    return GLOBAL_OP(x)\n", namespace)
namespace["GLOBAL_OP"] = torch.Tensor.abs
program = namespace["program"]

compiled = torch.compile(program, dynamic=True)
compiled(x)                                  # abs
namespace["GLOBAL_OP"] = torch.Tensor.neg    # rebind the global
compiled(x)                                  # still abs -> WRONG
```

`rebind_fuzz.py` rebinds 18 kinds of module global after compilation — a builtin function,
`Tensor` descriptors and dunders, a float, an int, a dtype, a pad-mode string, a class
attribute, an `nn.Module` instance, a shape tuple, **whole tensors** (4-element and 0-dim), a
dict, a list, a bool, a slice, and `functools.partial` objects. Only the values that are (or
wrap) a `torch.Tensor` method descriptor are not guarded:

| rebound global | guarded? |
|---|---|
| `torch.Tensor.abs` → `torch.Tensor.neg` | **no** |
| `torch.Tensor.__add__` → `torch.Tensor.__mul__` | **no** |
| `functools.partial(T.__add__)` → `functools.partial(T.__mul__)` | **no** |
| global tensor `full(4,1.0)` → `full(4,100.0)` | yes |
| global 0-dim tensor `1.0` → `100.0` | yes |
| `torch.neg` → `torch.exp`, `functools.partial(torch.add)` → `partial(torch.mul)` | yes |
| float, int, dtype, string, bool, slice, shape tuple, list, dict, class attribute, `nn.Module` | yes |

Notably a rebindable **global tensor** — the more dangerous pattern — is guarded correctly, so
the defect is confined to the descriptor case.

5. **Module attribute swapped after compilation** — `self.op = T.__add__` → `T.__mul__` once
   the module has already been compiled; `mutation_guard_fuzz.py` reproduces it on CPU and CUDA.

## Root cause

Dynamo emits no guard for a value whose type is `types.MethodDescriptorType`. Every guard
list collected here (`guard_dump.txt`, `guard_dump_module_attr.txt`) contains guards for the
tensor arguments and, in the module case, the receiver's type — but never a guard on the
descriptor itself.

Because the Dynamo cache is keyed on the code object plus guards, a second function built
from the same code object with a different descriptor satisfies the first entry's guards and
silently reuses its graph. `torch/_dynamo/variables/user_defined.py` materialises C-level
descriptors with `VariableTracker.build(tx, cls_attr, source)` in the
`inspect.ismethoddescriptor(cls_attr)` branch, without installing a guard on the value.

## Suggested fix

Emit a guard (e.g. `ID_MATCH`, or a `FUNCTION_MATCH` on the descriptor's
`__objclass__`/`__name__`) whenever a closure cell or default argument is materialised into
a variable, including the `types.MethodDescriptorType` path. Alternatively include the
closed-over values in the cache key. A regression test is easy: compile two factory
clones capturing `Tensor.__add__` and `Tensor.__mul__` and assert that the second result
matches eager.

## Related upstream work (novelty assessment)

Dynamo's handling of C descriptors has seen several recent fixes, all about *binding* a
descriptor through `tp_descr_get_impl` (`obj.method`), and none of them about the guard on a
*closure cell* that stores the raw descriptor:

* #190755 "Some Dynamo descriptors (`tp_descr_get_impl`) are missing a `descr_check` equivalent" (closed)
* #190776 "[Dynamo] Add descr_check to C descriptor VT tp_descr_get_impl" (closed)
* #191261 "Run descr_check when binding wrapper/method descriptors in Dynamo" (closed)
* #194556 "[dynamo] Preserve binding for nonstandard C method descriptors" (closed)
* #185823 "[Dynamo] Handle class method_descriptor calls generically" (closed)

This report is about the closure-cell guard, a different mechanism, and the behaviour is
still present in `2.12.1+cu130`. Targeted tracker searches for
`Tensor.__add__ torch.compile`, `python_closure guard dynamo`, `method_descriptor guard
dynamo` and `torch.compile silently wrong result closure` return no matching report.

## Impact

Silently wrong results are the worst possible failure mode for `torch.compile`. The factory
pattern — build a family of small functions from a table of operations — is common
(dispatch tables, `functools.partial`-style wrappers, generated models), and the failure is
invisible: the program compiles, runs, and returns another operation's output. Any model
that compiles two such generated functions in one process is silently corrupted.

## Files

* `repro.py` — CPU + CUDA reproducer with controls and replay counts
* `scope_fuzz.py` — 39-case guard-emission fuzzer across captured-value kinds
* `rebind_fuzz.py` — 10-case fuzzer that rebinds module globals after compilation
* `guard_dump.txt` — `TORCH_LOGS=guards` output for the closure case (no guard on `op`)
* `guard_dump_module_attr.txt` — `TORCH_LOGS=guards` output for the module case (no guard on `self.op`)
* `collect_env.txt` — environment
