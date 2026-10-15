# `torch.sym_ite` crashes under `torch.compile(dynamic=True)` when the two branches have different Python types

**Component:** `torch.compile` (Dynamo) / symbolic shapes
**Severity:** crash (compilation failure) — valid eager program cannot be compiled
**Status:** present on `main` (verified against `torch/__init__.py` at `main`)
**Found with:** Rondo differential oracle (Reference/eager vs Fresh-compiled) + symbolic-scalar fuzzing

---

## Summary

`torch.sym_ite(cond, then, other)` rejects operands whose Python types differ:

```python
# torch/__init__.py
def sym_ite(b, t, f):
    """SymInt-aware utility for ternary operator (``t if b else f``.)"""
    if overrides.has_torch_function((b, t, f)):
        return overrides.handle_torch_function(sym_ite, (b, t, f), b, t, f)
    if not isinstance(b, (SymBool, builtins.bool)):
        raise AssertionError(f"expected SymBool or bool, got {type(b)}")
    if type(t) is not type(f):
        raise AssertionError(f"type mismatch: {type(t)} vs {type(f)}")   # <-- here
    if isinstance(b, SymBool):
        return b.__sym_ite__(t, f)
    return t if b else f
```

In eager code a dynamic scalar argument is a plain `int`/`float`/`bool`, so
`torch.sym_ite(cond, k, 1)` has `type(t) is type(f) is int` and works. Under
`torch.compile(dynamic=True)` the same argument becomes a `torch.SymInt`, so the
identical expression has `type(t) is SymInt` and `type(f) is int`, and the strict
`is not` check fires.

Mixing a symbolic branch with a literal branch is the normal, intended use of a
symbolic conditional (the literal is the constant arm), and it works in eager — so
this rejects valid user code.

## Minimal reproducer

```python
import torch

def f(x, k):
    return x + float(torch.sym_ite(k > 2, k, 1))   # k becomes SymInt under dynamic=True

x = torch.randn(4, 3)
print(f(x, 5))                                    # eager: works

torch.compile(f, dynamic=True, fullgraph=True)(x, 5)
# torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
#   Explanation: Dynamo failed to run FX node with fake tensors:
#     call_function <function sym_ite>(*(Gt(s0, 2), s0, 1), **{}):
#     got AssertionError("type mismatch: <class 'torch.SymInt'> vs <class 'int'>")
```

`float(...)` is not required — `x + torch.sym_ite(k > 2, k, 1)` fails the same way,
as does using the result as a shape.

## Reproduction results

`python repro.py --replays 3` (see `repro.py` in this directory):

| case | eager | compiled (`dynamic=True`) |
|---|---|---|
| `torch.sym_ite(k > 2, k, 1)` (SymInt vs `int`) | ok | **AssertionError: `SymInt` vs `int`** |
| `torch.sym_ite(k > 2, 1, k)` (`int` vs SymInt) | ok | **AssertionError: `int` vs `SymInt`** |
| `torch.sym_ite(k % 2 == 0, k * 2, 1)` | ok | **AssertionError: `SymInt` vs `int`** |
| `torch.ones(torch.sym_ite(k > 2, k, 1))` (shape use) | ok | **AssertionError: `SymInt` vs `int`** |
| `torch.ones(torch.sym_ite(x.shape[0] > 2, x.shape[0], 1))` — **no conversion, pure shape use** | ok | **AssertionError: `SymInt` vs `int`** |
| `x + float(torch.sym_ite(x.shape[0] > 2, x.shape[0], 1))` | ok | **AssertionError: `SymInt` vs `int`** |
| `torch.sym_ite(k > 2, fa, 1.0)` (SymFloat vs `float`) | ok | **AssertionError: `SymFloat` vs `float`** |
| `torch.sym_ite(ba, k > 2, True)` (SymBool vs `bool`) | ok | **AssertionError: `SymBool` vs `bool`** |
| control: `torch.sym_ite(k > 2, k, k + 0)` (SymInt vs SymInt) | ok | ok |

* 16/16 non-control runs fail (8 cases × CPU + CUDA), 3/3 replays each, deterministic.
* The sixth case needs no `float()`/`int()` conversion at all — the mixed arms are the
  tensor's own symbolic size and a literal — so this is not a scalar-tensorification
  corner case but a plain shape computation.
* Control case passes on both devices, confirming the difference is purely the operand types.
* `dynamic=False` specialises `k` to a constant, so it is unaffected — the bug is specific to
  dynamic shapes.

## Root cause

`torch/__init__.py`, `sym_ite()`:

```python
if type(t) is not type(f):
    raise AssertionError(f"type mismatch: {type(t)} vs {type(f)}")
```

The identity comparison is intended to catch genuinely incompatible branches
(e.g. a `SymInt` in one arm and a `SymFloat` in the other), but it also rejects the
perfectly well-defined combination *symbolic value + Python literal*, which eager
execution accepts and which Dynamo itself produces when it turns a scalar argument
into a `SymInt`/`SymFloat`/`SymBool`.

Dynamo reaches `torch.sym_ite` while executing the FX node with fake tensors
(`torch/_dynamo/utils.py:get_fake_value`), so the assertion surfaces as an
`TorchRuntimeError` ("RuntimeError when making fake tensor call") rather than a clear
type error.

## Suggested fix

Promote the literal arm instead of rejecting it, e.g. coerce `int`/`float`/`bool`
operands to the corresponding symbolic wrapper before the identity check:

```python
def sym_ite(b, t, f):
    if overrides.has_torch_function((b, t, f)):
        return overrides.handle_torch_function(sym_ite, (b, t, f), b, t, f)
    if not isinstance(b, (SymBool, builtins.bool)):
        raise AssertionError(f"expected SymBool or bool, got {type(b)}")
    if isinstance(t, SymInt) and isinstance(f, builtins.int):
        f = torch.sym_int(f)
    elif isinstance(f, SymInt) and isinstance(t, builtins.int):
        t = torch.sym_int(t)
    # ... same for SymFloat/float and SymBool/bool ...
    if type(t) is not type(f):
        raise AssertionError(f"type mismatch: {type(t)} vs {type(f)}")
    ...
```

At minimum, the check should accept `(SymInt, int)` / `(SymFloat, float)` /
`(SymBool, bool)` in either order, since `SymInt` already interoperates with `int`
everywhere else.

## Prior art

* `torch.sym_ite` was added in #111440 "Add support for sym_ite".
* Upstream tests only ever call `torch.sym_ite` with **both** branches symbolic
  (`test/test_dynamic_shapes.py:test_sym_ite`, `test_tracing_sym_ite`), so the mixed
  case is untested.
* Searches for `sym_ite` combined with `type mismatch` / `AssertionError` on the PyTorch
  tracker return only unrelated PRs (#184327 "Fix symbolic magic method lowerings",
  #186027, #185545); no report of this assertion firing.
* The `type(t) is not type(f)` check is still present on `main`.

## Impact

Any `torch.compile(dynamic=True)` user who writes a symbolic conditional with a
constant arm — a very natural pattern, e.g.
`torch.sym_ite(n > 0, n, 1)`, `torch.sym_ite(flag, batch, 1)` — gets a compiler crash
instead of a compiled graph. It also blocks `torch.sym_max`/`sym_min`-style bound
clamping built from `sym_ite`.
