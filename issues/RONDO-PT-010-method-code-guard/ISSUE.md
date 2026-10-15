# `[dynamo]` the `__code__` of an inlined **bound method** is not guarded — mutating `method.__func__.__code__` after `torch.compile` is silently ignored

> **See also:** `findings/DYNAMO-GUARD-MODEL.md` and the sibling guard-model findings
> `RONDO-PT-005/006/007/008/009`.  This one is *not* fixed by the class-`__dict__` guard proposed
> in PT-006: here the method object identity and the class `__dict__` value are both unchanged —
> only the code object inside the function changes.

**Component:** `torch.compile` (Dynamo) — guard emission for inlined user functions
**Severity:** **silent wrong results** (not a crash), all backends including `backend="eager"`
**Status:** present in `torch 2.12.1+cu130`; guard dump below shows the asymmetry
**Found with:** Rondo post-compile state-mutation oracle + guard-tree audit

---

## Summary

Dynamo guards `f.__code__` (via a `CodeGuardAccessor`) whenever it inlines a function it reaches
as a global, as a plain class attribute, as a `staticmethod`, as a stored bound-method object, or
through `super()`.  When it inlines the **bound method** produced by `obj.m` or `Cls.cm`, the only
guards installed are:

```
+- GuardManager: source=G['model'], type=<class 'Model'>
|  +- TYPE_MATCH: ___check_type_id(G['model'], ...), type=<class 'Model'>
|  +- GuardManager: source=G['model'].__dict__
|  |  +- DICT_CONTAINS: not ___dict_contains('m', G['model'].__dict__)
```

There is no guard on the function object and none on its `__code__`.  Assigning
`Model.m.__code__ = other.__code__` in place — the standard way to monkeypatch a function's
implementation without replacing it — therefore changes what eager returns while every guard still
passes, and the retained compiled callable keeps running the old code.

```python
import torch

class Model:
    def m(self, t):
        return t * 2.0

def replacement(self, t):
    return t * 9.0

model = Model()
x = torch.arange(1.0, 4.0)
compiled = torch.compile(lambda t: model.m(t))
compiled(x)                          # [0., 2., 4., 6.]

Model.m.__code__ = replacement.__code__
model.m(x)                           # [0., 9., 18., 27.]   (eager sees the new code)
compiled(x)                          # [0., 2., 4., 6.]     <-- STALE, no warning
```

A fresh `torch.compile` after the assignment produces `[0., 9., 18., 27.]`, so the compilation
itself is correct — only the guard is missing.

## Reproduction results

`python repro.py --replays 2` (in this directory): 22 checks, 0 failures.

Triggers (4, all stale and silent, all with a matching signature so CPython accepts the
assignment):

| case | guarded? | result |
|---|---|---|
| `model.m(t)` — `Model.m.__code__ = other.__code__` | **no** | 2/2 replays stale |
| `derived.m(t)` — inherited method, `Base.m.__code__` replaced | **no** | 2/2 replays stale |
| `Model.cm(t)` — `classmethod`'s `__func__.__code__` replaced | **no** | 2/2 replays stale |
| `nn.Module` helper called from `forward` | **no** | 2/2 replays stale |

Controls (7, all correctly guarded — this is what makes the finding an internal inconsistency
rather than "Dynamo does not track code"):

| case | guarded? |
|---|---|
| module-level function `g.__code__ = other.__code__` | yes |
| plain function stored as a class attribute, `C.fn(t)` | yes |
| `staticmethod`, `C.sm(t)` | yes |
| a bound-method *object* stored in an attribute, `holder.bound(t)` | yes |
| `super().m(t)` | yes |
| method `__defaults__` replaced | yes |
| method closure cell contents replaced | yes |

Backends and modes (all reproduce):

| configuration | stale? |
|---|---|
| `backend="inductor"`, `dynamic=False` / `dynamic=True` | yes / yes |
| `backend="aot_eager"`, `dynamic=False` / `dynamic=True` | yes / yes |
| `backend="eager"` | **yes** — proves this is a Dynamo guard bug, not an Inductor bug |
| CUDA | yes |

## Mechanism

`TORCH_LOGS=guards` (`guard_dump.txt`) for the two access paths side by side:

```
# bound method -> no code guard
+- GuardManager: source=G['model'], type=<class '__main__.Model'>
|  +- TYPE_MATCH: ___check_type_id(G['model'], 106793671805840), type=<class '__main__.Model'>
|  +- GuardManager: source=G['model'].__dict__, accessed_by=GetGenericDictGuardAccessor
|  |  +- DICT_CONTAINS: not ___dict_contains('m', G['model'].__dict__)

# plain function attribute -> code guard present
+- GuardManager: source=G['Holder'], type=<class 'type'>
|  +- GuardManager: source=G['Holder'].fn, accessed_by=GetAttrGuardAccessor(fn), type=<class 'function'>
|  |  +- GuardManager: source=G['Holder'].fn.__code__, accessed_by=CodeGuardAccessor, type=<class 'code'>
```

The same function inlined through a different access path gets `fn.__code__` guarded; through the
bound-method path it gets nothing.  This is why the two paths disagree about the same mutation.

## Root cause

`UserFunctionVariable` adds a `CodeGuardAccessor` guard on the code object of the function it
inlines.  The bound-method path (`UserMethodVariable` / the `obj.m` attribute lookup) reuses the
same function object but does not add that guard — it only guards the instance's type and the
absence of an instance-`__dict__` shadow.  Note that this is *independent* of the missing
class-`__dict__` guard in `RONDO-PT-006`: guarding the class attribute value would still miss an
in-place `__code__` assignment, because the value (the function object) does not change.

## Suggested fix

In the bound-method inlining path, install the same `CodeGuardAccessor` guard that
`UserFunctionVariable` installs for a plain function, i.e. an `ID_MATCH` on
`source.__func__.__code__` for the resolved function.  A regression test is small: compile
`lambda t: obj.m(t)`, assign `type(obj).m.__code__ = other.__code__`, call the compiled callable
again and assert it matches eager.

## Related upstream work (novelty assessment)

* `RONDO-PT-005` (own finding) is about `types.MethodDescriptorType` closure cells, and
  `RONDO-PT-006` (own finding) about class-`__dict__` *values*; neither covers an in-place
  `__code__` assignment on a function obtained through a bound method.
* Tracker searches for `torch.compile __code__ method guard`, `monkeypatch __code__
  torch.compile`, `dynamo does not guard function code object` and `method __code__ change ignored
  torch.compile` return no matching report.

## Impact

Replacing a function's code object in place is a documented Python capability and a standard
monkeypatching technique (instrumentation/profiling shims, test doubles, runtime adapters).  After
`torch.compile`, such a patch is silently ignored: the compiled model keeps executing the old code
with no error, no warning, and no graph break.  The failure mode matches `RONDO-PT-006` and
`RONDO-PT-009` (silent stale semantics), but the missing guard is a different one, so all three
need separate fixes.

## Files

* `repro.py` — 4 triggers + 7 controls, replay counts, exit code = number of failures
* `guard_dump.txt` — `TORCH_LOGS=guards` output for both access paths side by side
* `collect_env.txt` — environment
