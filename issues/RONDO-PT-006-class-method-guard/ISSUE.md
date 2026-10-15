# `[dynamo]` class-level method / special-method replacement after `torch.compile` is not guarded — the compiled callable silently keeps calling the old implementation

> **See also:** `findings/DYNAMO-GUARD-MODEL.md` places this report alongside the other three guard-model findings (`RONDO-PT-005/006/007/008`), lists everything that was verified as *correctly* guarded, and sketches a single fix.


**Component:** `torch.compile` (Dynamo) — guard emission for method dispatch through an instance
**Severity:** **silent wrong results** (not a crash)
**Status:** present in `torch 2.12.1+cu130`; guard dump below shows the missing guard
**Found with:** Rondo environment-change oracle (`env_change_fuzz.py`) + minimal reduction

---

## Summary

Dynamo's guard set for a captured object covers the object's **type** and its **instance
`__dict__`**, but never the **class `__dict__`** — even though the instance-`__dict__` guard
exists precisely to catch instance-level shadowing. Because Python resolves methods, operators
and *special methods* (`__getitem__`, `__len__`, ...) on the type, replacing any of them on the
class after compilation passes every guard, and the already-compiled callable silently keeps
calling the old implementation.

```python
import torch

class Scaler:
    def scale(self, x):
        return x * 2.0

s = Scaler()

def f(x):
    return s.scale(x)

x = torch.arange(4, dtype=torch.float32)
compiled = torch.compile(f, dynamic=True)
compiled(x)                                   # [0., 2., 4., 6.]

Scaler.scale = lambda self, y: y * 100.0      # patch the class method
f(x)                                          # [0., 100., 200., 300.]  (eager sees it)
compiled(x)                                   # [0., 2., 4., 6.]       <-- STALE
```

A fresh `torch.compile` after the patch produces the correct `[0., 100., 200., 300.]`, so the
compilation itself is right — only the guard is missing.

## Reproduction results

`python repro.py --replays 3` (in this directory):

| case | guarded? | result |
|---|---|---|
| `s.scale(x)` (captured instance, bound-method call) | **no** | 3/3 replays stale, CPU + CUDA; fresh compile correct |
| `nn.Module.self.helper(x)` (method called from `forward`) | **no** | 3/3 replays stale, CPU + CUDA |
| `t * container[0]` (`__getitem__` replaced on the class) | **no** | 3/3 replays stale, CPU + CUDA |
| `t + float(len(container))` (`__len__` replaced on the class) | **no** | 3/3 replays stale, CPU + CUDA |
| control: `Scaler.scale(s, x)` (explicit unbound call) | yes | correct |
| control: `getattr(type(s), "scale")(s, x)` (explicit class-attribute access) | yes | correct |
| control: instance attribute shadowing `s.scale = ...` | yes | correct |
| control: rebinding a module-level function `g` | yes | correct |

| `t * container[0]` (`__getitem__` replaced on the class) | **no** | 3/3 replays stale, CPU + CUDA |
| `t + float(len(container))` (`__len__` replaced on the class) | **no** | 3/3 replays stale, CPU + CUDA |
| `t * CG[int]` (`__class_getitem__` replaced on the class) | **no** | 3/3 replays stale, CPU + CUDA |
| `t * ViaMetaSubscript[int]` (`__getitem__` replaced on the **metaclass**) | **no** | 3/3 replays stale, CPU + CUDA |
| a **submodule class's `forward`** replaced (`self.sub(x)`) | **no** | 3/3 replays stale, CPU + CUDA |
| `nn.Module.__call__` overridden on a subclass and then replaced | **no** | 3/3 replays stale, CPU + CUDA |

14 failing checks in total (7 triggers × CPU + CUDA), deterministic across 3 replays each.

The submodule case is the most realistic form of the bug: compiling a parent model and then
patching the `forward` of a library submodule's class (a common way to adapt third-party
modules) is silently ignored by the already-compiled parent.

`object_model_fuzz.py` and `type_system_fuzz.py` extend the search over further object-model and
type-system constructs. Three more triggers belong to this same root cause — they are all method
lookup through the class `__dict__` and are all unguarded:

| case | guarded? |
|---|---|
| a `__slots__` class's method replaced (no instance `__dict__` to guard at all) | **no** |
| a `classmethod` replaced | **no** |
| a plain function assigned to the class after construction (bound via `types.MethodType`) | **no** |
| `CG[int]` — `__class_getitem__` replaced on the class | **no** |

Everything below is **correctly guarded**, so the gap is specifically method/special-method
resolution through the class:

| case | guarded? |
|---|---|
| `nn.Module.forward` itself replaced | yes |
| functor `__call__` replaced | yes |
| user class `__add__` replaced | yes |
| `property` getter state changed | yes |
| base-class method replaced, reached through `super()` | yes |
| `obj.__class__` reassigned to a compatible class | yes |
| nested class attribute replaced | yes |
| custom descriptor `__get__` replaced | yes |
| `staticmethod` replaced | yes |
| `__getattr__` defined on the class, driven by class state | yes |
| metaclass attribute changed | yes |
| `functools.singledispatch` registration added | yes |
| `nn.Module` forward hook registered after compilation | yes |
| `property` getter replaced | yes |
| `__index__` object used as a tensor index, state changed | yes |
| `__contains__` state changed | yes |
| `__iter__` state changed | yes |
| `__getattribute__` state changed | yes |
| a functor's `__call__` class state changed | yes |
| a `dict` subclass's `__missing__` state changed | yes |
| a module-level `__getattr__` (PEP 562) state changed | yes |
| a `__slots__` descriptor replaced by a property | yes |
| a plain class attribute read | yes |
| a **submodule replaced by an instance of a different class** | yes (instance identity is guarded) |
| a forward **pre**-hook registered after compilation | yes |
| a full **backward** hook registered after compilation | yes |
| `nn.Dropout` switched from train to `eval()` | yes |
| `BatchNorm1d` running statistics written in place in eval mode | yes |

Both compile entry points behave identically: `torch.compile(model)` and
`torch.compile(model.forward)` both return the stale result after the class method is replaced,
while `model.eval()` (an *instance* attribute change) is correctly guarded through either entry
point. So this is not an artefact of how the module is handed to the compiler.

The second trigger is the practically important one: `self.helper(x)` inside `nn.Module.forward`
is ordinary code, and patching a helper method on a module class is a normal thing to do
(test doubles, adapter/LoRA-style patching, monkeypatching for debugging).

## Mechanism

`TORCH_LOGS=guards` for the example (`guard_dump.txt`):

```
+- GuardManager: source=G['s'], accessed_by=DictGetItemGuardAccessor('s'),
|                type=<class '__main__.Scaler'>
|  +- TYPE_MATCH: ___check_type_id(G['s'], ...), type=<class '__main__.Scaler'>
|  +- GuardManager: source=G['s'].__dict__, accessed_by=GetGenericDictGuardAccessor
|  |  +- DICT_CONTAINS: not ___dict_contains('scale', G['s'].__dict__)
```

The guard set says: "`s` is still a `Scaler`" and "`s` does not have its own `scale`
attribute". Both remain true when `Scaler.scale` is replaced, so the cached graph is reused.
There is **no guard on `Scaler.__dict__['scale']`** (or on the resolved function object),
which is exactly what the instance-shadowing guard was presumably meant to protect against.

The identical pattern appears for a user container's special methods
(`guard_dump_getitem.txt`):

```
+- GuardManager: source=G['c'] (type Container)
|  +- TYPE_MATCH: type(G['c']) is Container
|  +- GuardManager: source=G['c'].__dict__
|  |  +- DICT_CONTAINS: not '__getitem__' in G['c'].__dict__
|  +- GuardManager: source=G['c']._v (list)
```

## Root cause

Dynamo models a captured instance as `type` + instance `__dict__`. Method lookup through an
instance (`obj.method`) is resolved at trace time and inlined, but the *resolution target* is
not guarded — only the instance-level shadowing is. Class-level mutation is therefore
invisible.

## Suggested fix

When a method or special method is resolved through a type and inlined, emit a guard on the
resolved function, e.g. `FUNCTION_MATCH` / `ID_MATCH` on `type(obj).__dict__[name]` (or a
`TYPE_ATTRIBUTE_MATCH` style guard on the class attribute), exactly as is already done for
module globals holding a function. Alternatively guard the class `__dict__` version counter —
one guard on the type's `__dict__` would cover all four triggers at once. A regression test is
small: compile `lambda x: obj.method(x)`, replace `cls.method`, call the compiled callable again
and assert it matches eager; the same for `__getitem__`/`__len__`.

## Related upstream work (novelty assessment)

Dynamo's guard/function-reconstruction area is under very active development, with several
open PRs that touch adjacent ground — `#196473` ("Carry a bound method's function and receiver
explicitly when getattr cannot resolve it"), `#195494`, `#196476`, `#196474`, `#196481`,
`#195353`, and `#196127` ("isinstance() with a runtime_checkable Protocol constant-folds the
result without guarding the attributes it reads"). None of them describes this defect:

* `#196127` is the closest in kind — a guard is missing for attributes read during a
  constant-folding decision — but it is about `isinstance` and Protocol attributes, not method
  dispatch.
* Targeted tracker searches for `monkeypatch class method torch.compile wrong result`,
  `torch.compile wrong after monkeypatching nn.Module method`, `guard class __dict__ dynamo`
  and `dynamo instance type guard does not cover class methods` return no matching report.

## Impact

Patching a method is a standard technique — test doubles, runtime adapters, debugging, and
library hooks (`torch.nn.Module` helpers in particular). After `torch.compile`, such a patch is
silently ignored, so the compiled model keeps running the old code path with no error, no
warning and no graph break. This is the same failure mode as `RONDO-PT-005` (an unguarded
captured value) but a different construct: there the value was a `method_descriptor` object,
here it is the result of method resolution against a mutable class.

## Files

* `repro.py` — CPU + CUDA reproducer with triggers, controls, replay counts and a fresh-compile check
* `guard_dump.txt` — `TORCH_LOGS=guards` output showing the type + instance-dict guards only
* `guard_dump_getitem.txt` — the same for a user container's `__getitem__`
* `object_model_fuzz.py` — 10 further object-model constructs
* `type_system_fuzz.py` — 10 type-system / dispatch constructs (found three extra triggers)
* `protocol_value_fuzz.py` — 7 value-producing protocol methods (found the `__class_getitem__` trigger)
* `runtime_lookup_fuzz.py` — 5 runtime-computed lookups (found the metaclass-`__getitem__` trigger)
* `module_graph_fuzz.py` — 7 module-graph / state cases (found the submodule-`forward` trigger)
* `collect_env.txt` — environment

## Additional instance found later (round 67)

Replacing `__getattr__` on the class after compilation is also silently ignored, which is the same root cause (the class `__dict__` is not guarded):

```python
class C:
    def __init__(self): self._v = 3.0
    def __getattr__(self, name):
        if name == 'w': return self._v * 1.0
        raise AttributeError(name)

c = C()
f = torch.compile(lambda t: t * c.w)
f(torch.tensor([2.0]))            # 6.0
C.__getattr__ = lambda self, name: self._v * 7.0 if name == 'w' else (_ for _ in ()).throw(AttributeError(name))
f(torch.tensor([2.0]))            # 6.0  (stale)
torch.compile(lambda t: t * c.w)(torch.tensor([2.0]))   # 42.0  (fresh)
t * c.w                            # 42.0  (eager)
```

Seven other new access patterns were checked in the same probe and are handled correctly:
`functools.partial.keywords` mutated in place, a class `property` replaced, a `__slots__` attribute changed, a metaclass attribute changed, a nested attribute whose inner object is replaced, and a class-level tensor attribute changed.  (`functools.partial.func` is read-only, so only the keywords dict can be mutated in place.)
