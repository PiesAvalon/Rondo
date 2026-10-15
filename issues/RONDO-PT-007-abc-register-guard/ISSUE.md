# `[dynamo]` `isinstance` / `issubclass` against an ABC is constant-folded without guarding the class relation, so `ABCMeta.register()` after compilation is silently ignored

> **See also:** `findings/DYNAMO-GUARD-MODEL.md` places this report alongside the other three guard-model findings (`RONDO-PT-005/006/007/008`), lists everything that was verified as *correctly* guarded, and sketches a single fix.


**Component:** `torch.compile` (Dynamo) — constant folding of `isinstance` / `issubclass`
**Severity:** **silent wrong results** (not a crash)
**Status:** present in `torch 2.12.1+cu130`; the traced graph below bakes the folded answer
**Found with:** Rondo environment-change oracle (`env_change_fuzz3.py`) + minimal reduction

---

## Summary

Dynamo evaluates `isinstance(obj, SomeABC)` / `issubclass(SomeClass, SomeABC)` at trace time and
bakes the answer into the graph. The guards it installs cover the *objects involved* (the
instance's type, the class object's identity) but not the **class relation** that the answer
depends on. `ABCMeta.register()` mutates exactly that relation, so registering a virtual
subclass after compilation passes every guard and the compiled callable keeps returning the old
answer.

```python
import abc
import torch

class Base(abc.ABC):
    pass

class Impl:
    pass

o = Impl()

def f(t):
    return t + (1.0 if isinstance(o, Base) else 100.0)

x = torch.arange(4, dtype=torch.float32)
compiled = torch.compile(f, dynamic=True)
compiled(x)                 # [100., 101., 102., 103.]

Base.register(Impl)         # Impl is now a virtual subclass of Base
f(x)                        # [1., 2., 3., 4.]      (eager sees the registration)
compiled(x)                 # [100., 101., 102., 103.]  <-- STALE
```

A fresh `torch.compile` after the registration returns the correct `[1., 2., 3., 4.]`.

The same holds when the relation is mutated through the class hierarchy instead of the ABC
registry — assigning `type.__bases__` on a heap type:

```python
class BaseA: pass
class BaseB: pass
class Sub(BaseA): pass

def f(t):
    return t + (1.0 if isinstance(Sub(), BaseA) else 100.0)

compiled = torch.compile(f, dynamic=True)
compiled(x)                 # [1., 2., 3., 4.]
Sub.__bases__ = (BaseB,)    # relation changed
compiled(x)                 # [1., 2., 3., 4.]  <-- STALE, eager gives [100., ...]
```

`guard_dump_bases.txt` shows the class object is guarded by identity (`type(G['Sub']) is type`)
and **nothing reads `Sub.__bases__`**.

## Reproduction results

`python repro.py --replays 3` (in this directory). Each case builds its own ABC, because the ABC
registry and the `isinstance` caches are process-global state:

| case | result |
|---|---|
| `isinstance(o, Base)` + `Base.register(Impl)` after compile | **FAIL** 3/3 stale, CPU + CUDA; fresh compile correct |
| `isinstance(Sub(), BaseA)` + `Sub.__bases__ = (BaseB,)` after compile | **FAIL** 3/3 stale, CPU + CUDA; fresh compile correct |
| `issubclass(Impl, Base)` + `Base.register(Impl)` after compile | **FAIL** 3/3 stale, CPU + CUDA; fresh compile correct |
| control: no mutation after compile | correct |
| control: registered **before** compile | correct (`[1., 2., 3., 4.]`) |

6 failing checks in total (3 triggers × CPU + CUDA).

## Mechanism

The traced graph (`graph_dump.txt`) does not contain any check at all — the branch was folded:

```
def forward(self, s13: "Sym(s13)", L_t_: "f32[s13][1]cpu"):
    # File: .../graph_pt007.py:10 in f, code: return t + (1.0 if isinstance(o, Base) else 100.0)
    add: "f32[s13][1]cpu" = l_t_ + 100.0;  l_t_ = None
    return (add,)
```

and `TORCH_LOGS=guards` shows the guards it did install:

```
+- GuardManager: source=G['o'], type=<class '__main__.Impl'>
|  +- TYPE_MATCH: ___check_type_id(G['o'], ...), type=<class '__main__.Impl'>
+- GuardManager: source=G['Base'], type=<class 'abc.ABCMeta'>
```

So the guards assert "`o` is still an `Impl`" and "the global `Base` is still the same object".
Both remain true after `Base.register(Impl)`, because registering a virtual subclass changes
`Base`'s internal registry, not `Base`'s identity nor `o`'s type. There is **no guard on the
relation between `Impl` and `Base`**.

## Root cause

Dynamo's `isinstance` / `issubclass` handling treats the answer for non-torch classes as a
compile-time constant. For a normal class hierarchy that is sound, because a class's bases cannot
change after creation — but ABCs are extensible at runtime through `ABCMeta.register()`
(virtual subclasses), which is precisely the escape hatch from that assumption. Nothing in the
guard set covers the registry.

## Suggested fix

Either

* stop folding when the second argument is an `abc.ABCMeta` instance whose hierarchy can still
  be extended (emit a runtime `isinstance` call / graph break instead), or
* install a guard on the relation the fold read — e.g. a guard on `Base._abc_registry`
  (identity/version) and on `_abc_negative_cache`, in the same spirit as the existing
  `TYPE_MATCH` guards — or, more generally, do not fold when the relation can still change
  (`ABCMeta.register` for ABCs, `type.__bases__` assignment for ordinary classes).

The same reasoning applies to `issubclass`, and to `typing.runtime_checkable` Protocols, which are
covered separately upstream (see below).

A regression test is small: compile a function containing `isinstance(x, SomeABC)`, call
`SomeABC.register(SomeClass)`, call the compiled function again and assert it matches eager.

## Related upstream work (novelty assessment)

`isinstance` constant folding is an area with active upstream work — all of it about
`runtime_checkable` **Protocols**, which read an object's *attributes*:

* **#196127** (open) "torch.compile: isinstance() with a runtime_checkable Protocol
  constant-folds the result without guarding the attributes it reads" — includes an AI-assisted
  analysis of the single-class path;
* **#196359** (open) "[dynamo] Guard attributes read by isinstance() on runtime_checkable
  Protocols" — the fix PR for #196127;
* **#196124** (open) "[dynamo] Fix isinstance() with unions/tuples of runtime_checkable
  Protocols".

This report is a different mechanism: nothing is read from the object, the answer depends on the
**ABC virtual-subclass registry**, and it affects `issubclass` as well as `isinstance`. Targeted
searches for `abc register isinstance torch.compile`, `ABCMeta register compile`,
`virtual subclass register dynamo guard` return no matching report. A general fix that stops
folding `isinstance` for extensible classes would cover both this and (partly) the Protocol cases,
so a maintainer may prefer to fold this into that work; it is filed separately because the trigger,
the mutated state and the guard that is missing are all different.

## Impact

Virtual subclass registration is the documented way to make an existing class satisfy an ABC
(`collections.abc` registrations, plugin frameworks, structural-typing shims, test doubles).
Doing it after `torch.compile` is silently ignored, so a compiled model keeps taking the old
branch with no error, no warning and no graph break. As with `RONDO-PT-005` and `RONDO-PT-006`
this is a missing-guard bug, but the state that is unguarded is the class *relation* rather than
a value or an attribute.

## Files

* `repro.py` — CPU + CUDA reproducer (2 triggers + 2 controls, 3 replays each, fresh-compile check)
* `graph_dump.txt` — traced graph showing the folded `+ 100.0` branch
* `guard_dump_bases.txt` — guards for the `__bases__` variant (class identity only)
* `collect_env.txt` — environment

## Anticipated objections and responses

* **"Constant-folding `isinstance` is intended; a guard would be too expensive."**  The guard model
  this file relies on is documented as covering ambient global state, and the registry is exactly
  that: `ABCMeta.register` mutates process-global state that eager `isinstance` observes.  The
  related change to `type.__bases__` (also shown in the repro) is a plain type mutation.  A cheap
  guard on the class relation (or on the registry version) is enough; the draft does not require
  re-checking the registry on every call.
* **"This is a documented limitation."**  No entry in the Dynamo documentation lists ABC registries
  as unsupported; the limitation list in this repository's own `DYNAMO-GUARD-MODEL.md` shows the
  boundary the six guard drafts sit on.
* **"The repro needs an unusual workflow."**  Registering an ABC after the first compiled call is
  ordinary plugin/monkey-patch code; the draft includes five triggers and their controls so the
  maintainer can pick the smallest one.
