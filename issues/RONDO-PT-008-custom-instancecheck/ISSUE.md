# `[dynamo]` `isinstance()` on a constant bypasses a custom metaclass `__instancecheck__`

> **See also:** `findings/DYNAMO-GUARD-MODEL.md` places this report alongside the other three guard-model findings (`RONDO-PT-005/006/007/008`), lists everything that was verified as *correctly* guarded, and sketches a single fix.


**Component:** `torch.compile` (Dynamo) — `isinstance` constant folding
**Severity:** **silent wrong results** (not a crash)
**Status:** present in `torch 2.12.1+cu130`; the traced graph below bakes the folded answer
**Found with:** Rondo protocol-semantics oracle (`env_change_fuzz5.py`) + minimal reduction

---

## Summary

When the object in `isinstance(obj, cls)` is a *constant*, Dynamo folds the answer with its own
static type check and **never calls `cls`'s metaclass `__instancecheck__`**. A checker that
accepts values the default type check rejects is therefore silently ignored, and the compiled
function takes the wrong branch.

```python
import torch

class Meta(type):
    def __instancecheck__(cls, instance):
        return True                 # accept everything

class C(metaclass=Meta):
    pass

def f(t):
    return t + (1.0 if isinstance(1, C) else 100.0)

x = torch.arange(4, dtype=torch.float32)
f(x)                                   # [1., 2., 3., 4.]      (eager honours the hook)
torch.compile(f, dynamic=True)(x)      # [100., 101., 102., 103.]  <-- WRONG
```

The hook is not merely mis-evaluated — it is **never invoked**: instrumenting
`__instancecheck__` with a counter shows one call in eager and **zero** calls while compiling and
running the compiled function.

## Reproduction results

`python repro.py --replays 3` (in this directory):

| constant passed to `isinstance(const, C)` | eager | compiled |
|---|---|---|
| `object()` | 1.0 | **100.0** |
| `1` (int) | 1.0 | **100.0** |
| `1.0` (float) | 1.0 | **100.0** |
| `"a"` (str) | 1.0 | **100.0** |
| `()` (tuple) | 1.0 | **100.0** |
| `None` | 1.0 | **100.0** |
| `[]` (list) | 1.0 | **100.0** |
| `isinstance(C(), C)` with a hook returning `False` | 1.0 (CPython exact-type fast path) | **100.0** |
| control: default semantics on a plain class | consistent | consistent |
| control: a standard `ABCMeta` relation with no override | consistent | consistent |

16 failing checks in total (8 triggers × CPU + CUDA), each 3/3 replays.

## Mechanism

The traced graph contains no check at all — the branch was folded to the "no" arm:

```
def forward(self, s13: "Sym(s13)", L_t_: "f32[s13][1]cpu"):
    # File: .../graph_pt008.py:9 in f, code: return t + (1.0 if isinstance(1, C) else 100.0)
    add: "f32[s13][1]cpu" = l_t_ + 100.0;  l_t_ = None
    return (add,)
```

and the guards cover only the class object's identity:

```
+- GuardManager: source=G, type=<class 'dict'>
|  +- GuardManager: source=G['C'], type=<class '__main__.Meta'>
|  |  +- ID_MATCH: ___check_obj_id(G['C'], ...), type=<class '__main__.C'>
```

The hook counter confirms it: **0 invocations** during tracing plus the compiled call.

The trigger is specifically a *constant* object. When the same check is applied to a **user-class
instance** the hook *is* called and the result is correct:

| hook returns | object | eager | compiled | hook calls |
|---|---|---|---|---|
| `True` | `object()` (constant) | 1.0 | **100.0** | 0 |
| `True` | user-class instance | 1.0 | 1.0 | 1 |
| `True` | `C()` | 1.0 | 1.0 | 1 |
| `False` | `object()` (constant) | 100.0 | 100.0 | 0 |
| `False` | user-class instance | 100.0 | 100.0 | 1 |
| `False` | `C()` | 1.0 | **100.0** | 1 |

The last row is a second, related divergence: CPython's `isinstance` short-circuits when
`type(obj) is cls` (returning `True` without consulting the hook), whereas Dynamo calls the hook
and uses its `False`.

## Root cause

For a constant input, Dynamo's `isinstance` handling resolves the answer statically from the
declared type and does not consult the classinfo's metaclass hook. Constants are exactly the case
where the fold looks safe — `1` is not an instance of `C` by any normal reading — but a custom
`__instancecheck__` is precisely the mechanism a library uses to override that reading.

## Suggested fix

Do not fold `isinstance`/`issubclass` when the classinfo's metaclass defines `__instancecheck__` /
`__subclasscheck__` (directly or inherited); graph break instead, which is the policy upstream
already adopted for the tensor case. Alternatively emit a guard on the hook and call it at runtime.

A regression test is small: compile a function containing `isinstance(1, C)` where
`type(C).__instancecheck__` returns `True`, and assert the compiled result matches eager.

## Related upstream work (novelty assessment)

The same defect for **tensor** values is known and was fixed:

* **#122093** (closed) "Bug in `torch.compile` with standard type checking tools beartype" —
  `beartype`/`jaxtyping` annotations on tensors, i.e. `isinstance(tensor, ...)`;
* **#186491** (closed) "Graph break tensor isinstance with custom instancecheck" — the fix, whose
  description is explicit about the scope: *"Do not constant-fold arbitrary custom
  `__instancecheck__` hooks **on tensor values** ... graph break for **tensor** isinstance checks
  that use custom classinfo hooks. The supported local tensor cases still use the existing fast
  path: tensor dtype pseudo-types, `nn.Parameter`, `nn.Buffer`, and standard `ABCMeta` checks."*

This report is the **constant / non-tensor** path, which that fix does not touch; the trigger here
is `isinstance(1, C)`, `isinstance(None, C)`, and so on. (In this build the tensor path is broken
too, i.e. the build predates #186491; the point of this report is that #186491's scope is tensors,
so the constant path needs the same treatment.)

The `runtime_checkable` Protocol variants are tracked separately (#196127, its fix PR #196359, and
#196124). Targeted searches for `custom instancecheck not supported dynamo`,
`dynamo __subclasscheck__ metaclass`, `isinstance ignores __instancecheck__ torch.compile` and
`torch.compile metaclass __instancecheck__` return no report for constant non-tensor values.

## Impact

Custom `__instancecheck__` hooks are how validation/annotation libraries implement structural
checks (`beartype`, `jaxtyping`, schema validators, `Optional`-style predicates). Applying such a
check to a constant — a config value, a string tag, `None` — is silently compiled to `False`, so
the compiled function takes the branch eager never takes, with no error, no warning and no graph
break. As with `RONDO-PT-005`/`006`/`007`, the failure is a missing guard around semantics Dynamo
assumed were static.

## Files

* `repro.py` — CPU + CUDA reproducer (8 triggers + 2 controls, replay counts, hook-call counter)
* `graph_dump.txt` — traced graph showing the folded `+ 100.0` branch
* `collect_env.txt` — environment

## Anticipated objections and responses

* **"`isinstance` on a constant is folded at trace time by design."**  Folding is fine when the
  answer cannot change; here it *skips user code* (`__instancecheck__`) that eager runs, and the
  result does change when the metaclass is written to depend on state.  The repro contains 8
  triggers plus controls where the fold is correct, so the boundary is explicit.
* **"The user should not rely on `__instancecheck__` with `torch.compile`."**  Nothing documents
  that restriction, and the failure is silent (wrong branch taken, not an error).
* **"This is the same as PT-007."**  PT-007 is the ABC *registry*; this is a user-defined
  `__instancecheck__` that is never invoked at all.  Different mechanism, different fix site.
