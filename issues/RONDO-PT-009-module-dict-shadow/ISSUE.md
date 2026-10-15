# `[dynamo]` a shadowing entry added to an `nn.Module`'s instance `__dict__` is not guarded — the compiled callable keeps using the registered submodule / parameter / buffer

> **See also:** `findings/DYNAMO-GUARD-MODEL.md` places this report alongside the other guard-model findings (`RONDO-PT-005/006/007/008`), lists everything that was verified as *correctly* guarded, and sketches a single fix.

**Component:** `torch.compile` (Dynamo) — guard emission for `nn.Module` attribute resolution
**Severity:** **silent wrong results** (not a crash), all backends including `backend="eager"`
**Status:** present in `torch 2.12.1+cu130`; guard dump below shows the missing guard
**Found with:** Rondo post-compile state-mutation oracle (`module_internals_fuzz.py`) + guard-tree audit

---

## Summary

Dynamo resolves `self.name` on an `nn.Module` at trace time by reading the instance
`__dict__` first and only falling back to the `_parameters` / `_buffers` / `_modules`
registries (`torch/_dynamo/variables/nn_module.py`, `getattr_helper`):

```python
subobj = None
if name in base_dict:                 # instance __dict__ wins, exactly like Python
    subobj = base_dict[name]
elif "_modules" in base_dict and name in base_dict["_modules"]: ...
elif "_parameters" in base_dict and name in base_dict["_parameters"]: ...
elif "_buffers" in base_dict and name in base_dict["_buffers"]: ...
```

The guards it then installs are rooted at `self._modules['name']` /
`self._parameters['name']` / `self._buffers['name']`. Nothing checks that the instance
`__dict__` still does **not** contain `name`. Python attribute lookup goes the other way —
an instance `__dict__` entry always wins over `__getattr__`, which is what serves the
registries — so adding such an entry changes what eager returns while every guard still
passes. The already-compiled callable silently keeps using the registered object.

```python
import torch, torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.sub.weight.copy_(torch.eye(4))

    def forward(self, t):
        return self.sub(t)

model = Wrapper()
x = torch.arange(4.0)
compiled = torch.compile(model)
compiled(x)                                   # [0., 1., 2., 3.]

replacement = nn.Linear(4, 4, bias=False)
with torch.no_grad():
    replacement.weight.fill_(5.0)
object.__setattr__(model, "sub", replacement) # plain Python: no nn.Module bookkeeping
model(x)                                      # [30., 30., 30., 30.]   (eager sees it)
compiled(x)                                   # [0., 1., 2., 3.]        <-- STALE
```

A fresh `torch.compile` after the write produces the correct `[30., 30., 30., 30.]`, so the
compilation itself is right — only the guard is missing.

## Why this is not "you corrupted the module"

Dynamo already implements exactly this guard, for exactly one attribute name.
`guard_to_detect_forward_monkeypatching` (`variables/nn_module.py:149`) documents the rule:

```python
# Users sometimes patch the forward method of a nn module instance to
# perform optimizations like quantization. Though this is not a good
# software practice, but python allows this and Dynamo needs to detect
# this patching.
...
if "forward" in mod.__dict__ and callable(mod.__dict__["forward"]):
    ... CLOSURE_MATCH on the patched forward ...
else:
    # Common case - check that the forward key is absent in mod __dict__
    install_guard(... DICT_NOT_CONTAINS('forward') ...)
```

The same argument ("python allows this and Dynamo needs to detect this patching") applies
verbatim to `sub`, `weight`, `scale`, `bias` — but no guard is emitted for them. The guard
tree below has `DICT_CONTAINS: not ___dict_contains('forward', ...)` for the submodule and
*no* corresponding entry for `'sub'` on `L['self'].__dict__`.

## Reproduction results

`python repro.py --replays 2` (in this directory): 24 checks, 0 failures.

Triggers (7, all stale and silent):

| case | guarded? | result |
|---|---|---|
| `module.__dict__["sub"] = other` (submodule) | **no** | 2/2 replays stale |
| `object.__setattr__(module, "sub", other)` | **no** | 2/2 replays stale |
| `vars(module).update({"sub": other})` | **no** | 2/2 replays stale |
| `outer.inner.__dict__["sub"] = other` (grandchild) | **no** | 2/2 replays stale |
| `linear.__dict__["weight"] = tensor` (parameter shadow) | **no** | 2/2 replays stale |
| `module.__dict__["scale"] = tensor` (buffer shadow) | **no** | 2/2 replays stale |
| submodule replaced by a **different class**, `__dict__` write | **no** | 2/2 replays stale (the `TYPE_MATCH` is on the unchanged `_modules["sub"]`) |

Controls (5, all correctly guarded):

| case | guarded? |
|---|---|
| `module.sub = other` (through `nn.Module.__setattr__`) | yes |
| `module._modules["sub"] = other` (registry write) | yes |
| `module._parameters["weight"] = Parameter` (registry write) | yes |
| `plain_object.__dict__["factor"] = 9.0` (non-module object) | yes |
| an attribute that genuinely lives in `__dict__` (`del _parameters["weight"]; m.weight = tensor`) | yes |

The last two controls matter: Dynamo *does* guard `__dict__` values that it resolves
directly, and it *does* guard the registries. The hole is only the resolution path itself —
Dynamo read `_modules["sub"]` at trace time and never checks that it would still read it.

Backends and modes (all reproduce):

| configuration | stale? |
|---|---|
| `torch.compile(model)`, `dynamic=False` | yes |
| `torch.compile(model)`, `dynamic=True` | yes |
| `backend="aot_eager"` | yes |
| `backend="eager"` | **yes** — proves this is a Dynamo guard bug, not an Inductor bug |
| `torch.compile(model.forward)` | yes |
| CUDA, default mode | yes |
| CUDA, `mode="reduce-overhead"` (cudagraphs) | yes |

## Mechanism

`TORCH_LOGS=guards` for the repro (`guard_dump.txt`), `L['self']` subtree, abridged:

```
+- GuardManager: source=L['self'], type=<class '__main__.Wrapper'>
|  +- TYPE_MATCH: ___check_type_id(L['self'], ...), type=<class '__main__.Wrapper'>
|  +- GuardManager: source=L['self'].__dict__, accessed_by=GetGenericDictGuardAccessor
|  |  +- GuardManager: source=L['self']._modules, ...
|  |  |  +- GuardManager: source=L['self']._modules['sub'], type=<class 'Linear'>
|  |  |  |  +- TYPE_MATCH: ___check_type_id(L['self']._modules['sub'], ...)
|  |  |  |  +- GuardManager: source=L['self']._modules['sub'].__dict__
|  |  |  |  |  +- DICT_CONTAINS: not ___dict_contains('forward', L['self']._modules['sub'].__dict__)
|  |  |  |  |  +- GuardManager: source=L['self']._modules['sub']._parameters
|  |  |  |  |  |  +- TENSOR_MATCH: check_tensor(L['self']._modules['sub']._parameters['weight'], ...)
|  |  +- GuardManager: source=L['self']._parameters, ...
```

`__dict__` is walked **only to reach `_modules`**; there is no
`DICT_CONTAINS: not ___dict_contains('sub', L['self'].__dict__)` — while `'forward'` gets
exactly that guard three levels down. Replacing `_modules["sub"]` fails the `TYPE_MATCH`;
adding a shadowing `__dict__["sub"]` fails nothing.

## Root cause

Trace-time attribute resolution and guard-time attribute resolution use different paths:
trace time reads `instance.__dict__[name]` first, guard time resolves the same attribute
through `_modules` / `_parameters` / `_buffers`. Whenever the two disagree the cached graph
is reused with the wrong object. The `forward` special case shows the correct pattern (guard
the *absence* of the key so that a later insertion invalidates the graph); it is simply
applied to one name only.

## Suggested fix

In `getattr_helper`, after resolving `name` through `_modules` / `_parameters` / `_buffers`,
also emit the same "key absent from instance `__dict__`" guard that
`guard_to_detect_forward_monkeypatching` emits for `forward`, i.e. a
`DICT_CONTAINS: not ___dict_contains(name, source.__dict__)` guard on the module source.
That is one guard per module attribute read, in the same spirit and with the same cost
profile as the existing `forward` guard. A regression test is small: compile
`model = torch.compile(wrapper)`, `object.__setattr__(wrapper, "sub", other)`, call again and
assert it matches eager.

## Related upstream work (novelty assessment)

* `#110048` / `#110803` are about the *opposite* direction (too many / spurious recompiles
  for module guards), and are closed.
* `#196127` (Prototype `isinstance` constant folding without guarding the attributes it
  reads) is the closest in kind, and is one of our own adjacent findings (`RONDO-PT-008`) —
  but it is about `isinstance`, not module attribute resolution.
* `guard_to_detect_forward_monkeypatching` documents that instance-level patching of module
  attributes is a supported pattern; only `forward` is covered.
* Tracker searches for `torch.compile module.__dict__`, `object.__setattr__ torch.compile`,
  `torch.compile wrong result after swapping submodule`, `module attribute shadowing
  torch.compile guard` and `_parameters _modules guard inconsistency` return no matching
  report.

## Impact

Mutating `module.__dict__` directly (or via `object.__setattr__` / `vars(...).update`) is how
several third-party libraries swap weights and submodules without triggering `nn.Module`'s
bookkeeping (`accelerate`-style device placement, quantization adapters, model-surgery
helpers). Because `backend="eager"` reproduces, any `torch.compile` user of such a library
gets silently stale submodules/parameters with no error, no warning and no graph break. The
failure mode is identical to `RONDO-PT-006` (class `__dict__` mutation), but on the instance
`__dict__` and through the `nn.Module` registry fast path, so the two need separate fixes.

## Files

* `repro.py` — 7 triggers + 5 controls, replay counts, exit code = number of failures
* `guard_dump.txt` — `TORCH_LOGS=guards` output showing the missing `DICT_CONTAINS` for `sub`
  next to the present one for `forward`
