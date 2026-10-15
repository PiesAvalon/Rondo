# `[dynamo]` a `random.Random` instance method inside `torch.compile` returns a garbage constant — the graph input is a `float32` placeholder fed from a `float64` source

> **See also:** `findings/DYNAMO-GUARD-MODEL.md` for the guard-model findings.
> This one is not a missing guard: it is a **dtype mismatch in Dynamo's calling convention**.

**Component:** `torch.compile` (Dynamo) — `RandomValueSource` / `wrap_unspecialized_primitive`
**Severity:** **silent wrong results** — values of order `1e34` where eager returns a value in `[0, 1)`
**Status:** present in `torch 2.12.1+cu130`; the graph dump and the bit pattern below pin the mechanism
**Found with:** Rondo persistent-callable oracle (`state_matrix_fuzz.py` RNG axis) + source/guard audit

---

## Summary

Dynamo supports Python's `random` module by recording each call and replaying it at run time
(`torch/_dynamo/output_graph.py::_get_gen_rand_values_fn`). For a `random.Random` instance the
replayed value is turned into a graph input by
`variables/builder.py::wrap_unspecialized_primitive`:

```python
wrapped_value = torch.tensor(value)      # builder.py:2971 -- DEFAULT dtype => float32
...
proxy = self.tx.output.root_tracer.create_graph_input(
    ..., type(wrapped_value), example_value, source=self.get_source())
```

but the run-time reconstruction of that same source uses the *full-precision* helper
(`source.py:1094` → `guards.py:769` → `torch/__init__.py:3022`):

```python
class FloatTensorSource(ChainedSource):
    @property
    def _name_template(self):
        return "___as_tensor({0})"          # ___as_tensor = torch._as_tensor_fullprec
...
def _as_tensor_fullprec(t):
    ty = type(t)
    if ty is builtins.float:
        return torch.as_tensor(t, dtype=torch.float64, device="cpu")   # <= float64!
```

So the compiled graph declares a **float32** scalar placeholder and the wrapper calls it with a
**float64** tensor. Inductor then reads the float64 buffer as float32, which on little-endian
hardware is the *low 32 bits of the double's bit pattern reinterpreted as a float32*.

```python
import random, torch

rng = random.Random(0)
def f(t):
    return t * rng.random()

compiled = torch.compile(f)
compiled(torch.arange(4.0))
# tensor([-0.0, -8.9886e+34, -1.7977e+35, -2.6966e+35])   <-- garbage
f(torch.arange(4.0))
# tensor([0.0000, 0.8444, 1.6888, 2.5333])                 <-- eager
```

The compiled value is *bit-exactly* the reinterpretation of the first draw's low 32 bits:

```
double                     0.8444218515250481
IEEE-754 bits (little endian) 3f eb 05 80 f9 8a 7d be
low 4 bytes read as float32  -8.988591512864255e+34
compiled result              -8.988591512864255e+34
```

## Reproduction results

`python repro.py --replays 2` (in this directory): 12 checks, 0 failures, identical in three
separate processes.

Triggers (all `float`-valued `random.Random` methods; the check asserts bit-exact equality with
the low-32-bit reinterpretation of the method's first value):

| case | eager | compiled |
|---|---|---|
| `rng.random()` | `0.8444` | `-8.9886e+34` |
| `rng.uniform(0, 1)` | `0.8444` | `-8.9886e+34` |
| `rng.uniform(-1, 1)` | `0.6888` | `-1.1804e+31` |
| `torch.tensor(rng.random())` | `0.8444` | `-8.9886e+34` |

Controls:

| case | result |
|---|---|
| a plain captured Python float | correct |
| **`torch.set_default_dtype(torch.float64)` before compiling** | **correct** — the placeholder becomes float64 and the mismatch disappears |

That last control is the direct confirmation of the mechanism: the bug is exactly the difference
between `torch.tensor(value)` (default dtype) and `_as_tensor_fullprec(value)` (float64).

## Why this is not the known default-dtype cache issue

pytorch#184405 / PR #184500 are about the *cross-process Inductor cache key* not including the
default dtype. This is a different bug: it happens inside one process with a single (float32)
default dtype, and it is caused by the placeholder dtype and the run-time source dtype being
computed by two different helpers.

## Root cause

The `RandomValueSource` path is the one path that does **not** go through `FloatTensorSource`,
even though the comment right above the relevant code says so:

```python
# TODO: Switch RandomValueSource over to use this, this is more
# accurate
assert not isinstance(self.get_source(), RandomValueSource)
```

`wrap_unspecialized_primitive` computes the placeholder type from `torch.tensor(value)` (default
dtype, float32) while the reconstruction (`RandomValueSource` → `___as_tensor` →
`torch._as_tensor_fullprec`) always produces float64 for a Python `float`. The two disagree
whenever the default dtype is not float64 (and disagree worse for bfloat16/float16 defaults).

## Suggested fix

Make the placeholder and the run-time source agree. Either

* build the dummy with full precision, i.e. `wrapped_value = torch._as_tensor_fullprec(value)` in
  `wrap_unspecialized_primitive` (this is what the TODO above asks for), or
* route `RandomValueSource` through `FloatTensorSource` so the recorded random value is
  reconstructed with the same dtype the graph was traced with.

A regression test is small: `torch.compile(lambda t: t * random.Random(0).random())`, assert the
result is in `[0, 1)` and matches eager to float32 precision; the same with
`torch.set_default_dtype(torch.float64)`.

## Related upstream work (novelty assessment)

* The `# TODO: Switch RandomValueSource over to use this, this is more accurate` comment shows
  the path was known to be *less accurate*; the failure mode reported here is not reduced
  precision but a **completely wrong value** (~1e34), and no issue in the tracker describes it.
* Tracker searches for `random.Random torch.compile returns wrong value`, `torch.compile python
  random garbage`, `dynamo random_value placeholder dtype`,
  `random value float32 placeholder float64 dynamo` and `_as_tensor_fullprec random` return no
  matching report.
* `RONDO-PT-009` and `RONDO-PT-010` are also Dynamo findings but are missing *guards*; this one is
  a calling-convention dtype bug.

## Impact

`random.Random(seed)` instances are the standard way to get reproducible randomness, and mixing
Python randomness with tensor code inside a compiled function is common (data augmentation,
dropout masks computed in Python, synthetic label noise). With the default dtype, such a function
silently produces values of order `1e34` (or `±inf`/`NaN` for other bit patterns) instead of the
intended `[0, 1)` value — no error, no warning, no graph break.

## Files

* `repro.py` — 4 triggers + 2 controls, bit-exact checks, exit code = number of failures
* `graph_dump.txt` — the `f32[][]cpu` graph input, the run-time output, and the source snippets
  (`builder.py:2971`, `source.py:1094`, `guards.py:769`, `__init__.py:3022`) plus the bit pattern
* `collect_env.txt` — environment
