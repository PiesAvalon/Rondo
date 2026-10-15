# `jax.export` loses `weak_type` when a function is serialised, so the deployed function promotes dtypes differently from the in-process one

**Component:** `jax.export` — `serialize()` / `deserialize()` (JAX 0.10.2, jaxlib 0.10.2, flatbuffers 25.12.19)
**Severity:** silent dtype change in the deployment path — a value that is weakly typed in-process becomes
strongly typed after a round trip, so arithmetic that used to stay in float32 silently produces float64
(values are identical; the dtype of everything computed from them is not)
**Status:** present in JAX 0.10.2
**Found with:** the `jax.export` round-trip probe (`artifact/hunt-v2/tools_jax_export_probe.py`, 16 cases
covering elementwise maths, reductions, matmul, control flow, pytrees, five dtypes, complex arithmetic and
a PRNG); certified reproducer `repro.py` here, `python repro.py --replays 2` -> `RESULT: 0 failing check(s).`

---

## Summary

JAX marks values that derive from Python scalars as **weakly typed**, and its promotion rules let a weak
value lose to a narrower dtype (`weak float64 + float32 = float32`).  `jax.export` keeps that information
while the `Exported` object is alive, but it does not survive serialisation:

| object | `in_avals` weak | `out_avals` weak | `.call(0.0)` result weak | `+ jnp.float32(1)` |
|---|---|---|---|---|
| `jax.jit(f)` in process | `True` | `True` | `True` | `float32` |
| `jax.export(jit(f))(0.0)` | `True` | `True` | `True` | — |
| `jax.export.deserialize(serialized)` | **`False`** | **`False`** | **`False`** | **`float64`** |

The values are bit-identical in every case - `f(c) = c * 1.5 + 1.0` evaluated at `0.0` gives `1.0` before
and after - so this is not a numerical difference and no tolerance check would notice it.  What changes is
the *type*: a deployment wrapper that adds a float32 constant, or that jits the restored function together
with float32 data, gets float64 after export where it got float32 in process.

```python
import jax, jax.numpy as jnp
import jax.export as je

f = lambda c: c * 1.5 + 1.0
jitted = jax.jit(f)
exported = je.export(jitted)(0.0)
restored = je.deserialize(exported.serialize())

print(jax.eval_shape(f, 0.0).weak_type)          # True
print(exported.in_avals[0].weak_type)            # True   (export is not the problem)
print(restored.in_avals[0].weak_type)            # False  <-- lost
print((jitted(0.0) + jnp.float32(1.0)).dtype)    # float32
print((restored.call(0.0) + jnp.float32(1.0)).dtype)   # float64
```

## Scope

Any function with a weak-typed input or output is affected; functions whose inputs and outputs are all
strongly typed are **not** (the reproducer asserts that control):

* a scalar argument passed as a Python float - the input side for essentially any function that takes one;
* a weak-typed **output**: `lax.scan` with a Python-float carry, for instance
  (`lax.scan(lambda carry, e: (carry * 1.5 + e, carry), 0.0, t)` returns weakly typed `ys` in process and
  strongly typed after the round trip), which the reproducer also asserts;
* ordinary array-in/array-out functions such as `t * 2.0` are unaffected, which is why the probe's other
  15 cases - including `while_loop`, `cond`, `scan`, `matmul`, `complex`, `f16`, `bf16` and a PRNG - were
  clean.

## Mechanism

`Exported.in_avals` / `out_avals` / `.call()` preserve `weak_type`, so the information exists up to the
point of serialisation; `serialize()` writes StableHLO, which has no weak-type concept, and `deserialize()`
therefore reconstructs strongly typed avals.  (A substring search of the serialised module is *not*
evidence either way - generated names can contain the word "weak" without carrying a type encoding - so the
reproducer establishes the loss by the round trip itself, which is the check that matters.)

## Impact

Deployment code that derives dtypes from a function or relies on JAX's weak-type promotion silently moves
to the wider dtype: double the memory traffic, different numerics (a float64 accumulation where the
in-process version accumulated in float32), and in the worst case an `UnexpectedTracerError`-free but
wrong-dtype result that a test comparing values will not catch.  The exported artifact is the thing users
ship, so the discrepancy appears only in production, not in the tests they ran in process.

## Novelty assessment

Searches on the JAX tracker (`jax-ml/jax`) for `export weak_type`, `weak_type export`,
`deserialize weak type` and `export out_avals` return no report; the campaign's earlier JAX work
(`jit` cache behaviour, algebraic folds, precision, width and sign sweeps, the stats edge surface) never
touched `jax.export` at all, because `flatbuffers` was missing from this machine's JAX runtime.  This is
the first report from that surface.

## Possible resolutions

Preserve the weak-type flags where the format allows, or - if StableHLO genuinely cannot represent them -
record them in the `Exported` metadata that JAX controls and document that a deserialised function is
strongly typed.  Either way the reproducer's controls pin the behaviour that must not change: values
bit-identical, strong-typed functions unaffected.
