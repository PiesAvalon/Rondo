# `tf.function` caches traces by Python argument equality, so `True`/`1`, `1.0`/`1`, `2.0`/`2`, `Fraction(1,2)`/`0.5`, `Decimal('0.5')`/`0.5` and `+0.0`/`-0.0` silently share one graph

**Component:** TensorFlow 2.21.0 (CPU), `tf.function` concrete-function cache (eager tracing)
**Severity:** silent wrong result with order dependence.  A second call whose Python argument only
*compares equal* to an earlier one reuses the first call's graph, so the function returns the first
branch's value while eager execution (and the documented retracing behaviour) returns the second
branch's.
**Status:** present in TensorFlow 2.21.0 (CPU)
**Found with:** a value-equal-but-bit-distinct cache-key probe across frameworks - the same class as
the wave-144 TF-011 (eager scalar cache behind `tf.constant`), but a different cache: this is the
`tf.function` trace cache, reached by Python arguments of a traced function.

---

## Summary

`tf.function` documents that it "builds a separate `tf.Graph` for each set of Python arguments that
it encounters" and that concrete functions are built "as the `PolymorphicFunction` is called with new
TensorFlow types or shapes, **or with new Python values as arguments**".  In 2.21.0 the cache key for
a pure Python argument is the argument's Python value through `==`/`hash`, which ignores the type and
the bit pattern.  Any two arguments that compare equal therefore share one cached trace:

| first call | second call | compiled result (2nd) | eager result (2nd) | concrete functions | traces |
|---|---|---|---|---|---|
| `f(x, True)` | `f(x, 1)` | `x + 1.0` | `x + 5.0` | 1 | 1 |
| `f(x, 1.0)` | `f(x, 1)` | `x + 2.0` | `x + 5.0` | 1 | 1 |
| `f(x, 2.0)` | `f(x, 2)` | `x + 2.0` | `x + 5.0` | 1 | 1 |
| `f(x, Fraction(1, 2))` | `f(x, 0.5)` | `x + 3.0` | `x + 2.0` | 1 | 1 |
| `f(x, Decimal('0.5'))` | `f(x, 0.5)` | `x + 4.0` | `x + 2.0` | 1 | 1 |
| `f(x, 0.0)` | `f(x, -0.0)` | `x * 1.0` | `x * -1.0` | 1 | 1 |

The order decides which branch survives and the effect is symmetric: with `1` first, `True` returns
the `int` graph.  `_list_all_concrete_functions()` and `experimental_get_tracing_count()` both report
a single trace, so TensorFlow has no idea the two calls are different computations:

```python
import tensorflow as tf

def branch(x, a):
    return x + 1.0 if isinstance(a, bool) else x + 5.0

fn = tf.function(branch)
float(fn(tf.constant([2.0], tf.float64), True)[0])   # 3.0
float(fn(tf.constant([2.0], tf.float64), 1)[0])      # 3.0   <-- eager gives 7.0
len(fn._list_all_concrete_functions())                # 1
```

The realistic shape of this is a flag that is sometimes a `bool` and sometimes an `int`
(`average=True` vs `average=1`), a numeric constant passed as `2` in one place and `2.0` in another,
or a sign-sensitive `0.0`/`-0.0`; in every case the second call silently adopts the first graph.

## Reproduction

`python repro.py --replays 2` → `RESULT: 0 failing replay(s).`
(45 checks per replay, 2 fresh processes.)

What the reproducer pins:

| dimension | result |
|---|---|
| pairs | five equal-comparing pairs (`True`/`1`, `1.0`/`1`, `2.0`/`2`, `Fraction(1,2)`/`0.5`, `Decimal('0.5')`/`0.5`): the first call is correct, the second differs from eager, and both `_list_all_concrete_functions()` and `experimental_get_tracing_count()` report **1** |
| order | the mirrored orders (`1` then `True`, `1` then `1.0`) are stale in the other direction |
| bit pattern | a sign-sensitive function with `0.0` then `-0.0` (and the reverse) returns the stale sign with one trace |
| controls | `2.0` then `3.0` retraces to **2** concrete functions and both results are correct; a `numpy.float64(2.0)` argument is **not** collapsed with the Python `2.0` (2 traces); tensor arguments follow dtype as documented (2 traces, correct values) |
| workaround | declaring the flag as a tensor (`input_signature=[TensorSpec, TensorSpec]`) restores distinct behaviour |

## Root cause (pointer)

The concrete-function cache is keyed by `(FunctionContext, FunctionType)`:

* `tensorflow/python/eager/polymorphic_function/polymorphic_function.py:494` creates
  `self._function_cache = function_cache.FunctionCache()`;
* `tensorflow/core/function/polymorphism/function_cache.py:30-49` keys an `OrderedDict` by
  `(context, FunctionType)`;
* `tensorflow/core/function/polymorphism/function_type.py:146-157` derives `Parameter.__eq__` /
  `__hash__` from `(name, kind, optional, type_constraint)`.

A pure Python argument becomes a value-based type constraint, so Python's equality decides the cache
entry, and `True == 1` (with equal hashes), `1.0 == 1`, `Fraction(1, 2) == 0.5`, `Decimal('0.5') ==
0.5` and `-0.0 == 0.0` all map to one entry.  A cache key that distinguished the argument's type and
its bit pattern (`type(a)` plus `struct.pack` of floats, or a `Literal` wrapper comparing both) would
restore the documented one-graph-per-Python-argument behaviour; a regression test is
`len(tf.function(f)._list_all_concrete_functions()) == 2` after `f(x, True)` and `f(x, 1)`.

## Novelty

* `gh search issues --repo tensorflow/tensorflow` for `tf.function True and 1`,
  `tf.function retracing equal arguments`, `tf.function negative zero`,
  `tf.function cache python argument`, `tf.function does not retrace` returns **nothing** relevant
  (the nearest hit is #46923, about a loss function behaving differently inside `tf.function`, which
  is unrelated).
* A web search surfaces the `tf.function` retracing documentation and unrelated retracing fixes, no
  report of equal-comparing arguments sharing a trace.
* This is distinct from the campaign's TF-011 (wave 144): TF-011 is the **eager scalar cache**
  behind `tf.constant`/`tf.fill`/`tf.Variable(0.0)`, where `tf.constant(-0.0)` returns the other
  zero; TF-012 is the **`tf.function` trace cache**, where the argument value selects which graph
  runs at all.  Different cache, different API, different observable (a whole traced graph vs a
  constant's sign).
