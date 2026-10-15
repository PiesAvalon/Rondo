# Operations that take a no-write path return stale memory: complex division by zero and `ops.unique` of an empty tensor

**Component:** MindSpore 2.9.0, CPU, PYNATIVE and GRAPH (`ms.jit`): `ops.div` / `/` / `mint.div` /
`mint.divide` on complex types, and `ops.unique` on an empty tensor
**Severity:** silent wrong value **and** cross-operator data leakage — division by zero never
writes its output, so the caller reads whatever the reused buffer already held: `0j` in a fresh
process, and the result of a previous operation (division, multiply, add, power, even a *real*
division) afterwards, including stale bytes from a tensor of a different shape or dtype.  The
returned value therefore depends on execution history, and in a fresh process a division by zero
silently yields zero.
**Status:** present in MindSpore 2.9.0 (CPU, PYNATIVE and GRAPH); NumPy, PyTorch, JAX and
TensorFlow all return the C99 Annex G values (`inf`/`nan`)
**Found with:** the wave-164 complex binary grid (`artifact/hunt-v2/results/w164_*`) for the
division case and the wave-172 indexing grid (`artifact/hunt-v2/results/w172_*`) for the `unique`
case, triaged by Rondo's `check_consensus` oracle with `cmath`/NumPy as arbiter; certified
reproducer `repro.py` in this directory (72 self-checking assertions,
`--replays 2` -> `RESULT: 0 failing replay(s).`)

---

## Summary

`ops.div` on complex tensors with an exactly-zero divisor returns **whatever the output buffer
already contains** - it produces no value of its own.  In a fresh process the buffer is zeroed, so
every division by zero returns `0j`:

```python
import numpy as np, mindspore as ms
from mindspore import ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
def T(z): return ms.Tensor(np.asarray(z, np.complex128), ms.complex128)

print(ops.div(T(1+0j), T(0+0j)).asnumpy())     # 0j          <- wrong, C99/NumPy: inf+nanj
print(ops.div(T(0+0j), T(0+0j)).asnumpy())     # 0j          <- wrong, C99/NumPy: nan+nanj
print(ops.div(T(2+3j), T(0+0j)).asnumpy())     # 0j          <- wrong, C99/NumPy: inf+infj
```

Once another operation has written the shared buffer, the same call returns **that operation's
result**:

```python
ops.div(T(2+2j), T(1+0j))                      # 2+2j
print(ops.div(T(1+0j), T(0+0j)).asnumpy())     # 2+2j        <- the previous quotient, not inf+nanj

ops.mul(T(2+2j), T(3+0j))                      # 6+6j
print(ops.div(T(1+0j), T(0+0j)).asnumpy())     # 6+6j        <- the previous product

ops.div(ms.Tensor(2.0, ms.float64), ms.Tensor(4.0, ms.float64))   # 0.5  (real)
print(ops.div(T(1+0j), T(0+0j)).asnumpy())     # 0.5+0j      <- a *real* result reused as complex
```

The stale bytes can come from a tensor of a different shape: after a length-4 complex division, a
scalar `1/0` reads one of that vector's elements (`0.5+0.5j`) when the allocator recycles the block,
and zeros otherwise.  `ms.jit` and `mint` share the behaviour.  Real-dtype division by zero is
correct (`1.0/0.0 = inf`, `0.0/0.0 = nan`, `-1.0/0.0 = -inf`), so the division half is specific to
the complex division kernel.

**The same no-write pattern makes `ops.unique` of an empty tensor return the previous call's
output.**  `ops.unique(ms.Tensor([], ms.float64))` should return an empty `output`; instead it
returns a non-empty tensor whose *length tracks the previous `ops.unique` call* (its bytes come from
the recycled block): after `unique([3, 1, 3, 2])` (output size 3) the empty call returns a size-3
tensor, and after `unique([-0., 0., -0.])` (output size 1) it returns a size-1 tensor, while the
`indices` output is correctly empty and a fresh process returns the zero-filled `[]`.  NumPy,
PyTorch, JAX and TensorFlow return the empty array.

## Expected vs actual

`x / (0+0j)` for the C99 Annex G result, and what MindSpore returns in a fresh process:

| `x` | NumPy / PyTorch / JAX / TensorFlow | MindSpore 2.9.0 (fresh) |
|---|---|---|
| `1+0j` | `inf+nanj` | `0j` |
| `-1+0j` | `-inf+nanj` | `0j` |
| `0+1j` | `nan+infj` | `0j` |
| `0-1j` | `nan-infj` | `0j` |
| `1+1j` | `inf+infj` | `0j` |
| `2+3j` | `inf+infj` | `0j` |
| `-2+3j` | `-inf+infj` | `0j` |
| `0+0j` | `nan+nanj` | `0j` |
| `1e308+0j` | `inf+nanj` | `0j` |
| `inf+0j` | `inf+nanj` | `0j` |
| `inf+infj` | `inf+infj` | `0j` |

## Root cause

The complex division kernel does not store a result when the divisor is exactly zero (an early
return, a mask/select that leaves the destination untouched, or a skipped store), and the output
tensor comes from MindSpore's reused memory pool, so the caller observes the previous writer's
bytes.  The evidence for "never writes":
* the value tracks the **immediately preceding** complex operation of any kind (`div`, `mul`, `add`,
  `pow`) rather than any mathematical function of the operands;
* a preceding **real** division is reused as the complex real part (`0.5+0j`);
* the value is `0j` in a fresh process and stable per process (two independent replays agree);
* `mint.div`, the `/` operator and `ms.jit` (GRAPH) behave identically;
* the real-dtype kernel writes correctly, so only the complex path skips the store.

## Suggested fix

Compute the zero-divisor branch explicitly per C99 Annex G (or use the real kernel componentwise)
instead of returning early without a store, and make the kernel always write its destination.  A
regression test should assert `isinf`/`isnan` for `ops.div(x, 0)` and, as a second line of defence,
that the result does not change when an unrelated complex operation runs first.

## Impact

* Silent wrong numerics: `x/0` returns `0` (a finite value) in any fresh process, so a singular
  division is not detectable by `isinf`/`isnan` checks and silently contributes zeros.
* History-dependent results: the same expression returns different values depending on previous
  computations in the process - including values from tensors of a different shape or dtype, i.e.
  the output buffer leaks data from unrelated computations.
* Both `complex64` and `complex128`, PYNATIVE and GRAPH, `ops.div`, `/`, `mint.div`, `mint.divide`.

## Controls (all in `repro.py`)

* Fresh process: `x/0 = 0j` for eleven numerators, none of which is `inf`/`nan`; NumPy's values are
  asserted alongside.
* Nonzero complex divisions (`(1+1j)/(1+2j)`, `(2+3j)/(-1+2j)`, the subnormal-exponent pair) match
  NumPy exactly, at both widths, in PYNATIVE and in `ms.jit`.
* Real division by zero is correct (`inf`, `nan`, `-inf`); real `2.0/4.0 = 0.5`.
* A left-over from section 1 is impossible: the fresh assertions run first, before any successful
  complex operation.

## Novelty

Searched the MindSpore tracker for `complex division by zero`, `div complex zero`, `ops.div complex
nan`, `complex divide nan`, `division by zero complex`, `stale result`, `uninitialized output`,
`divide by zero returns`, `complex div zero stale`: the only hit is mindspore#386 ("`mnp.divide`
returns `-Inf` for `NaN` divided by zero", open), which is a real-dtype `mnp` value-convention issue
and unrelated to buffer reuse.  No report of complex division by zero returning zero or the previous
operation's output exists.  The campaign's earlier MindSpore division findings are different
operators: MS-001 (`fmod` in GRAPH), MS-014 (`fmod`/`remainder` zero signs), MS-019 (`fmod`/
`remainder` infinite divisor).

## Environment

MindSpore 2.9.0 CPU wheel, NumPy 1.26.4, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64.
Full dump in `collect_env.txt`; characterization transcripts in `evidence.txt`.

---

## Anticipated objections and responses

* **"Division by zero is undefined."**  It is not undefined for IEEE-754/C99 complex arithmetic:
  Annex G fixes the result (`inf`/`nan` with defined signs), and all four other frameworks in this
  campaign return it.  Returning `0` or an unrelated previous tensor is wrong under any convention.
* **"The value is just uninitialized memory; that is a quality-of-implementation detail."**  It is
  observable, deterministic per process, reproducible across two replays, and it leaks data between
  unrelated computations (including across shapes and dtypes), which makes it a correctness and
  data-isolation bug rather than a cosmetic one.
* **"Only division by exact zero is affected."**  Yes - and that is precisely the branch a numerical
  routine reaches when a denominator underflows to zero or a singular system is detected, i.e. the
  case where the caller most needs `inf`/`nan` rather than a silent `0`.
