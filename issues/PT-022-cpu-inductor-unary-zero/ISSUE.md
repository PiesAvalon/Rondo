# CPU Inductor's unary kernels diverge from eager on zero and subnormal results (`expm1`, `atanh`, `erf`, `asinh`, `frac`, `elu`, `selu`)

**Component:** PyTorch 2.12.1+cu130, **CPU**, `torch.compile` (Inductor) vs eager.
**Severity:** silent wrong results - a lost sign of zero (which flips `1/x` between `-inf` and `+inf`),
a flushed subnormal (`erf(1e-38)` -> `0.0` instead of `1.1283791e-38`) and, from wave 191, a flush
that reaches `1e-8` (`expm1(1e-8)`/`atanh(1e-8)`/`elu(-1e-8)` -> `0.0`) plus the reversed sign loss
`frac(-0.0) -> -0.0`.
**Found with:** Rondo's width oracle, then confirmed with a strict compiler harness that puts **one
(op, width) combination per process** (in-process multi-variant compilation turned out to be
unreliable - see "Methodology" below).

---

## Summary

`torch.compile` chooses a scalar kernel for a 1-element tensor and a vectorised kernel from 16
elements upwards.  On CPU the vectorised kernel disagrees with both eager and the scalar compiled
kernel:

| expression | eager | compiled `n=1` | compiled `n=16` | fp64 reference |
|---|---|---|---|---|
| `torch.expm1(-0.0)` | `-0.0` | `-0.0` | **`+0.0`** | `-0.0` |
| `torch.atanh(-0.0)` | `-0.0` | `-0.0` | **`+0.0`** | `-0.0` |
| `torch.erf(1e-38)` | `1.1283791e-38` | `1.1283791e-38` | **`+0.0`** | `1.1283791e-38` |

```python
import torch
x = torch.full((16,), -0.0)
print(repr(float(torch.expm1(x)[0])))                                  # -0.0   (eager)
print(repr(float(torch.compile(torch.expm1)(x)[0])))                   # 0.0    <-- wrong sign
y = torch.full((16,), 1e-38)
print(repr(float(torch.erf(y)[0])))                                    # 1.1283791e-38 (subnormal)
print(repr(float(torch.compile(torch.erf)(y)[0])))                     # 0.0    <-- flushed
```

The sign loss is observable downstream: `1.0 / torch.expm1(-0.0)` is `-inf` eagerly and `+inf`
through the compiled kernel.  The subnormal loss is observable directly as a value difference, not
just a sign: `erf(1e-38)` is a representable float32 subnormal and eager returns it.

## Fusion does not hide it, and the impact is a value, not just a sign

Measured one expression per process (`tools_compile_fusion_probe.py`, 2 repeats each), after an earlier
harness of mine had suggested that fusing masked the defect - that observation was an artifact of
in-process multi-variant compilation and is retracted:

| expression, `n=16` | eager | compiled | fp64 |
|---|---|---|---|
| `expm1(-0.0)` | `-0.0` | **`+0.0`** | `-0.0` |
| `expm1(-0.0) * 1.0` | `-0.0` | **`+0.0`** | `-0.0` |
| `copysign(1.0, expm1(-0.0))` | `-1.0` | **`+1.0`** | `-1.0` |
| `1.0 / expm1(-0.0)` | `-inf` | **`+inf`** | `-inf` |
| `expm1(1e-38)` | `9.999999350456404e-39` | **`0.0`** | `1e-38` |
| `1.0 / expm1(1e-38)` | `1.0000000694406173e+38` | **`inf`** | `1e+38` |

So the loss is in the `expm1` kernel and propagates through fused expressions, at `n=16` but not at
`n=1`.  The last row is the most consequential: the subnormal flush turns a **finite `1e38` into
`inf`** once a reciprocal is fused in, and `copysign`/`signbit`-style consumers see `+1.0` instead of
`-1.0`.  Method note: a class-only comparison (nan / +-inf / signed zero) reports the `copysign` rows
as "agree" because `-1.0` and `+1.0` are both finite, so the reproducer checks those raw values too.

## Reproduction

`python repro.py --replays 2` -> `RESULT: 0 failing check(s).`

The reproducer spawns **one subprocess per (op, width)** (six subprocesses), each of which compiles
one function once, calls it once, and prints eager and compiled classes; it then asserts:

* `expm1(-0.0)` and `atanh(-0.0)`: compiled `+0.0` at `n=16` while eager and fp64 are `-0.0`;
* `erf(1e-38)`: compiled `+0.0` at `n=16` while eager and fp64 keep `1.1283791e-38`;
* the compiled `n=1` kernel agrees with eager for all three (the scalar kernel is fine);
* controls that must agree: `log1p`, `sqrt`, `tanh`, `asinh`, `ndtr`, `sin`, `reciprocal`, `rsqrt`
  at the same widths.

## Methodology (why the harness looks unusual)

The first two versions of this measurement were wrong, and the artifacts are worth stating because
they are the reason for the subprocess-per-combination design:

* passing a raw builtin (e.g. `torch.expm1` as-is) to `torch.compile` makes dynamo skip it silently,
  so the "compiled" column is really eager;
* wrapping many ops in one shared closure **code object** fills dynamo's per-code-object recompile
  cache, after which later variants also silently run eager (the tell is `compiled == eager` exactly);
* even with a distinct code object per op, compiling two variants of the *same* op in one process is
  unreliable - `gelu`/`softshrink`/`maximum`/`minimum` still collapsed to eager, and one combination
  disagreed with a later probe of the same process;
* fusion changes the answer (standalone compiled `expm1` loses the sign, while the fused
  `torch.reciprocal(torch.expm1(x))` did not in the same harness).

Everything asserted above therefore comes from one subprocess per combination, and the raw values are
printed so a reviewer can see the actual bits rather than a derived verdict.

## Related reports and scope

This is a **CPU** instance in a family that is tracked per platform, so the references matter:

| report | scope | state |
|---|---|---|
| `pytorch#189799` | umbrella: Inductor mishandles `-0.0` across operations (`add`, `elu`, `sqrt+rsqrt`) | open |
| `pytorch#194347` | **CPU** `torch.minimum` signed zeros under `torch.compile` | open |
| `pytorch#191038` | **MPS** `expm1(-0.0)` (explicitly MPS-scoped) | closed |
| `pytorch#185878` | CUDA/Triton `expm1` analogue / "Disable FTZ for Inductor eager numerics mode" | closed |
| `pytorch#188541` | **CUDA** `F.logsigmoid` flushes negative float32 subnormal outputs to `+0.0` | open |
| `pytorch#183606` | `torch.frac` signed-zero under `torch.compile` | closed |

No report covers CPU `expm1`/`atanh` sign loss or the CPU subnormal flush in `erf`; the CPU
signed-zero case has a separate issue for a different op (#194347), which is the precedent for a CPU
report being distinct from the MPS/CUDA ones.  If a maintainer prefers to treat this as another
instance of #189799 rather than a separate issue, the per-op, per-width table above is the instance
list to attach there - the subnormal part is a different mechanism from the signed-zero umbrella.

## Files

* `repro.py` - self-checking reproducer (subprocess per combination, raw values, controls, 2 replays)
* `evidence.txt` - raw observations
* `collect_env.txt` - environment

## Wave-191 extension: the flush reaches `1e-8`, and `erf` keeps the sign while losing the value

A cold-cache sweep of 48 unary ops at 14 special values (one op per process, with an empty
`TORCHINDUCTOR_CACHE_DIR`) found the same vectorised path in three more operators and showed that the
flush is not limited to the subnormal range:

| expression, `n=16` | eager | compiled | fp64 reference |
|---|---|---|---|
| `expm1(1e-8)` | `9.99999994e-09` | **`+0.0`** | `1e-08` |
| `atanh(1e-8)` | `9.99999994e-09` | **`+0.0`** | `1e-08` |
| `asinh(1.4012985e-45)` | `1.4012985e-45` | **`+0.0`** | `1.4012985e-45` |
| `erf(-1e-38)` | `-1.1283791e-38` | **`-0.0`** | `-1.1283791e-38` |
| `elu(-1e-8)` | `-9.99999994e-09` | **`+0.0`** | `-1e-08` |
| `selu(-1e-8)` | `-1.7580993e-08` | **`+0.0`** | `-1.7580993e-08` |

`elu`/`selu` are `expm1`-based, so the flush reaches them through the same kernel; `asinh` flushes
the smallest subnormal; `erf` flushes a negative input to `-0.0`, keeping the sign while losing the
magnitude; and the `1e-8` rows are three orders of magnitude above the subnormal range, so this is a
value loss rather than a denormal-mode effect.  All six are correct at `n=1` (the reproducer carries a
width-1 control for each), which is the same scalar-vs-vectorised split as the original witnesses.

### Found during the extension and excluded as already known

* `frac(-0.0) -> -0.0` (`x - trunc(x)` is `+0.0`; wrong at `n=1` too): pytorch#183606 covers
  `torch.frac` signed zeros under `torch.compile` and is closed.
* `elu(-0.0)`/`selu(-0.0) -> +0.0`: the #189799 umbrella explicitly lists `elu` for `-0.0`
  mishandling, so these are not carried as new witnesses here.
* `gelu(-0.0)`: eager returns `+0.0` where fp64 returns `-0.0` - the *eager* side loses the sign,
  covered by PT-021 and pytorch#189804.

## Excluded as precision, not a defect

The same sweep found <=2-ulp differences on ordinary inputs for `sin`, `cos`, `tan`, `atan`, `asinh`,
`erf` and `expm1` (for example `cos(1.0)`: eager `0.5403023362` vs compiled `0.5403022766`).  Those
are the expected scalar-libm vs vectorised-libm difference and are not reported.  `gelu(-0.0)` is
excluded too: eager returns `+0.0` where fp64 gives `-0.0`, i.e. the *eager* side loses the sign,
which is already covered by PT-021 and pytorch#189804.
