# `torch.abs` on float8 CPU leaves the sign of `-0.0` (and of a negative NaN) in place

**Component:** PyTorch 2.12.1+cu130, CPU.  Two lanes: the eager ATen kernel (wrong for complete
64-element blocks) and `torch.compile` / Inductor C++ codegen (wrong for every tensor with `n >= 2`)
**Severity:** silent wrong value — `abs(-0.0)` returns `-0.0`, so any sign-sensitive consumer
(`torch.signbit`, `copysign`, `1/x`, `torch.sign`, serialized bytes, tests against NumPy/ml_dtypes)
sees the opposite sign from IEEE-754 and from every other float dtype; the eager result also changes
at a tensor-length boundary (63 vs 64 elements)
**Status:** present in PyTorch 2.12.1+cu130 on a CPU reporting `capability AVX2`; `float16`,
`bfloat16`, `float32` and `float64` are correct in both lanes
**Found with:** the wave-189 Rondo fp8 grid — 63 `torch.compile`-vs-eager comparisons over all five
`float8_*` dtypes.  `abs` was the only op that both lanes executed with different results (14 of the
15 jointly-supported ops agreed); the reproducer is `repro.py` in this directory
(`--replays 2` -> `RESULT: 0 failing replay(s).`, 56 checks), with the cold-cache divergence witness
run through Rondo's Fresh oracle (`fresh_witness.py`).

---

## Summary

IEEE-754 `abs` clears the sign bit.  For the float8 dtypes on CPU PyTorch does not:

```python
import torch
dt = torch.float8_e4m3fn

x = torch.full((2,), -0.0, dtype=dt)
torch.abs(x)                                     # bytes [0x00, 0x00] = +0.0   correct
torch.compile(lambda t: torch.abs(t))(x)         # bytes [0x80, 0x80] = -0.0   wrong

torch.full((63,), -0.0, dtype=dt).abs()          # all +0.0                   correct
torch.full((64,), -0.0, dtype=dt).abs()          # 64/64 still -0.0           wrong (eager)
torch.full((65,), -0.0, dtype=dt).abs()          # first 64 wrong, last one correct
```

Measured with a cold Inductor cache (one fresh process and cache directory per cell):

```text
  dtype          n      eager -0.0 kept   compiled -0.0 kept
  float8_e4m3fn  1            0/1                0/1        (Inductor falls back to eager)
  float8_e4m3fn  2..63        0/n                n/n        <- compiler divergence
  float8_e4m3fn  64          64/64              64/64
  float8_e4m3fn  65          64/65              65/65
  float8_e4m3fn  96          64/96              96/96
  float8_e4m3fn 128         128/128           128/128
  float8_e5m2    1..128     same pattern       same pattern
```

Eager is therefore wrong exactly inside **complete 64-element blocks** (the tail of a larger tensor
and non-contiguous inputs are correct, and in-place `abs_()` on a 64-block is wrong the same way);
Inductor's generated kernel is wrong for **every** `n >= 2`.  Warm in-process recompiles can fall
back to the eager op for some shapes, so the reproduction uses a dedicated cold-cache worker.

The same lowering also leaves the sign of a negative NaN: eager clears `0xFF -> 0x7F` for a single
element but keeps `0xFF` inside a 64-block, and the compiled kernel returns `0xFF` (`0xFE` for
`float8_e5m2`) for every size.

## Root cause

`ATen/cpu/vec/vec_base.h` defines `Vectorized<T>::abs()` three times behind SFINAE.  The
floating-point overload is the one that understands `-0.0`, and its own comment says so:

```cpp
template <typename other_t_abs = T,
          typename std::enable_if_t<!is_floating_point_v<other_t_abs> &&
                                    !c10::is_complex<other_t_abs>::value, int> = 0>
Vectorized<T> abs() const {
  return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
}

template <typename float_t_abs = T,
          typename std::enable_if_t<is_floating_point_v<float_t_abs>, int> = 0>
Vectorized<T> abs() const {
  // Specifically deal with floating-point because the generic code above
  // won't handle -0.0 (which should result in 0.0) properly.
  return map([](T x) -> T { return std::abs(x); });
}
```

`std::is_floating_point_v<at::Float8_e4m3fn>` and `...<at::Float8_e5m2>` are **false**, so the fp8
types select the first overload: `x < 0 ? -x : x`, for which `-0.0 < 0` is false, and the negative
NaN comparison is false as well.  The AVX512 specialisation in
`ATen/cpu/vec/vec512/vec512_float8.h` does use `andnot` (which would clear the bit)
```cpp
Vectorized<T> abs() const {
  return _mm512_andnot_si512(_mm512_set1_epi8(0x80), values);
}
```
but this host reports `capability AVX2`, so that path is not the one compiled here — and neither
`ATEN_CPU_CAPABILITY=avx512` nor `AVX2` changes the eager result, which confirms the generic
`vec_base` overload is the one in use.

## Impact

* Any fp8 quantization or inference pipeline that takes `abs` of activations/weights and then
  inspects the sign, divides by it, or compares against a NumPy/ml_dtypes reference silently sees a
  different value.  `1/abs(x)` at a zero is `+inf` where eager fp8 gives `-inf` in the block path.
* The eager result depends on the tensor length: the same 64 values give different signs at `n=63`
  and `n=64`, and the tail of a large tensor disagrees with its own first block.
* The compiled and eager lanes disagree for `2 <= n <= 63`, which breaks the standard
  `torch.compile`-vs-eager validation used by libraries that check bitwise reproducibility.

## Suggested fix

Make the `std::abs` overload cover the reduced-precision float types, e.g.

```cpp
template <typename float_t_abs = T,
          typename std::enable_if_t<std::is_floating_point_v<float_t_abs> ||
                                    c10::is_reduced_floating_point_v<float_t_abs>, int> = 0>
Vectorized<T> abs() const { return map([](T x) -> T { return std::abs(x); }); }
```

(or add explicit `Float8_e4m3fn`/`Float8_e5m2` overloads), and add a CPU test asserting that
`abs(-0.0)` is `+0.0` and that a negative NaN loses its sign bit for every float dtype in both eager
and `torch.compile`.

## Controls (all in `repro.py`, 56 checks)

* `float16`, `bfloat16`, `float32`, `float64`: both lanes clear the sign at `n=8` and `n=64`.
* `abs` of non-zero magnitudes is correct in both lanes (`-1.0 -> 1.0`, `-2.5 -> 2.5`); only the zero
  signs differ, so this is not a broken or missing `abs` kernel.
* A strided (non-contiguous) 64-element input and the tail of a 65/96-element tensor are correct in
  eager, which localises the eager defect to the contiguous 64-element vector block.
* `float8_e4m3fnuz` and `float8_e5m2fnuz` cannot represent `-0.0` at all (the input byte is `0x00`),
  so they are excluded as witnesses rather than counted as controls.
* `float8_e8m0fnu` has no eager `abs` (`abs_cpu not implemented`), so it is excluded too.
* Cold-cache Fresh-oracle cells (`fresh_witness.py`) give the same `compiled -0.0 = n/n` result on
  both replays.

## Novelty

Searched the PyTorch tracker and the web for `torch.abs` + `float8` / `-0.0` / `is_floating_point_v`
/ `vec_base abs`: no report matches.  The adjacent signed-zero work is for other operators and
lowerings — [PR #194472](https://github.com/pytorch/pytorch/pull/194472) (`minimum`/`maximum`),
[PR #193660](https://github.com/pytorch/pytorch/pull/193660) (CPU sum reduction), and the
CUDA-only `abs_cuda not implemented for 'Float8_e4m3fn'` report
([musubi-tuner #870](https://github.com/kohya-ss/musubi-tuner/issues/870)).  The campaign's other
signed-zero drafts cover different dtypes/ops (PT-024 complex `add`, MS-014 `fmod`/`remainder`,
MS-020 `copysign`, TF-011 eager scalar cache, TF-015 `cumsum`).

## Environment

PyTorch 2.12.1+cu130 (CPU execution), NumPy 2.5.3, Python 3.12.13, WSL2 (Linux 6.6.87.2) x86_64, CPU
capability `AVX2` (no AVX512), Inductor C++ codegen.  Full dump in `collect_env.txt`; transcripts in
`evidence.txt`.

---

## Anticipated objections and responses

* **"Float8 `abs` is not supported on CPU."**  It is: eager computes it (correctly for small
  tensors) and Inductor generates a kernel for it.  The defect is the sign of zero, not the absence
  of support.
* **"-0.0 is bit-level noise."**  It is observable: `torch.signbit` flips, `1/abs(x)` returns the
  wrong infinity, serialized bytes differ, and the same values give different results at `n=63` vs
  `n=64`.  The framework treats the same case as a bug for `float16`/`bfloat16`/`float32`/`float64`,
  and `vec_base.h` documents the -0.0 hazard in the very comment above the overload that fp8 misses.
* **"Only the AVX2 path is affected."**  That is the path exercised here; the AVX512 specialisation
  does clear the bit, but it is unreachable on AVX2-only hosts, which is exactly where the generic
  overload decides the semantics.  The fix belongs in `vec_base.h` so every ISA agrees.
