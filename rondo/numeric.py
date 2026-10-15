"""Representation-aware numeric comparison."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "DEFAULT_TOLERANCES",
    "MismatchKind",
    "NumericPolicy",
    "compare_scalar",
    "float_bits",
    "is_exact_zero",
    "is_nan_bits",
    "is_subnormal",
    "sign_bit",
    "subnormal_integer",
]

FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
COMPLEX_DTYPES = ("complex64", "complex128")
INT_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16")
_SIGNED_FLOAT_DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES

#: ``(relative tolerance, absolute tolerance)`` per dtype.
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "float16": (2e-3, 2e-3),
    "bfloat16": (2e-2, 2e-2),
    "float32": (1e-5, 1e-6),
    "float64": (1e-8, 1e-10),
    "complex64": (1e-5, 1e-6),
    "complex128": (1e-8, 1e-10),
}


class MismatchKind:
    """Canonical symptom kinds (also used in :class:`SymptomSignature`)."""

    NAN_PATTERN = "nan-pattern-mismatch"
    NAN_PAYLOAD = "nan-payload-mismatch"
    INFINITE = "infinite-value-mismatch"
    SIGNED_ZERO = "signed-zero-mismatch"
    ZERO_VS_NONZERO = "zero-vs-nonzero-mismatch"
    SUBnormal = "subnormal-mismatch"
    NUMERIC = "numeric-mismatch"
    EXACT = "exact-mismatch"
    DTYPE = "dtype-mismatch"
    SHAPE = "shape-mismatch"
    DEVICE = "device-mismatch"
    STRIDES = "strides-mismatch"
    STRUCTURE = "structure-mismatch"
    SCALAR_TYPE = "scalar-type-mismatch"
    EXCEPTION = "exception-mismatch"
    ELEMENT_COUNT = "element-count-mismatch"


_BIT_LAYOUT = {
    "float16": (16, 5, 10),
    "bfloat16": (16, 8, 7),
    "float32": (32, 8, 23),
    "float64": (64, 11, 52),
}


def _float_bit_layout(dtype: str) -> tuple[int, int, int]:
    if dtype not in _BIT_LAYOUT:
        raise ValueError(f"{dtype!r} is not a floating-point dtype")
    return _BIT_LAYOUT[dtype]


def float_bits(value: float, dtype: str) -> int:
    """Raw IEEE-754 bit pattern of ``value`` interpreted as ``dtype``."""
    total, _, _ = _float_bit_layout(dtype)
    if dtype == "float64":
        return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    if dtype == "float32":
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", float(value)))[0]
    # bfloat16: round the float32 pattern to nearest-even, then keep 16 bits.
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return (rounded >> 16) & 0xFFFF if total == 16 else rounded


def sign_bit(bits: int, total: int) -> int:
    return (bits >> (total - 1)) & 1


def _fields(bits: int, dtype: str) -> tuple[int, int, int]:
    """Return ``(sign, exponent_field, mantissa_field)`` for a float dtype."""
    total, exp_bits, mant_bits = _float_bit_layout(dtype)
    sign = (bits >> (total - 1)) & 1
    exponent = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    mantissa = bits & ((1 << mant_bits) - 1)
    return sign, exponent, mantissa


def is_nan_bits(bits: int, dtype: str) -> bool:
    _, exponent, mantissa = _fields(bits, dtype)
    _, exp_bits, _ = _float_bit_layout(dtype)
    return exponent == (1 << exp_bits) - 1 and mantissa != 0


def is_exact_zero(value: float, dtype: str = "float64") -> bool:
    """True when ``value`` is bit-exactly ``+0.0`` or ``-0.0``."""
    bits = float_bits(value, dtype)
    total, exp_bits, mant_bits = _float_bit_layout(dtype)
    sign, exponent, mantissa = _fields(bits, dtype)
    del sign, total, exp_bits, mant_bits
    return exponent == 0 and mantissa == 0


def is_subnormal(value: float, dtype: str = "float64") -> bool:
    bits = float_bits(value, dtype)
    _, exponent, mantissa = _fields(bits, dtype)
    return exponent == 0 and mantissa != 0


def subnormal_integer(value: float, dtype: str = "float64") -> int:
    """Exact integer scale of a subnormal: its raw mantissa."""
    bits = float_bits(value, dtype)
    _, exponent, mantissa = _fields(bits, dtype)
    if exponent != 0 or mantissa == 0:
        raise ValueError(f"{value!r} is not a subnormal {dtype}")
    return mantissa


def _within_relative_tolerance(left: int, right: int, rtol: float) -> bool:
    """Exact integer relative comparison (no float rounding)."""
    if left == right:
        return True
    numerator, denominator = float(rtol).as_integer_ratio()
    difference = abs(left - right)
    scale = max(abs(left), abs(right))
    return difference * denominator <= numerator * scale


@dataclass(frozen=True)
class NumericPolicy:
    """Tolerances and structural switches of result comparison."""

    tolerances: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_TOLERANCES)
    )
    sign_of_zero: bool = True
    zero_vs_nonzero: bool = True

    @classmethod
    def relaxed(
        cls, tolerances: Mapping[str, tuple[float, float]] | None = None
    ) -> "NumericPolicy":
        return cls(
            tolerances=dict(tolerances or DEFAULT_TOLERANCES),
            sign_of_zero=False,
            zero_vs_nonzero=False,
        )

    def tolerance(self, dtype: str) -> tuple[float, float]:
        return self.tolerances.get(dtype, (0.0, 0.0))


DEFAULT_NUMERIC_POLICY = NumericPolicy()


def _complex_parts(value: Any) -> tuple[float, float] | None:
    if isinstance(value, complex):
        return value.real, value.imag
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def compare_scalar(
    left: Any,
    right: Any,
    dtype: str,
    policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
) -> str | None:
    """Compare two leaves of the same declared dtype."""
    if dtype in COMPLEX_DTYPES:
        left_parts = _complex_parts(left)
        right_parts = _complex_parts(right)
        if left_parts is None or right_parts is None:
            return None if left == right else MismatchKind.EXACT
        component = "float32" if dtype == "complex64" else "float64"
        for a, b in zip(left_parts, right_parts):
            reason = compare_float(a, b, component, policy)
            if reason is not None:
                return reason
        return None

    if dtype in FLOAT_DTYPES:
        try:
            left_float = float(left)
            right_float = float(right)
        except (TypeError, ValueError):
            return None if left == right else MismatchKind.EXACT
        return compare_float(left_float, right_float, dtype, policy)

    if dtype == "bool" or dtype in INT_DTYPES:
        return None if left == right else MismatchKind.EXACT

    return None if left == right else MismatchKind.EXACT


def compare_float(
    left: float,
    right: float,
    dtype: str,
    policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
) -> str | None:
    """Compare two Python floats interpreted as ``dtype``."""
    left_nan = math.isnan(left)
    right_nan = math.isnan(right)
    if left_nan or right_nan:
        if not (left_nan and right_nan):
            return MismatchKind.NAN_PATTERN
        left_bits = float_bits(left, dtype)
        right_bits = float_bits(right, dtype)
        return None if left_bits == right_bits else MismatchKind.NAN_PAYLOAD

    if math.isinf(left) or math.isinf(right):
        return None if left == right else MismatchKind.INFINITE

    left_zero = is_exact_zero(left, dtype)
    right_zero = is_exact_zero(right, dtype)
    if left_zero and right_zero:
        if policy.sign_of_zero:
            total, _, _ = _float_bit_layout(dtype)
            if sign_bit(float_bits(left, dtype), total) != sign_bit(
                float_bits(right, dtype), total
            ):
                return MismatchKind.SIGNED_ZERO
        return None
    if policy.zero_vs_nonzero and (left_zero or right_zero):
        return MismatchKind.ZERO_VS_NONZERO

    rtol, atol = policy.tolerance(dtype)
    if is_subnormal(left, dtype) or is_subnormal(right, dtype):
        # Absolute tolerance is meaningless below the normal range; compare the
        left_scale = subnormal_integer(left, dtype) if is_subnormal(left, dtype) else 0
        right_scale = (
            subnormal_integer(right, dtype) if is_subnormal(right, dtype) else 0
        )
        if _within_relative_tolerance(left_scale, right_scale, rtol):
            return None
        return MismatchKind.SUBnormal

    if math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
        return None
    return MismatchKind.NUMERIC
