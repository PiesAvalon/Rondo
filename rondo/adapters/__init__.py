"""Framework adapters."""

from __future__ import annotations

from ._common import CompiledFrameworkAdapter
from .base import FrameworkAdapter
from .jax import JaxAdapter
from .mindspore import MindSporeAdapter
from .pytorch import PyTorchAdapter
from .reference import CompilerDefectProfile, ReferenceAdapter
from .tensorflow import TensorFlowAdapter

__all__ = [
    "CompiledFrameworkAdapter",
    "CompilerDefectProfile",
    "FrameworkAdapter",
    "JaxAdapter",
    "MindSporeAdapter",
    "PyTorchAdapter",
    "ReferenceAdapter",
    "TensorFlowAdapter",
]

ADAPTERS = {
    "reference": ReferenceAdapter,
    "pytorch": PyTorchAdapter,
    "tensorflow": TensorFlowAdapter,
    "jax": JaxAdapter,
    "mindspore": MindSporeAdapter,
}
