"""PyTorch adapter (``torch.compile``)."""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import Decision, Invocation, Scenario
from ._common import CompiledFrameworkAdapter
from .reference import dtype_class, spec_tag

__all__ = ["PyTorchAdapter"]


def _torch_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }.get(name, torch.float32)


class PyTorchAdapter(CompiledFrameworkAdapter):
    framework = "pytorch"

    def __init__(self, scenario: Scenario, model_factory, state_factory, space=None):
        super().__init__(scenario, model_factory, state_factory, space)
        self._compiled_signatures: set = set()

    # primitives
    def _materialize(self, spec) -> list:
        import torch

        from .base import materialize_values

        values = materialize_values(spec)
        tensor = torch.tensor(values, dtype=_torch_dtype(spec.dtype), device=self.scenario.device)
        return [tensor.reshape(spec.shape).contiguous()]

    def _eager(self, model, args):
        return model(*args)

    def _compile(self, model, options: Mapping):
        import torch

        return torch.compile(
            model,
            backend=str(options.get("backend", "inductor")),
            dynamic=bool(options.get("dynamic", False)),
            fullgraph=bool(options.get("fullgraph", False)),
        )

    def _compiled(self, compiled, args):
        return compiled(*args)

    def _reset_compiler_state(self) -> None:
        import torch

        torch._dynamo.reset()

    def _fresh_model(self, model):
        import torch

        fresh = self.model_factory()
        try:
            fresh.load_state_dict(model.state_dict())
        except Exception:
            pass
        del torch
        return fresh

    def _decision(self, step: Invocation) -> Decision:
        # A production adapter reads this from Dynamo (a new graph captured, a
        spec = step.spec
        if spec is None:
            return Decision.REUSED
        signature = (spec.operator, spec_tag(spec), dtype_class(spec.dtype))
        if signature in self._compiled_signatures:
            return Decision.REUSED
        self._compiled_signatures.add(signature)
        return Decision.COMPILED

    def _lowering(self, step: Invocation) -> str:
        return "dynamo+inductor"

    def _apply_state(self, model, operation: str, parameters: Mapping) -> None:
        import torch

        named = dict(model.named_parameters())
        if operation == "update":
            with torch.no_grad():
                for name, value in parameters.items():
                    parameter = named.get(name)
                    if parameter is not None:
                        parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))
        elif operation == "restore":
            with torch.no_grad():
                for name, value in parameters.items():
                    parameter = named.get(name)
                    if parameter is not None:
                        parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))
        elif operation == "mode":
            model.train(str(parameters.get("mode", "train")) == "train")
        elif operation == "backward":
            for parameter in named.values():
                parameter.requires_grad_(True)

    def _observe(self, value: Any, spec, state) -> Any:
        return super()._observe(value, spec, state)
