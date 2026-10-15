"""The 13 mutation operators and their exact proposal probabilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Mapping

from .domain import Invocation, Sequence, StateOperation, StepSpec

__all__ = [
    "FIELD_FOR_OPERATOR",
    "INPUT_OPERATORS",
    "MUTATION_OPERATORS",
    "STATE_OPERATOR_ORDER",
    "Mutator",
    "OperatorSpace",
    "Proposal",
]

#: The seven input operators and the :class:`StepSpec` field each one changes.
FIELD_FOR_OPERATOR: Mapping[str, str] = {
    "shape": "shape",
    "dtype": "dtype",
    "layout": "layout",
    "device": "device",
    "resample": "value_seed",
    "rescale": "scale",
    "special_values": "value_class",
}
INPUT_OPERATORS: tuple[str, ...] = tuple(FIELD_FOR_OPERATOR)

STATE_OPERATOR_ORDER: tuple[str, ...] = ("update", "mode", "restore", "backward")

MUTATION_OPERATORS: tuple[str, ...] = (
    "shape",
    "dtype",
    "layout",
    "device",
    "resample",
    "rescale",
    "special_values",
    "update",
    "mode",
    "restore",
    "backward",
    "return",
    "truncate",
)

_MUTATION_KINDS: tuple[str, ...] = ("input", "state", "return", "truncate")

_SPACE_ATTRIBUTE = {
    "shape": "shapes",
    "dtype": "dtypes",
    "layout": "layouts",
    "device": "devices",
    "value_class": "value_classes",
    "value_seed": "value_seeds",
    "scale": "scales",
}


@dataclass(frozen=True)
class OperatorSpace:
    """The finite declared domains of the step language."""

    operators: tuple[str, ...] = ("matmul", "mul", "add", "relu", "softmax", "sum")
    shapes: tuple[tuple[int, ...], ...] = (
        (1, 4),
        (2, 4),
        (4, 4),
        (16, 16),
        (1, 16),
        (2, 2, 2),
    )
    dtypes: tuple[str, ...] = ("float32", "float64")
    value_classes: tuple[str, ...] = (
        "ordinary",
        "zero",
        "neg_zero",
        "subnormal",
        "inf",
        "nan",
    )
    layouts: tuple[str, ...] = ("contiguous", "transposed")
    devices: tuple[str, ...] = ("cpu",)
    value_seeds: tuple[int, ...] = (0, 1, 2, 3)
    scales: tuple[float, ...] = (0.01, 1.0, 100.0, 1.0e6)
    update_parameters: tuple[Mapping[str, float], ...] = (
        {"weight": 0.0},
        {"weight": 1.0},
        {"weight": 4.0},
    )
    default_operator: str = "matmul"
    default_shape: tuple[int, ...] = (4, 4)

    def default_spec(self) -> StepSpec:
        return StepSpec(
            operator=self.default_operator,
            shape=self.default_shape,
            dtype=self.dtypes[0],
            value_class=self.value_classes[0],
            layout=self.layouts[0],
            device=self.devices[0],
        )

    def domain(self, base: StepSpec, field_name: str) -> tuple:
        """Finite domain of ``field_name`` for a spec of this shape rank."""
        if field_name == "shape":
            rank = len(base.shape)
            shapes = tuple(
                shape for shape in self.shapes if len(shape) == rank and shape != base.shape
            )
            return shapes or (base.shape,)
        return tuple(getattr(self, _SPACE_ATTRIBUTE[field_name]))


@dataclass(frozen=True)
class Proposal:
    """One generated sequence together with its forward probability."""

    sequence: Sequence
    operator: str
    kind: str
    forward_probability: float
    support: int
    generator: str = "mutation"

    def to_json(self) -> dict:
        return {
            "operator": self.operator,
            "kind": self.kind,
            "forward_probability": self.forward_probability,
            "support": self.support,
            "generator": self.generator,
            "sequence": self.sequence.to_json(),
        }


#: ``make_invocation(step_id, spec, repeat_of) -> Invocation``
InvocationFactory = Callable[..., Invocation]
#: ``make_state(step_id, operation, parameters) -> StateOperation``
StateFactory = Callable[..., StateOperation]
#: ``legal(sequence) -> bool``
LegalityCheck = Callable[[Sequence], bool]


class Mutator:
    """Samples one mutation of a sequence and prices it exactly."""

    def __init__(
        self,
        space: OperatorSpace,
        make_invocation: InvocationFactory,
        make_state: StateFactory,
        rng: random.Random,
        *,
        kind_weights: Mapping[str, float] | None = None,
        legal: LegalityCheck | None = None,
    ) -> None:
        self.space = space
        self.make_invocation = make_invocation
        self.make_state = make_state
        self.rng = rng
        self.kind_weights = dict(
            kind_weights
            if kind_weights is not None
            else {"input": 0.60, "state": 0.20, "return": 0.10, "truncate": 0.10}
        )
        self.legal = legal

    # proposal
    def propose(self, origin: Sequence) -> Proposal | None:
        """Draw one mutation of ``origin``."""
        for _ in range(8):
            kind = self._choose_kind(origin)
            if kind is None:
                continue
            entries = self._entries(origin, kind)
            if not entries:
                continue
            entry = entries[self.rng.randrange(len(entries))]
            outcome = self._apply(origin, kind, entry)
            if outcome is None or outcome.digest == origin.digest:
                continue
            if self.legal is not None and not self.legal(outcome):
                continue
            # Count *every* draw that yields this outcome, not only the one
            probability = self.probability(origin, outcome)
            if probability <= 0:
                continue
            return Proposal(
                sequence=outcome,
                operator=_describe(kind, entry),
                kind=kind,
                forward_probability=probability,
                support=len(entries),
            )
        return None

    def probability(self, origin: Sequence, outcome: Sequence) -> float:
        """Exact ``q(outcome | origin)`` of one :meth:`propose` call."""
        total = 0.0
        for kind in _MUTATION_KINDS:
            entries = self._entries(origin, kind)
            if not entries:
                continue
            matches = 0
            for entry in entries:
                candidate = self._apply(origin, kind, entry)
                if candidate is not None and candidate.digest == outcome.digest:
                    matches += 1
            if matches:
                total += self._kind_mass(kind, origin) * (matches / len(entries))
        return total

    # kinds
    def _applicable_kinds(self, origin: Sequence) -> list[str]:
        kinds = ["input"]
        if self._entries(origin, "state"):
            kinds.append("state")
        if self._entries(origin, "return"):
            kinds.append("return")
        if origin.length > 1:
            kinds.append("truncate")
        return kinds

    def _choose_kind(self, origin: Sequence) -> str | None:
        kinds = self._applicable_kinds(origin)
        if not kinds:
            return None
        weights = [self.kind_weights.get(kind, 0.0) for kind in kinds]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(kinds)
        threshold = self.rng.random() * total
        cumulative = 0.0
        for kind, weight in zip(kinds, weights):
            cumulative += weight
            if threshold <= cumulative:
                return kind
        return kinds[-1]

    def _kind_mass(self, kind: str, origin: Sequence) -> float:
        kinds = self._applicable_kinds(origin)
        if kind not in kinds:
            return 0.0
        weights = [self.kind_weights.get(item, 0.0) for item in kinds]
        total = sum(weights)
        if total <= 0:
            return 1.0 / len(kinds)
        return self.kind_weights.get(kind, 0.0) / total

    # support
    def _entries(self, origin: Sequence, kind: str) -> list[tuple]:
        if kind == "input":
            return self._input_entries(origin)
        if kind == "state":
            return self._state_entries(origin)
        if kind == "return":
            return self._return_entries(origin)
        if kind == "truncate":
            return [()] if origin.length > 1 else []
        raise ValueError(f"unknown mutation kind {kind!r}")

    def _input_entries(self, origin: Sequence) -> list[tuple]:
        default = self.space.default_spec()
        entries: list[tuple] = []
        for field_name in FIELD_FOR_OPERATOR.values():
            for index, step in enumerate(origin.steps):
                if not isinstance(step, Invocation):
                    continue
                base = step.spec or default
                current = getattr(base, field_name)
                for value in self.space.domain(base, field_name):
                    if value != current:
                        entries.append((field_name, index, value, None))
            # Appending a new invocation also chooses *which* model operation it
            for operator in self.space.operators:
                base = default.with_fields(operator=operator)
                for value in self.space.domain(base, field_name):
                    entries.append((field_name, None, value, operator))
        return entries

    def _state_entries(self, origin: Sequence) -> list[tuple]:
        entries: list[tuple] = []
        for operation in STATE_OPERATOR_ORDER:
            if operation == "update":
                for parameters in self.space.update_parameters:
                    entries.append((operation, dict(parameters)))
            elif operation == "mode":
                for mode_name in ("train", "eval"):
                    entries.append((operation, {"mode": mode_name}))
            else:
                entries.append((operation, {}))
        return entries

    def _return_entries(self, origin: Sequence) -> list[tuple]:
        return [
            (index,)
            for index, step in enumerate(origin.steps)
            if isinstance(step, Invocation)
        ]

    # application
    def _apply(self, origin: Sequence, kind: str, entry: tuple) -> Sequence | None:
        try:
            if kind == "input":
                return self._apply_input(origin, entry)
            if kind == "state":
                return self._apply_state(origin, entry)
            if kind == "return":
                return self._apply_return(origin, entry)
            if kind == "truncate":
                return origin.truncate(origin.length - 1)
        except (ValueError, KeyError):
            return None
        return None

    def _apply_input(self, origin: Sequence, entry: tuple) -> Sequence:
        field_name, index, value, operator = entry
        if index is None:
            base = self.space.default_spec()
            if operator is not None:
                base = base.with_fields(operator=operator)
            spec = base.with_fields(**{field_name: value})
            invocation = self.make_invocation(_fresh_id(origin), spec, None)
            return origin.append(invocation)
        step = origin.steps[index]
        assert isinstance(step, Invocation)
        base = step.spec or self.space.default_spec()
        spec = base.with_fields(**{field_name: value})
        invocation = self.make_invocation(step.step_id, spec, step.repeat_of)
        return origin.replace_step(index, invocation)

    def _apply_state(self, origin: Sequence, entry: tuple) -> Sequence:
        operation, parameters = entry
        step = self.make_state(_fresh_id(origin), operation, parameters)
        return origin.append(step)

    def _apply_return(self, origin: Sequence, entry: tuple) -> Sequence:
        (index,) = entry
        target = origin.steps[index]
        assert isinstance(target, Invocation)
        invocation = self.make_invocation(
            _fresh_id(origin), target.spec, target.step_id
        )
        return origin.append(invocation)


def _fresh_id(sequence: Sequence) -> str:
    existing = {step.step_id for step in sequence.steps}
    counter = sequence.length
    while f"s{counter}" in existing:
        counter += 1
    return f"s{counter}"


def _describe(kind: str, entry: tuple) -> str:
    if kind == "input":
        field_name, index, _, operator = entry
        operator_name = next(
            name for name, target in FIELD_FOR_OPERATOR.items() if target == field_name
        )
        position = "append" if index is None else f"replace@{index}"
        suffix = f":{operator}" if operator else ""
        return f"{operator_name}:{position}{suffix}"
    if kind == "state":
        return f"state:{entry[0]}"
    if kind == "return":
        return f"return@{entry[0]}"
    return "truncate"
