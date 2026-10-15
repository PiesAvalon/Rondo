"""Guard-directed synthesis of inputs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence as TypingSequence

from .domain import GuardGoal, StepSpec
from .generation import OperatorSpace, Proposal

try:  # pragma: no cover - exercised only when z3 is installed
    import z3
except ImportError:  # pragma: no cover
    z3 = None

__all__ = [
    "SynthesisError",
    "SynthesisOutcome",
    "Synthesizer",
    "UnsupportedConstraint",
    "Z3_AVAILABLE",
]

Z3_AVAILABLE = z3 is not None


class SynthesisError(ValueError):
    """A goal could not be solved."""


class UnsupportedConstraint(SynthesisError):
    """The declared goal IR cannot express a guard."""


@dataclass(frozen=True)
class SynthesisOutcome:
    """Result of one synthesis query."""

    goal: GuardGoal
    status: str  # "sat" | "unsat" | "unsupported"
    solutions: tuple[StepSpec, ...] = ()
    solver: str = "enumerator"
    reason: str = ""

    @property
    def solved(self) -> bool:
        return self.status == "sat" and bool(self.solutions)


class Synthesizer:
    """Turns an unexplored guard outcome into a concrete invocation spec."""

    def __init__(
        self,
        space: OperatorSpace,
        rng: random.Random,
        *,
        use_smt: bool = True,
        solution_limit: int = 8,
    ) -> None:
        self.space = space
        self.rng = rng
        self.use_smt = use_smt and Z3_AVAILABLE
        self.solution_limit = max(1, solution_limit)
        self.queries = 0
        self.sat = 0
        self.unsat = 0
        self.unsupported = 0

    # query
    def synthesize(
        self, goal: GuardGoal, *, avoid: Iterable[tuple[int, ...]] = ()
    ) -> SynthesisOutcome:
        """Return a finite set of specs that take the goal's outcome."""
        self.queries += 1
        avoided = {tuple(shape) for shape in avoid}
        try:
            if self.use_smt:
                shapes, solver = self._solve_with_z3(goal)
            else:
                shapes, solver = self._solve_by_enumeration(goal), "enumerator"
        except UnsupportedConstraint as error:
            self.unsupported += 1
            return SynthesisOutcome(goal, "unsupported", reason=str(error))
        except SynthesisError as error:
            self.unsat += 1
            return SynthesisOutcome(goal, "unsat", reason=str(error))

        shapes = tuple(shape for shape in shapes if shape not in avoided)
        if not shapes:
            self.unsat += 1
            return SynthesisOutcome(
                goal,
                "unsat",
                solver=solver,
                reason="no unvisited shape satisfies the goal",
            )
        solutions = tuple(
            self._spec_for(goal, shape) for shape in shapes[: self.solution_limit]
        )
        self.sat += 1
        return SynthesisOutcome(
            goal, "sat", solutions=solutions, solver=solver, reason=goal.describe()
        )

    # proposal
    def propose(
        self,
        origin,
        goals: TypingSequence[GuardGoal],
        *,
        make_invocation,
        legal=None,
    ) -> Proposal | None:
        """Draw one synthesis proposal from ``origin``."""
        if not goals:
            return None
        ordered_goals = tuple(goals)
        goal = ordered_goals[self.rng.randrange(len(ordered_goals))]
        avoid = [
            step.spec.shape
            for step in origin.steps
            if getattr(step, "spec", None) is not None
            and len(step.spec.shape) == (goal.rank or 2)
        ]
        outcome = self.synthesize(goal, avoid=avoid)
        if not outcome.solved:
            return None
        spec = outcome.solutions[self.rng.randrange(len(outcome.solutions))]

        positions = self._positions(origin)
        if not positions:
            return None
        chosen = positions[self.rng.randrange(len(positions))]
        candidate = self._apply(origin, chosen, spec, make_invocation)
        if candidate is None or candidate.digest == origin.digest:
            return None
        if legal is not None and not legal(candidate):
            return None

        # Price the *outcome*, so identical solutions reachable through several
        probability = self.probability(origin, candidate, ordered_goals)
        if probability <= 0:
            return None
        return Proposal(
            sequence=candidate,
            operator=f"synthesis:{goal.describe()}",
            kind="synthesis",
            forward_probability=probability,
            support=len(ordered_goals) * len(outcome.solutions) * len(positions),
            generator="synthesis",
        )

    def probability(self, origin, outcome, goals: TypingSequence[GuardGoal]) -> float:
        """Exact ``q(outcome | origin)`` of one :meth:`propose` call."""
        if not goals:
            return 0.0
        ordered_goals = tuple(goals)
        positions = self._positions(origin)
        if not positions:
            return 0.0
        total = 0.0
        for goal in ordered_goals:
            avoid = [
                step.spec.shape
                for step in origin.steps
                if getattr(step, "spec", None) is not None
                and len(step.spec.shape) == (goal.rank or 2)
            ]
            result = self.synthesize(goal, avoid=avoid)
            if not result.solved:
                continue
            matches = 0
            # The adapter's materialization is deterministic, so a solution is
            for spec in result.solutions:
                for position in positions:
                    if self._matches(origin, outcome, position, spec):
                        matches += 1
            if matches:
                total += (
                    (1.0 / len(ordered_goals))
                    * (1.0 / len(result.solutions))
                    * (matches / len(positions))
                )
        return total

    # positions
    def _positions(self, origin) -> list[int | None]:
        options: list[int | None] = [None]  # append
        for index, step in enumerate(origin.steps):
            if getattr(step, "spec", None) is not None:
                options.append(index)
        return options

    def _apply(self, origin, position, spec: StepSpec, make_invocation):
        new_id = _fresh_id(origin)
        if position is None:
            return origin.append(make_invocation(new_id, spec, None))
        step = origin.steps[position]
        invocation = make_invocation(step.step_id, spec, step.repeat_of)
        return origin.replace_step(position, invocation)

    def _matches(self, origin, outcome, position, spec: StepSpec) -> bool:
        if position is None:
            appended = outcome.steps[-1] if outcome.length == origin.length + 1 else None
            return (
                appended is not None
                and getattr(appended, "spec", None) == spec
            )
        if outcome.length != origin.length:
            return False
        step = outcome.steps[position]
        return getattr(step, "spec", None) == spec and step.step_id == origin.steps[position].step_id

    # solvers
    def _spec_for(self, goal: GuardGoal, shape: tuple[int, ...]) -> StepSpec:
        base = self.space.default_spec()
        return base.with_fields(
            shape=tuple(shape),
            operator=goal.op or base.operator,
            dtype=goal.dtype or base.dtype,
            value_class=goal.value_class or base.value_class,
        )

    def _solve_by_enumeration(self, goal: GuardGoal) -> tuple[tuple[int, ...], ...]:
        rank = goal.rank
        shapes = [
            shape
            for shape in self.space.shapes
            if (rank is None or len(shape) == rank) and _shape_satisfies(shape, goal)
        ]
        if not shapes:
            raise SynthesisError("the declared shape domain has no satisfying shape")
        return tuple(shapes)

    def _solve_with_z3(self, goal: GuardGoal) -> tuple[tuple[tuple[int, ...], ...], str]:
        rank = goal.rank
        if rank is None:
            # Rank is only a filter over the declared domain, not a constraint.
            return self._solve_by_enumeration(goal), "enumerator"
        dimensions = [z3.Int(f"d{index}") for index in range(rank)]
        solver = z3.Solver()
        for index, dim in enumerate(dimensions):
            solver.add(dim >= 1, dim <= 256)
        _add_shape_constraints(solver, dimensions, goal)

        solutions: list[tuple[int, ...]] = []
        while len(solutions) < self.solution_limit:
            if solver.check() != z3.sat:
                break
            model = solver.model()
            shape = tuple(
                int(model.evaluate(dim, model_completion=True).as_long())
                for dim in dimensions
            )
            solutions.append(shape)
            solver.add(z3.Not(z3.And([dim == value for dim, value in zip(dimensions, shape)])))
        if not solutions:
            raise SynthesisError("guard and scenario preconditions are unsatisfiable")
        return tuple(solutions), "z3"


def _shape_satisfies(shape: tuple[int, ...], goal: GuardGoal) -> bool:
    if goal.shape_eq is not None and shape[0] != goal.shape_eq:
        return False
    if goal.shape_dim_1 is not None:
        index = min(goal.shape_dim_1, len(shape) - 1)
        if shape[index] != 1:
            return False
    if goal.shape_divisible_by is not None:
        if any(dim % goal.shape_divisible_by for dim in shape):
            return False
    if goal.rank is not None and len(shape) != goal.rank:
        return False
    return True


def _add_shape_constraints(solver, dimensions, goal: GuardGoal) -> None:
    if goal.shape_eq is not None:
        if not dimensions:
            raise UnsupportedConstraint("shape_eq needs a rank")
        solver.add(dimensions[0] == goal.shape_eq)
    if goal.shape_dim_1 is not None:
        if goal.shape_dim_1 >= len(dimensions):
            raise UnsupportedConstraint(
                f"shape_dim_1={goal.shape_dim_1} exceeds rank {len(dimensions)}"
            )
        solver.add(dimensions[goal.shape_dim_1] == 1)
    if goal.shape_divisible_by is not None:
        if goal.shape_divisible_by <= 0:
            raise UnsupportedConstraint("shape_divisible_by must be positive")
        for dim in dimensions:
            solver.add(dim % goal.shape_divisible_by == 0)


def _fresh_id(sequence) -> str:
    existing = {step.step_id for step in sequence.steps}
    counter = sequence.length
    while f"y{counter}" in existing:
        counter += 1
    return f"y{counter}"
