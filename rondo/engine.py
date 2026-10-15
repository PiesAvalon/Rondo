"""The Rondo engine: exploration → oracle → refinement."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .adapters.base import FrameworkAdapter
from .domain import GuardGoal, Mode, Sequence, Witness, WitnessClass
from .feedback import BehaviorFeedback, RewardConfig
from .generation import Mutator
from .oracle import Oracle
from .refinement import Refiner, RefinementConfig
from .sampler import RondoMCMC, RoundRecord, SamplerConfig
from .synthesis import Synthesizer

__all__ = ["CampaignResult", "Rondo", "RondoConfig"]


@dataclass(frozen=True)
class RondoConfig:
    """Everything the engine needs to run a campaign."""

    seed: int = 0
    exploration_rounds: int = 200
    check_every: int = 1
    max_witnesses: int = 32
    use_smt: bool = True
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)


@dataclass
class CampaignResult:
    """What a campaign found (a record of the method, not a measurement)."""

    witnesses: list[Witness] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    oracle_checks: int = 0
    candidates: int = 0
    candidates_by_class: dict[str, int] = field(default_factory=dict)
    feedback: dict[str, int] = field(default_factory=dict)

    @property
    def stateful(self) -> list[Witness]:
        return [w for w in self.witnesses if w.witness_class is WitnessClass.STATEFUL]

    def by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for witness in self.witnesses:
            counts[witness.witness_class.value] = counts.get(witness.witness_class.value, 0) + 1
        return counts

    def summarize(self) -> str:
        lines = [
            f"rounds            : {len(self.rounds)}",
            f"oracle checks     : {self.oracle_checks}",
            f"candidates        : {self.candidates}",
            f"confirmed witnesses: {len(self.witnesses)}",
        ]
        for name, count in sorted(self.by_class().items()):
            lines.append(f"  {name:<18}: {count}")
        if self.candidates_by_class:
            lines.append("candidate classes :")
            for name, count in sorted(self.candidates_by_class.items()):
                lines.append(f"  {name:<24}: {count}")
        return "\n".join(lines)


class Rondo:
    """Runs the whole method against one adapter and one scenario."""

    def __init__(self, adapter: FrameworkAdapter, config: RondoConfig | None = None) -> None:
        self.adapter = adapter
        self.config = config or RondoConfig()
        self.rng = random.Random(self.config.seed)
        self.feedback = BehaviorFeedback(self.config.reward)
        self.mutator = Mutator(
            adapter.space,
            adapter.make_invocation,
            adapter.make_state,
            self.rng,
            legal=adapter.legal,
        )
        self.synthesizer = Synthesizer(
            adapter.space, self.rng, use_smt=self.config.use_smt
        )
        self.oracle = Oracle(adapter)
        self.refiner = Refiner(adapter, self.config.refinement)
        self._explored_tags: dict[str, set[str]] = {}

    # run
    def run(self, initial: Sequence | None = None) -> CampaignResult:
        result = CampaignResult()
        start = initial or Sequence(scenario=self.adapter.scenario.name)
        seen: set[tuple[str, str]] = set()

        sampler = RondoMCMC(
            start,
            self._evaluate_for_feedback,
            self.mutator,
            self.feedback,
            self.config.sampler,
            synthesizer=self.synthesizer,
            goal_provider=self._unexplored_goals,
            legal=self.adapter.legal,
        )

        for round_index in range(self.config.exploration_rounds):
            record = sampler.step()
            result.rounds.append(record)
            if record.kind == "none":
                continue
            if round_index % max(1, self.config.check_every):
                continue
            oracle_result = self.oracle.check(record.sequence)
            result.oracle_checks += 1
            if oracle_result.candidate is None:
                continue
            result.candidates += 1
            name = oracle_result.classification.category.value
            result.candidates_by_class[name] = result.candidates_by_class.get(name, 0) + 1
            witness = self.refiner.confirm(oracle_result.candidate)
            if witness.is_confirmed:
                witness = self.refiner.minimize(witness)
                fingerprint = (
                    witness.witness_class.value,
                    witness.minimized_sequence.digest,
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                result.witnesses.append(witness)
                if len(result.witnesses) >= self.config.max_witnesses:
                    break

        result.feedback = self.feedback.describe()
        return result

    # sampling support
    def _evaluate_for_feedback(self, sequence: Sequence):
        """Behaviour feedback comes from the retained-compiler-state run."""
        return self.adapter.run(sequence, Mode.COMPILED_WITH_STATE)

    def _unexplored_goals(self, sequence: Sequence) -> tuple[GuardGoal, ...]:
        goals = self.adapter.guard_goals()
        if not goals:
            return ()
        unexplored = [goal for goal in goals if not self._tag_explored(goal)]
        return tuple(unexplored or goals)

    def _tag_explored(self, goal: GuardGoal) -> bool:
        tag = _goal_tag(goal)
        if tag is None or goal.op is None:
            return False
        explored = self._explored_tags.get(goal.op, set())
        for cell in self.feedback.seen_cells:
            if cell[0] == goal.op:
                explored.add(cell[2])
        self._explored_tags[goal.op] = explored
        return tag in explored


def _goal_tag(goal: GuardGoal) -> str | None:
    if goal.shape_dim_1 is not None:
        return "s1"
    if goal.shape_divisible_by == 16:
        return "tile16"
    if goal.shape_eq is not None:
        return "s1"
    return None
