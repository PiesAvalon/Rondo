"""The adaptive Metropolis-Hastings sampler of the exploration stage."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence as TypingSequence

from .domain import GuardGoal, Sequence, Trace
from .feedback import BehaviorFeedback
from .generation import Mutator, Proposal
from .synthesis import Synthesizer

__all__ = ["Archive", "RondoMCMC", "RoundRecord", "SamplerConfig"]


@dataclass(frozen=True)
class SamplerConfig:
    max_length: int = 8
    min_length: int = 1
    beta: float = 1.5
    seed: int = 0
    generator_weights: tuple[tuple[str, float], ...] = (
        ("mutation", 1.0),
        ("synthesis", 0.5),
    )
    acceptance: str = "mh"  # "mh" | "sa" | "greedy"
    sa_temperature: float = 1.0
    sa_decay: float = 0.995
    restart_after: int = 60
    restart_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.min_length < 0:
            raise ValueError("min_length must be >= 0")
        if self.acceptance not in ("mh", "sa", "greedy"):
            raise ValueError("acceptance must be 'mh', 'sa' or 'greedy'")


@dataclass(frozen=True)
class RoundRecord:
    """One decision of the chain, kept for inspection (not for measurement)."""

    round_index: int
    sequence: Sequence
    trace: Trace
    kind: str
    generator: str
    operator: str
    accepted: bool
    reward: float
    delta_reward: float
    q_forward: float
    q_reverse: float
    new_keys: int
    current_length: int


class Archive:
    """Digest-deduplicated store of sequences that produced a new key."""

    def __init__(self) -> None:
        self.entries: list[Sequence] = []
        self._digests: set[str] = set()

    def add(self, sequence: Sequence) -> bool:
        if sequence.digest in self._digests:
            return False
        self._digests.add(sequence.digest)
        self.entries.append(sequence)
        return True

    def choose(self, rng: random.Random) -> Sequence:
        if not self.entries:
            raise IndexError("archive is empty")
        return self.entries[rng.randrange(len(self.entries))]

    def __len__(self) -> int:
        return len(self.entries)


class RondoMCMC:
    """The exploration chain: propose, run, price, accept."""

    def __init__(
        self,
        initial: Sequence,
        evaluate: Callable[[Sequence], Trace],
        mutator: Mutator,
        feedback: BehaviorFeedback,
        config: SamplerConfig | None = None,
        *,
        synthesizer: Synthesizer | None = None,
        goal_provider: Callable[[Sequence], TypingSequence[GuardGoal]] | None = None,
        legal: Callable[[Sequence], bool] | None = None,
    ) -> None:
        self.config = config or SamplerConfig()
        self.rng = random.Random(self.config.seed)
        self.current = initial
        self.evaluate = evaluate
        self.mutator = mutator
        self.synthesizer = synthesizer
        self.goal_provider = goal_provider
        self.feedback = feedback
        self.legal = legal
        self.archive = Archive()
        self.rounds: list[RoundRecord] = []
        self.accepted = 0
        self.evaluated = 0
        self._temperature = self.config.sa_temperature
        self._stalls = 0
        self.current_trace = self.evaluate(self.current)
        self.feedback.observe(self.current_trace)
        self.feedback.freeze()

    # one round
    def step(self) -> RoundRecord:
        index = len(self.rounds)
        self.feedback.freeze()
        goals = tuple(self.goal_provider(self.current)) if self.goal_provider else ()
        proposal = self._propose(goals)
        if proposal is None:
            record = RoundRecord(
                round_index=index,
                sequence=self.current,
                trace=self.current_trace,
                kind="none",
                generator="none",
                operator="none",
                accepted=False,
                reward=self.feedback.score(self.current_trace),
                delta_reward=0.0,
                q_forward=0.0,
                q_reverse=0.0,
                new_keys=0,
                current_length=self.current.length,
            )
            self.rounds.append(record)
            return record

        proposed_sequence = proposal.sequence
        if proposed_sequence.length > self.config.max_length:
            return self._reject(index, proposal, self.current_trace, "over-length")

        proposed_trace = self.evaluate(proposed_sequence)
        self.evaluated += 1

        current_score = self.feedback.score(self.current_trace)
        proposed_score = self.feedback.score(proposed_trace)
        delta = proposed_score - current_score

        q_forward = self._generator_mass(proposal.generator) * proposal.forward_probability
        q_reverse = self._reverse_probability(proposed_sequence, goals)
        correction = q_reverse / q_forward if q_forward > 0 else 0.0

        accepted = self._accept(delta, correction)
        if accepted:
            self.current = proposed_sequence
            self.current_trace = proposed_trace
            self.accepted += 1
            self._stalls = 0
        else:
            self._stalls += 1
            if self._stalls >= self.config.restart_after and len(self.archive):
                self.current = self.archive.choose(self.rng)
                self.current_trace = self.evaluate(self.current)
                self.feedback.observe(self.current_trace)
                self._stalls = 0

        new_keys = self.feedback.novel_keys(proposed_trace)
        self.feedback.observe(proposed_trace)
        if new_keys:
            self.archive.add(proposed_sequence)

        record = RoundRecord(
            round_index=index,
            sequence=proposed_sequence,
            trace=proposed_trace,
            kind=proposal.kind,
            generator=proposal.generator,
            operator=proposal.operator,
            accepted=accepted,
            reward=proposed_score,
            delta_reward=delta,
            q_forward=q_forward,
            q_reverse=q_reverse,
            new_keys=len(new_keys),
            current_length=proposed_sequence.length,
        )
        self.rounds.append(record)
        return record

    def run(self, rounds: int) -> list[RoundRecord]:
        if rounds < 0:
            raise ValueError("rounds must be non-negative")
        for _ in range(rounds):
            self.step()
        return self.rounds

    # proposal mixture
    def _enabled_generators(self) -> list[tuple[str, float]]:
        enabled: list[tuple[str, float]] = []
        for name, weight in self.config.generator_weights:
            if weight <= 0:
                continue
            if name == "synthesis" and (self.synthesizer is None or self.goal_provider is None):
                continue
            enabled.append((name, weight))
        return enabled or [("mutation", 1.0)]

    def _generator_mass(self, name: str) -> float:
        generators = self._enabled_generators()
        total = sum(weight for _, weight in generators)
        for candidate, weight in generators:
            if candidate == name:
                return weight / total
        return 0.0

    def _propose(self, goals: TypingSequence[GuardGoal]) -> Proposal | None:
        generators = self._enabled_generators()
        total = sum(weight for _, weight in generators)
        threshold = self.rng.random() * total
        cumulative = 0.0
        chosen = generators[-1][0]
        for name, weight in generators:
            cumulative += weight
            if threshold <= cumulative:
                chosen = name
                break

        if chosen == "synthesis" and self.synthesizer is not None:
            proposal = self.synthesizer.propose(
                self.current, goals, make_invocation=self.mutator.make_invocation,
                legal=self.legal,
            )
            if proposal is not None:
                return proposal
        return self.mutator.propose(self.current)

    def _reverse_probability(
        self, proposed: Sequence, goals: TypingSequence[GuardGoal]
    ) -> float:
        reverse = self._generator_mass("mutation") * self.mutator.probability(
            proposed, self.current
        )
        if self.synthesizer is not None and goals:
            reverse += self._generator_mass("synthesis") * self.synthesizer.probability(
                proposed, self.current, goals
            )
        return reverse

    # acceptance
    # alpha = min{1, exp(beta * dS) * q(X|X') / q(X'|X)}; counts frozen before the call
    def _accept(self, delta: float, correction: float) -> bool:
        if self.config.acceptance == "greedy":
            if delta > 1e-12:
                return True
            if abs(delta) <= 1e-12:
                return self.rng.random() < self.config.restart_probability
            return False

        if self.config.acceptance == "sa":
            scale = max(1e-3, self._temperature)
            delta = delta / scale
            self._temperature = max(1e-3, self._temperature * self.config.sa_decay)

        if not self.feedback.config.enabled:
            ratio = correction
        else:
            ratio = math.exp(self.config.beta * delta) * correction
        return self.rng.random() < min(1.0, ratio)

    # helpers
    def _reject(
        self, index: int, proposal: Proposal, trace: Trace, reason: str
    ) -> RoundRecord:
        record = RoundRecord(
            round_index=index,
            sequence=proposal.sequence,
            trace=trace,
            kind=reason,
            generator=proposal.generator,
            operator=proposal.operator,
            accepted=False,
            reward=self.feedback.score(trace),
            delta_reward=0.0,
            q_forward=0.0,
            q_reverse=0.0,
            new_keys=0,
            current_length=self.current.length,
        )
        self.rounds.append(record)
        return record

    def summary(self) -> Mapping[str, float]:
        return {
            "rounds": len(self.rounds),
            "accepted": self.accepted,
            "evaluated": self.evaluated,
            "archive": len(self.archive),
            "distinct_keys": len(self.feedback.counts),
        }
