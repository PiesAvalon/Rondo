"""Demonstration entry point (not an experiment)."""

from __future__ import annotations

import argparse
from typing import Sequence as TypingSequence

from .adapters.reference import CompilerDefectProfile, ReferenceAdapter
from .domain import Mode, Scenario, Sequence, StateOperation
from .engine import Rondo, RondoConfig
from .oracle import Oracle
from .refinement import Refiner
from .sampler import SamplerConfig
from .synthesis import Z3_AVAILABLE

__all__ = ["main"]


def _reference_adapter(profile: CompilerDefectProfile | None = None) -> ReferenceAdapter:
    scenario = Scenario(
        name="reference-wx",
        framework="reference",
        observable_state_fields=("weight",),
        max_sequence_length=8,
    )
    return ReferenceAdapter(scenario=scenario, profile=profile or CompilerDefectProfile())


def _wx_sequence(adapter: ReferenceAdapter) -> Sequence:
    """``A`` (fold a parameter), ``U: w←4``, then ``B = return(A)``."""
    sequence = Sequence(scenario=adapter.scenario.name)
    a = adapter.make_invocation(
        "a", adapter.space.default_spec().with_fields(operator="mul", shape=(1, 4))
    )
    update = adapter.make_state("u", "update", {"weight": 4.0})
    b = adapter.make_invocation(
        "b", adapter.space.default_spec().with_fields(operator="mul", shape=(1, 4)), "a"
    )
    return sequence.append(a).append(update).append(b)


def cmd_method() -> int:
    print("Rondo = three modules over one representation: test case = <model, invocation sequence>")
    print()
    print("  1. Test-space exploration")
    print("     - 13 mutation operators + SMT synthesis of guard-satisfying inputs")
    print("     - behavior keys (operator, lowering, specialization, dtype class, decision)")
    print("     - adaptive Metropolis-Hastings on S(.'X') = sum_k 1/(1+c(k))")
    print()
    print("  2. Test oracle")
    print("     - Without compilation / Compiled without state / Compiled with state")
    print("     - structured comparison: NaN, infinities, signed zero, subnormals, bits")
    print("     - six independent semantic checks")
    print()
    print("  3. Stateful candidate refinement")
    print("     - repeat the exposing invocation with compiler state retained vs reset")
    print("     - classify: stateful / point-wise / history-exposed")
    print("     - minimize while preserving class, exposing invocation and symptoms")
    print()
    print(f"  z3 available for SMT synthesis: {Z3_AVAILABLE}")
    return 0


def cmd_witness() -> int:
    adapter = _reference_adapter()
    sequence = _wx_sequence(adapter)
    oracle = Oracle(adapter)
    result = oracle.check(sequence)

    print("sequence: A(w=2 compiled) -> U:w<-4 -> B=return(A)")
    print()
    for mode in (Mode.WITHOUT_COMPILATION, Mode.COMPILED_WITHOUT_STATE, Mode.COMPILED_WITH_STATE):
        trace = adapter.run(sequence, mode)
        b = trace.observation("b")
        print(f"  {mode.value:<24} B = {list(b.output.data)}")
    print()
    print(f"oracle class: {result.classification.category.value}")
    print(f"reason      : {result.classification.reason}")

    if result.candidate is not None:
        refiner = Refiner(adapter)
        witness = refiner.confirm(result.candidate)
        print()
        print("controlled replay of B from the same program state (w=4):")
        print(f"  compiler state retained : wrong = {not witness.kept_correct}")
        print(f"  compiler state reset    : correct = {witness.reset_correct}")
        print(f"  standalone from s_0     : reproduces = {witness.standalone_reproduces}")
        print(f"  witness class           : {witness.witness_class.value}")
    return 0


def cmd_search(rounds: int, seed: int) -> int:
    adapter = _reference_adapter()
    config = RondoConfig(
        seed=seed,
        exploration_rounds=rounds,
        sampler=SamplerConfig(seed=seed, max_length=6, beta=1.5),
    )
    rondo = Rondo(adapter, config)
    result = rondo.run()
    print(result.summarize())
    print()
    for witness in result.witnesses:
        print("witness:", witness.summarize())
        print("  sequence:", _format(witness.minimized_sequence))
    return 0


def _format(sequence: Sequence) -> str:
    parts = []
    for step in sequence.steps:
        if isinstance(step, StateOperation):
            parts.append(f"{step.operation}({dict(step.parameters)})")
        else:
            spec = step.spec
            label = f"{spec.operator}:{spec.shape}:{spec.dtype}:{spec.value_class}"
            if step.repeat_of:
                label += f"=return({step.repeat_of})"
            parts.append(label)
    return " -> ".join(parts) or "<empty>"


def main(argv: TypingSequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rondo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("method", help="describe the method")
    sub.add_parser("witness", help="classify the w*x stateful witness by hand")
    search = sub.add_parser("search", help="run the search on the reference backend")
    search.add_argument("--rounds", type=int, default=120)
    search.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "method":
        return cmd_method()
    if args.command == "witness":
        return cmd_witness()
    if args.command == "search":
        return cmd_search(args.rounds, args.seed)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
