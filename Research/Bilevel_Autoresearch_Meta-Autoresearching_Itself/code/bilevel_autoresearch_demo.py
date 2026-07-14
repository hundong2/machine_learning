"""Toy bilevel autoresearch simulator.

This is a small teaching example inspired by arXiv:2603.23420v2.
It does not call an LLM or train a model. Instead, it simulates a biased
inner proposal loop and shows how a meta-level mechanism can alter the
future search distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class Config:
    lr: float = 1.0
    weight_decay: float = 1.0
    window_pattern: str = "baseline"
    total_batch_size: int = 1024
    head_dim: int = 64
    final_lr_frac: float = 1.0

    def apply(self, changes: dict[str, object]) -> "Config":
        return replace(self, **changes)


@dataclass
class Trial:
    iteration: int
    mechanism: str
    changes: dict[str, object]
    val_bpb: float
    best_delta: float
    accepted: bool
    frozen: tuple[str, ...]


@dataclass
class SearchState:
    current: Config
    best_loss: float
    baseline_loss: float
    history: list[Trial]
    frozen: set[str]
    rng: random.Random
    iteration: int
    seed: int


def stable_noise(config: Config, seed: int) -> float:
    """Tiny deterministic noise so repeats are similar but not identical."""

    key = repr((config, seed)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return (unit - 0.5) * 0.0015


def evaluate(config: Config, seed: int) -> float:
    """Hidden objective. Lower is better, like val_bpb in the paper."""

    loss = 1.100

    if config.weight_decay == 0.7:
        loss -= 0.009
    elif config.weight_decay == 0.4:
        loss -= 0.006

    if config.window_pattern == "SSSS":
        loss -= 0.006
    elif config.window_pattern == "SLSL":
        loss -= 0.003

    if config.lr == 0.8:
        loss -= 0.0015
    elif config.lr == 1.2:
        loss += 0.006

    if config.head_dim == 96:
        loss -= 0.0005
    elif config.head_dim == 128:
        loss += 0.004

    if config.final_lr_frac == 0.8:
        loss -= 0.0005

    if config.total_batch_size == 2048:
        loss += 0.018
    elif config.total_batch_size == 512:
        loss -= 0.030
    elif config.total_batch_size == 256:
        loss -= 0.037

    return loss + stable_noise(config, seed)


def change_key(changes: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(changes.items()))


def changed_params(history: Iterable[Trial]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trial in history:
        counts.update(trial.changes.keys())
    return counts


class ProposalMechanism:
    name = "base"

    def propose(self, state: SearchState) -> dict[str, object]:
        raise NotImplementedError


class DefaultPrior(ProposalMechanism):
    """A biased inner-loop prior that gets stuck in familiar changes."""

    name = "default-prior"

    def __init__(self) -> None:
        self.script = [
            {"total_batch_size": 2048},
            {"weight_decay": 0.7},
            {"window_pattern": "SSSS"},
            {"weight_decay": 0.7},
            {"window_pattern": "SSSS"},
            {"total_batch_size": 2048},
        ]
        self.fallback = [
            {"lr": 0.8},
            {"head_dim": 96},
            {"final_lr_frac": 0.8},
            {"lr": 1.2},
            {"head_dim": 128},
        ]

    def propose(self, state: SearchState) -> dict[str, object]:
        attempts = len(state.history)
        for offset in range(len(self.script)):
            candidate = self.script[(attempts + offset) % len(self.script)]
            if not any(param in state.frozen for param in candidate):
                return candidate

        for offset in range(len(self.fallback)):
            candidate = self.fallback[(attempts + offset) % len(self.fallback)]
            if not any(param in state.frozen for param in candidate):
                return candidate

        return {"lr": 1.0}


class TabuMechanism(ProposalMechanism):
    """Avoid recent failed regions and try unexplored high-impact directions."""

    name = "tabu-search"

    def __init__(self) -> None:
        self.plan = [
            {"total_batch_size": 512},
            {"total_batch_size": 256},
            {"lr": 0.8},
            {"final_lr_frac": 0.8},
            {"head_dim": 96},
            {"window_pattern": "SLSL"},
        ]

    def propose(self, state: SearchState) -> dict[str, object]:
        recent_failed = {
            change_key(trial.changes)
            for trial in state.history[-8:]
            if not trial.accepted
        }
        tried = {change_key(trial.changes) for trial in state.history}

        for candidate in self.plan:
            key = change_key(candidate)
            if key in recent_failed:
                continue
            if key in tried and state.current.apply(candidate) == state.current:
                continue
            return candidate

        return DefaultPrior().propose(state)


class OrthogonalMechanism(ProposalMechanism):
    """Force the search to cover dimensions the default prior avoids."""

    name = "orthogonal-exploration"

    values = {
        "total_batch_size": [512, 256],
        "lr": [0.8, 1.2],
        "head_dim": [96, 128],
        "final_lr_frac": [0.8],
        "window_pattern": ["SLSL", "SSSS"],
        "weight_decay": [0.4, 0.7],
    }

    def propose(self, state: SearchState) -> dict[str, object]:
        has_failed_increase = any(
            trial.changes.get("total_batch_size") == 2048 and not trial.accepted
            for trial in state.history
        )
        has_tried_decrease = any(
            trial.changes.get("total_batch_size") in {512, 256}
            for trial in state.history
        )
        if has_failed_increase and not has_tried_decrease:
            return {"total_batch_size": 512}

        counts = changed_params(state.history)
        ordered_params = sorted(self.values, key=lambda param: (counts[param], param))
        for param in ordered_params:
            for value in self.values[param]:
                candidate = {param: value}
                if state.current.apply(candidate) != state.current:
                    return candidate

        return DefaultPrior().propose(state)


class BanditMechanism(ProposalMechanism):
    """A lightweight UCB-style parameter selector.

    This mechanism is intentionally imperfect. It can spend too much budget on
    parameters that already looked promising and miss the hidden batch-size
    decrease, which mirrors the paper's warning that mechanism quality matters.
    """

    name = "multi-scale-bandit"

    values = {
        "lr": [0.8, 1.2],
        "weight_decay": [0.7, 0.4],
        "window_pattern": ["SSSS", "SLSL"],
        "head_dim": [96, 128],
        "final_lr_frac": [0.8],
        "total_batch_size": [512, 256],
    }
    priors = {
        "lr": 0.018,
        "weight_decay": 0.025,
        "window_pattern": 0.020,
        "head_dim": 0.010,
        "final_lr_frac": 0.010,
        "total_batch_size": -0.010,
    }

    def propose(self, state: SearchState) -> dict[str, object]:
        counts: Counter[str] = Counter()
        rewards: defaultdict[str, float] = defaultdict(float)

        previous_best = state.baseline_loss
        for trial in state.history:
            params = list(trial.changes)
            reward = max(0.0, previous_best - trial.val_bpb) if trial.accepted else 0.0
            for param in params:
                counts[param] += 1
                rewards[param] += reward
            if trial.accepted:
                previous_best = min(previous_best, trial.val_bpb)

        total = max(1, sum(counts.values()))
        scores: dict[str, float] = {}
        for param in self.values:
            if counts[param] == 0:
                scores[param] = self.priors[param] + 0.02
            else:
                mean_reward = rewards[param] / counts[param]
                bonus = math.sqrt(math.log(total + 1) / counts[param]) * 0.01
                scores[param] = self.priors[param] + mean_reward + bonus

        for param, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            if param in state.frozen and param != "total_batch_size":
                continue
            for value in self.values[param]:
                candidate = {param: value}
                if state.current.apply(candidate) != state.current:
                    return candidate

        return DefaultPrior().propose(state)


def update_strategy(history: list[Trial], frozen: set[str]) -> set[str]:
    """Level 1.5: freeze parameters that recently failed."""

    updated = set(frozen)
    recent = history[-5:]
    failed_counts: Counter[str] = Counter()
    for trial in recent:
        if not trial.accepted:
            failed_counts.update(trial.changes.keys())

    for param, count in failed_counts.items():
        if count >= 1:
            updated.add(param)

    return updated


def choose_level2_mechanism(history: list[Trial], rng: random.Random, guided: bool) -> ProposalMechanism:
    """Level 2: choose a new mechanism after inspecting the trace."""

    recent = history[-10:]
    repeated_failures = sum(1 for trial in recent if not trial.accepted)
    repeated_params = changed_params(recent).most_common(2)

    if guided and repeated_failures >= 5:
        if repeated_params and repeated_params[0][0] in {"weight_decay", "window_pattern"}:
            return TabuMechanism()
        return OrthogonalMechanism()

    if guided:
        return rng.choice([TabuMechanism(), OrthogonalMechanism(), BanditMechanism()])

    return rng.choice([TabuMechanism(), BanditMechanism(), OrthogonalMechanism(), DefaultPrior()])


def run_group(group: str, seed: int, iterations: int) -> tuple[float, list[Trial], list[str]]:
    use_strategy = group in {"B", "C"}
    use_level2 = group in {"C", "D"}
    guided_level2 = group == "C"

    rng = random.Random(seed)
    current = Config()
    baseline_loss = evaluate(current, seed)
    best_loss = baseline_loss
    frozen: set[str] = set()
    history: list[Trial] = []
    level2_events: list[str] = []
    mechanism: ProposalMechanism = DefaultPrior()

    for iteration in range(1, iterations + 1):
        if use_level2 and iteration in {11, 21}:
            mechanism = choose_level2_mechanism(history, rng, guided=guided_level2)
            level2_events.append(f"iter {iteration}: activated {mechanism.name}")
            if isinstance(mechanism, (TabuMechanism, OrthogonalMechanism)):
                frozen.discard("total_batch_size")

        state = SearchState(
            current=current,
            best_loss=best_loss,
            baseline_loss=baseline_loss,
            history=history,
            frozen=frozen,
            rng=rng,
            iteration=iteration,
            seed=seed,
        )
        changes = mechanism.propose(state)
        candidate = current.apply(changes)
        val_bpb = evaluate(candidate, seed)
        accepted = val_bpb < best_loss - 1e-12
        if accepted:
            current = candidate
            best_loss = val_bpb

        history.append(
            Trial(
                iteration=iteration,
                mechanism=mechanism.name,
                changes=changes,
                val_bpb=val_bpb,
                best_delta=best_loss - baseline_loss,
                accepted=accepted,
                frozen=tuple(sorted(frozen)),
            )
        )

        if use_strategy and iteration % 5 == 0:
            frozen = update_strategy(history, frozen)

    return best_loss - baseline_loss, history, level2_events


def format_change(changes: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(changes.items()))


def print_trace(group: str, history: list[Trial], events: list[str]) -> None:
    print()
    print(f"Trace for Group {group}")
    if events:
        print("Level 2 events:")
        for event in events:
            print(f"  - {event}")
    print()
    print("iter  mechanism              accepted  best_delta  change")
    print("----  ---------------------  --------  ----------  ---------------------------")
    for trial in history:
        accepted = "yes" if trial.accepted else "no"
        print(
            f"{trial.iteration:>4}  "
            f"{trial.mechanism:<21}  "
            f"{accepted:<8}  "
            f"{trial.best_delta:>10.4f}  "
            f"{format_change(trial.changes)}"
        )


def summarize(groups: list[str], repeats: int, iterations: int, seed: int) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    traces: dict[str, tuple[list[Trial], list[str]]] = {}

    for group in groups:
        deltas: list[float] = []
        for repeat in range(repeats):
            run_seed = seed + repeat * 101 + ord(group)
            best_delta, history, events = run_group(group, run_seed, iterations)
            deltas.append(best_delta)
            if repeat == 0:
                traces[group] = (history, events)
        results[group] = deltas

    print("Toy Bilevel Autoresearch Results")
    print("Lower best_delta is better, matching Delta val_bpb convention.")
    print()
    print("group  mean_delta  std_delta   repeats")
    print("-----  ----------  ---------   -------")
    for group in groups:
        deltas = results[group]
        mean = statistics.mean(deltas)
        std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        print(f"{group:<5}  {mean:>10.4f}  {std:>9.4f}   {len(deltas):>7}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", choices=["A", "B", "C", "D"], help="Print a detailed trace for one group.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args.groups, args.repeats, args.iterations, args.seed)

    if args.trace:
        run_seed = args.seed + ord(args.trace)
        _best_delta, history, events = run_group(args.trace, run_seed, args.iterations)
        print_trace(args.trace, history, events)


if __name__ == "__main__":
    main()
