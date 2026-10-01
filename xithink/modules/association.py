from __future__ import annotations

from collections import defaultdict
from xithink.models import Atom


class AssociativeNetwork:
    """Small spreading-activation network standing in for neural association."""

    def build(self, facts: list[Atom]) -> dict[str, dict[str, float]]:
        graph: dict[str, dict[str, float]] = defaultdict(dict)
        for f in facts:
            weight = max(0.05, min(1.0, f.confidence))
            graph[f.subject][f.object] = max(graph[f.subject].get(f.object, 0.0), weight)
            graph[f.object][f.subject] = max(graph[f.object].get(f.subject, 0.0), weight * 0.8)
        return graph

    def activate(
        self,
        seeds: list[str],
        facts: list[Atom],
        steps: int = 2,
        decay: float = 0.55,
        threshold: float = 0.08,
    ) -> dict[str, float]:
        graph = self.build(facts)
        activation: dict[str, float] = defaultdict(float)
        frontier: dict[str, float] = {s: 1.0 for s in seeds}
        for s in seeds:
            activation[s] = 1.0

        for _ in range(steps):
            nxt: dict[str, float] = defaultdict(float)
            for node, energy in frontier.items():
                for neighbor, weight in graph.get(node, {}).items():
                    spread = energy * weight * decay
                    if spread >= threshold:
                        nxt[neighbor] = max(nxt[neighbor], spread)
                        activation[neighbor] = max(activation[neighbor], spread)
            frontier = nxt
            if not frontier:
                break
        return dict(sorted(activation.items(), key=lambda kv: kv[1], reverse=True))
