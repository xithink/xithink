from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace

from xithink.models import Atom, Event, GraphEdge, Rule, clamp
from xithink.modules.memory import MemoryStore


class ConceptGraph:
    """Persistent weighted relation graph over canonical concept names.

    The graph is deliberately independent from the symbolic reasoner: the
    reasoner answers whether a rule can be proven, while this graph answers
    which concepts are connected, how strongly, and through which paths.
    """

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self._edges: dict[tuple[str, str, str, int, str], GraphEdge] = {}
        for edge in memory.load_graph_edges():
            self._edges[self._key(edge)] = edge

    @staticmethod
    def _key(edge: GraphEdge) -> tuple[str, str, str, int, str]:
        return (edge.source, edge.relation, edge.target, edge.polarity, edge.source_type)

    @staticmethod
    def _norm(name: str) -> str:
        return " ".join(name.strip().split())

    def add_edge(self, edge: GraphEdge, *, persist: bool = True) -> GraphEdge:
        if not edge.source or not edge.target or edge.source.startswith("?") or edge.target.startswith("?"):
            return edge
        edge = replace(
            edge,
            source=self._norm(edge.source),
            target=self._norm(edge.target),
            weight=clamp(edge.weight, 0.01, 1.0),
            polarity=1 if edge.polarity >= 0 else -1,
            evidence_count=max(1, edge.evidence_count),
        )
        key = self._key(edge)
        current = self._edges.get(key)
        if current is None:
            merged = edge
        else:
            merged = replace(
                current,
                weight=max(current.weight, edge.weight),
                evidence_count=current.evidence_count + edge.evidence_count,
            )
        self._edges[key] = merged
        if persist:
            self.memory.save_graph_edge(edge)
        return merged

    def ingest_fact(self, atom: Atom, source_type: str | None = None) -> GraphEdge:
        return self.add_edge(GraphEdge(
            source=atom.subject,
            relation=atom.predicate,
            target=atom.object,
            weight=atom.confidence,
            polarity=-1 if atom.negated else 1,
            source_type=source_type or ("inference" if atom.source.startswith("inference:") else "fact"),
        ))

    def ingest_rule(self, rule: Rule) -> None:
        # Only concrete rule atoms belong in the concept graph. Variable rules
        # are kept by the symbolic reasoner and become graph edges when inferred.
        atoms = (*rule.premises, rule.conclusion)
        for atom in atoms:
            if not atom.subject.startswith("?") and not atom.object.startswith("?"):
                self.ingest_fact(atom, source_type="rule")

    def ingest_event(self, event: Event) -> GraphEdge | None:
        if not event.object:
            return None
        return self.add_edge(GraphEdge(
            source=event.actor,
            relation=event.action,
            target=event.object,
            weight=event.confidence,
            polarity=-1 if event.negated else 1,
            source_type="event",
        ))

    def edges(self) -> list[GraphEdge]:
        return list(self._edges.values())

    def neighbors(self, node: str, *, include_negative: bool = True) -> list[GraphEdge]:
        node = self._norm(node)
        out: list[GraphEdge] = []
        for edge in self._edges.values():
            if not include_negative and edge.polarity < 0:
                continue
            if edge.source == node:
                out.append(edge)
            elif edge.target == node:
                out.append(GraphEdge(
                    source=node,
                    relation=f"逆:{edge.relation}",
                    target=edge.source,
                    weight=edge.weight * 0.82,
                    polarity=edge.polarity,
                    evidence_count=edge.evidence_count,
                    source_type=edge.source_type,
                ))
        return sorted(out, key=lambda e: (e.weight, e.evidence_count), reverse=True)

    def activate(
        self,
        seeds: list[str],
        *,
        max_depth: int = 3,
        decay: float = 0.62,
        threshold: float = 0.06,
    ) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        queue: deque[tuple[str, float, int]] = deque()
        for seed in dict.fromkeys(self._norm(x) for x in seeds if x.strip()):
            scores[seed] = 1.0
            queue.append((seed, 1.0, 0))

        best_seen: dict[tuple[str, int], float] = {}
        while queue:
            node, energy, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.neighbors(node, include_negative=True):
                polarity_penalty = 0.72 if edge.polarity < 0 else 1.0
                next_energy = energy * edge.weight * decay * polarity_penalty
                if next_energy < threshold:
                    continue
                key = (edge.target, depth + 1)
                if next_energy <= best_seen.get(key, 0.0):
                    continue
                best_seen[key] = next_energy
                scores[edge.target] = max(scores[edge.target], next_energy)
                queue.append((edge.target, next_energy, depth + 1))
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def strongest_path(self, source: str, target: str, max_depth: int = 4) -> list[GraphEdge]:
        """Return a high-product-weight path, bounded for predictable cost."""
        source, target = self._norm(source), self._norm(target)
        if source == target:
            return []
        queue: deque[tuple[str, list[GraphEdge], float]] = deque([(source, [], 1.0)])
        best: dict[tuple[str, int], float] = {(source, 0): 1.0}
        winning: tuple[list[GraphEdge], float] | None = None
        while queue:
            node, path, score = queue.popleft()
            if len(path) >= max_depth:
                continue
            for edge in self.neighbors(node):
                next_score = score * edge.weight * (0.75 if edge.polarity < 0 else 1.0)
                next_path = path + [edge]
                if edge.target == target:
                    if winning is None or next_score > winning[1]:
                        winning = (next_path, next_score)
                    continue
                key = (edge.target, len(next_path))
                if next_score <= best.get(key, 0.0):
                    continue
                best[key] = next_score
                queue.append((edge.target, next_path, next_score))
        return winning[0] if winning else []

    def opposite_edges(self, atom: Atom) -> list[GraphEdge]:
        polarity = 1 if atom.negated else -1
        return [
            e for e in self._edges.values()
            if e.source == atom.subject and e.relation == atom.predicate
            and e.target == atom.object and e.polarity == polarity
        ]

    def snapshot(self, limit: int = 40) -> dict:
        ranked = sorted(
            self._edges.values(),
            key=lambda e: (e.evidence_count, e.weight),
            reverse=True,
        )[:limit]
        nodes = {e.source for e in self._edges.values()} | {e.target for e in self._edges.values()}
        return {
            "nodes": len(nodes),
            "edges": len(self._edges),
            "top_edges": [
                {
                    "source": e.source,
                    "relation": e.relation,
                    "target": e.target,
                    "weight": e.weight,
                    "polarity": e.polarity,
                    "evidence_count": e.evidence_count,
                    "source_type": e.source_type,
                }
                for e in ranked
            ],
        }
