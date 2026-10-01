from __future__ import annotations

from xithink.models import Atom, Rule, WorldviewBelief
from xithink.modules.memory import MemoryStore
from xithink.modules.symbolic import SymbolicReasoner


class WorldviewModel:
    """Environment-scoped experiential belief model.

    It records what this particular XThink instance has repeatedly observed.
    Scoped beliefs may participate in a contextual symbolic reasoner, but remain
    provenance-labelled and are never silently promoted into universal facts.
    """

    def __init__(self, store: MemoryStore, *, min_confidence: float = 0.58) -> None:
        self.store = store
        self.min_confidence = min_confidence

    @staticmethod
    def normalize_environment(environment: str | None) -> str:
        return (environment or "default").strip() or "default"

    def observe(self, atoms: list[Atom], environment: str) -> None:
        env = self.normalize_environment(environment)
        self.store.save_worldview_observations([
            (atom, env, max(0.05, atom.confidence)) for atom in atoms
        ])

    def beliefs(self, environment: str, *, min_confidence: float | None = None) -> list[WorldviewBelief]:
        env = self.normalize_environment(environment)
        threshold = self.min_confidence if min_confidence is None else min_confidence
        return [b for b in self.store.load_worldview_beliefs(env) if b.confidence >= threshold]

    def facts(self, environment: str, *, min_confidence: float | None = None) -> list[Atom]:
        # Only the dominant polarity clears > .5 confidence; ties are excluded.
        return [b.atom for b in self.beliefs(environment, min_confidence=min_confidence)]

    def build_reasoner(
        self,
        *,
        base_facts: list[Atom],
        base_rules: list[Rule],
        learned_rules: list[Rule],
        environment: str,
        current_facts: list[Atom] | None = None,
    ) -> SymbolicReasoner:
        reasoner = SymbolicReasoner()
        for atom in base_facts:
            if not atom.source.startswith("inference:"):
                reasoner.add_fact(atom)
        for atom in self.facts(environment):
            reasoner.add_fact(atom)
        for atom in current_facts or []:
            reasoner.add_fact(Atom(
                atom.subject, atom.predicate, atom.object, atom.negated,
                atom.confidence, f"worldview:{self.normalize_environment(environment)}:current",
            ))
        for rule in [*base_rules, *learned_rules]:
            reasoner.add_rule(rule)
        reasoner.infer()
        return reasoner

    def relevant_environments(self, subject: str, predicate: str, object: str) -> list[str]:
        envs: list[str] = []
        for belief in self.store.load_worldview_beliefs():
            atom = belief.atom
            if atom.subject == subject and atom.predicate == predicate and atom.object == object:
                if belief.environment not in envs:
                    envs.append(belief.environment)
        return envs

    def snapshot(self, environment: str | None = None, limit: int = 30) -> dict:
        beliefs = self.store.load_worldview_beliefs(self.normalize_environment(environment)) if environment else self.store.load_worldview_beliefs()
        return {
            "environment": self.normalize_environment(environment) if environment else None,
            "beliefs": [
                {
                    "atom": b.atom.render(), "environment": b.environment,
                    "support": b.support, "opposition": b.opposition,
                    "exposures": b.exposures, "confidence": b.confidence,
                }
                for b in beliefs[:limit]
            ],
            "authority": "environment-scoped-symbolic-evidence",
        }
