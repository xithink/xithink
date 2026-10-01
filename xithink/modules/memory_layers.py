from __future__ import annotations

import math
import time
from itertools import combinations

from xithink.models import Atom, Experience, MemoryMatch, Rule, SubconsciousTrace
from xithink.modules.embedding_memory import EmbeddingMemory
from xithink.modules.memory import MemoryStore


class LongTermMemory:
    """Verified/accepted symbolic memory.

    Only this layer is allowed to become premises for SymbolicReasoner. Other
    memory layers can alter retrieval priority, but cannot silently create a
    fact or a rule.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def remember_fact(self, atom: Atom) -> None:
        self.store.save_fact(atom)

    def remember_rule(self, rule: Rule) -> None:
        self.store.save_rule(rule)

    def facts(self) -> list[Atom]:
        return self.store.load_facts()

    def rules(self) -> list[Rule]:
        return self.store.load_rules()

    def snapshot(self) -> dict:
        return {
            "facts": len(self.store.load_facts()),
            "rules": len(self.store.load_rules()),
            "authority": "symbolic-premise",
        }


class ExperienceMemory:
    """Episodic interaction/outcome memory.

    Experience recall can suggest relevant cases but is never inserted into the
    symbolic fact base automatically.
    """

    def __init__(self, store: MemoryStore, embedding: EmbeddingMemory) -> None:
        self.store = store
        self.embedding = embedding

    def remember(self, experience: Experience) -> Experience:
        self.store.add_experience(experience)
        self.embedding.index(
            f"experience:{experience.id}",
            "experience",
            experience.content,
            concepts=experience.concepts,
            metadata={
                "kind": experience.kind,
                "outcome": experience.outcome,
                "reward": experience.reward,
                "confidence": experience.confidence,
                "source": experience.source,
                "context": experience.context,
            },
            created_at=experience.created_at,
        )
        return experience

    def recall(
        self,
        text: str,
        *,
        concepts: list[str] | None = None,
        top_k: int = 5,
        min_score: float = 0.12,
    ) -> list[MemoryMatch]:
        return self.embedding.query(
            text,
            concepts=concepts or [],
            top_k=top_k,
            min_score=min_score,
            kinds={"experience"},
        )

    def count(self) -> int:
        return self.store.experience_count()

    def recent(self, limit: int = 10) -> list[Experience]:
        return self.store.recent_experiences(limit)


class SubconsciousMemory:
    """Implicit association/priority memory.

    This layer deliberately has no API for returning facts. It only emits a
    bounded activation bias used to decide which concepts are explored first.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        half_life_seconds: float = 30 * 24 * 3600,
        max_activation: float = 0.42,
    ) -> None:
        self.store = store
        self.half_life_seconds = max(1.0, half_life_seconds)
        self.max_activation = max(0.05, min(0.75, max_activation))
        self._cache: dict[tuple[str, str], SubconsciousTrace] = {
            (t.cue, t.association): t for t in self.store.load_subconscious_traces()
        }

    @staticmethod
    def _clean(items: list[str]) -> list[str]:
        return list(dict.fromkeys(x.strip() for x in items if x and x.strip()))

    def _decay(self, trace: SubconsciousTrace, now: float) -> float:
        age = max(0.0, now - trace.last_activated)
        return trace.strength * math.pow(0.5, age / self.half_life_seconds)

    def reinforce(
        self,
        concepts: list[str],
        *,
        reward: float = 0.0,
        source: str = "experience",
        learning_rate: float = 0.10,
    ) -> None:
        concepts = self._clean(concepts)[:12]
        if len(concepts) < 2:
            return
        now = time.time()
        reward = max(-1.0, min(1.0, reward))
        changed: list[SubconsciousTrace] = []
        for left, right in combinations(concepts, 2):
            for cue, association in ((left, right), (right, left)):
                key = (cue, association)
                previous = self._cache.get(key)
                base = self._decay(previous, now) if previous else 0.0
                strength = min(1.0, base + learning_rate * (1.0 - base))
                valence = reward if previous is None else max(-1.0, min(1.0, previous.valence * 0.8 + reward * 0.2))
                trace = SubconsciousTrace(
                    cue=cue,
                    association=association,
                    strength=strength,
                    valence=valence,
                    exposures=(previous.exposures + 1) if previous else 1,
                    source=source,
                    last_activated=now,
                )
                self._cache[key] = trace
                changed.append(trace)
        self.store.save_subconscious_traces(changed)

    def activate(self, cues: list[str], *, limit: int = 12) -> dict[str, float]:
        now = time.time()
        scores: dict[str, float] = {}
        for cue in self._clean(cues):
            for (stored_cue, association), trace in self._cache.items():
                if stored_cue != cue:
                    continue
                strength = self._decay(trace, now)
                # Valence changes priority slightly, never truth value.
                valence_factor = 1.0 + 0.12 * trace.valence
                score = min(self.max_activation, strength * 0.48 * valence_factor)
                if score > scores.get(association, 0.0):
                    scores[association] = score
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit])

    def count(self) -> int:
        return self.store.subconscious_count()

    def snapshot(self, cues: list[str] | None = None, limit: int = 12) -> dict:
        activation = self.activate(cues or [], limit=limit) if cues else {}
        return {
            "traces": self.count(),
            "activation": list(activation.items()),
            "authority": "retrieval-priority-only",
        }
