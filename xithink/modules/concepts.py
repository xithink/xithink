from __future__ import annotations

import re
from xithink.models import Concept
from xithink.modules.memory import MemoryStore


class ConceptRegistry:
    """Canonical concept registry with stable IDs and alias support."""

    TYPE_HINTS = {
        "人": "entity",
        "机器人": "entity",
        "AI": "entity",
        "自由": "abstract",
        "安全": "abstract",
        "风险": "abstract",
        "选择": "abstract",
        "记忆": "abstract",
        "身份": "abstract",
    }

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self._by_name: dict[str, Concept] = {}
        self._alias_to_name: dict[str, str] = {}
        for concept in memory.load_concepts():
            self._register_local(concept)

    @staticmethod
    def normalize(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip(" \t\r\n,，。.!！?？:：;；'\"")).strip()

    def _register_local(self, concept: Concept) -> Concept:
        self._by_name[concept.canonical_name] = concept
        self._alias_to_name[concept.canonical_name] = concept.canonical_name
        for alias in concept.aliases:
            self._alias_to_name[alias] = concept.canonical_name
        return concept

    def get_or_create(
        self,
        name: str,
        *,
        kind: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict | None = None,
    ) -> Concept | None:
        name = self.normalize(name)
        if not name:
            return None
        canonical = self._alias_to_name.get(name, name)
        existing = self._by_name.get(canonical)
        if existing:
            changed = False
            for alias in aliases or []:
                alias = self.normalize(alias)
                if alias and alias not in existing.aliases and alias != existing.canonical_name:
                    existing.aliases.append(alias)
                    self._alias_to_name[alias] = existing.canonical_name
                    changed = True
            if attributes:
                for key, value in attributes.items():
                    if existing.attributes.get(key) != value:
                        existing.attributes[key] = value
                        changed = True
            if changed:
                self.memory.save_concept(existing)
            return existing

        concept = Concept(
            canonical_name=canonical,
            kind=kind or self.TYPE_HINTS.get(canonical, "abstract"),
            aliases=aliases or [],
            attributes=attributes or {},
        )
        self._register_local(concept)
        self.memory.save_concept(concept)
        return concept

    def resolve(self, name: str) -> Concept | None:
        name = self.normalize(name)
        canonical = self._alias_to_name.get(name, name)
        return self._by_name.get(canonical)

    def all(self) -> list[Concept]:
        return sorted(self._by_name.values(), key=lambda c: c.canonical_name)
