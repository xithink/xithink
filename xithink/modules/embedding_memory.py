from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Iterable

from xithink.models import Atom, Event, MemoryMatch, Thought
from xithink.modules.memory import MemoryStore


SparseVector = dict[int, float]


class HashEmbedding:
    """Dependency-free sparse embedding for the 0.3 local core.

    Chinese characters, character bigrams, ASCII words and explicit concept
    tokens are feature-hashed into a signed vector. Only non-zero dimensions
    are stored. This is not a replacement for a trained semantic encoder, but
    it provides deterministic offline retrieval and a stable adapter boundary.
    """

    def __init__(self, dims: int = 384) -> None:
        if dims < 64:
            raise ValueError("dims must be >= 64")
        self.dims = dims

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = text.strip().lower()
        chars = [c for c in text if not c.isspace() and c not in "，。！？；：,.!?;:'\"()[]{}"]
        tokens: list[str] = []
        tokens.extend(chars)
        tokens.extend("".join(chars[i:i+2]) for i in range(max(0, len(chars) - 1)))
        tokens.extend(re.findall(r"[a-z0-9_\-]{2,}", text))
        return tokens

    def encode(self, text: str, extra_tokens: Iterable[str] = ()) -> SparseVector:
        vector: SparseVector = {}
        tokens = self._tokens(text) + [f"concept:{x.strip().lower()}" for x in extra_tokens if x.strip()]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(digest[:8], "big") % self.dims
            sign = 1.0 if digest[8] & 1 else -1.0
            weight = 1.35 if token.startswith("concept:") or len(token) > 1 else 1.0
            vector[idx] = vector.get(idx, 0.0) + sign * weight
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if not norm:
            return {}
        return {idx: value / norm for idx, value in vector.items() if value}

    @staticmethod
    def _as_sparse(vector: SparseVector | list[float] | dict[str, float]) -> SparseVector:
        if isinstance(vector, list):
            return {i: float(v) for i, v in enumerate(vector) if v}
        return {int(k): float(v) for k, v in vector.items() if v}

    @classmethod
    def cosine(cls, a: SparseVector | list[float], b: SparseVector | list[float] | dict[str, float]) -> float:
        aa = cls._as_sparse(a)
        bb = cls._as_sparse(b)
        if not aa or not bb:
            return 0.0
        if len(aa) > len(bb):
            aa, bb = bb, aa
        return max(0.0, min(1.0, sum(value * bb.get(idx, 0.0) for idx, value in aa.items())))


class EmbeddingMemory:
    def __init__(self, memory: MemoryStore, dims: int = 384) -> None:
        self.memory = memory
        self.encoder = HashEmbedding(dims=dims)
        self._entries: dict[str, dict] = {
            row["item_id"]: row for row in self.memory.load_embeddings()
        }

    def _refresh_if_needed(self) -> None:
        # Several WebSocket sessions may own separate engines over one DB. A
        # count change cheaply tells a session that another writer added items.
        if self.memory.embedding_count() != len(self._entries):
            self._entries = {row["item_id"]: row for row in self.memory.load_embeddings()}

    def index(
        self,
        item_id: str,
        kind: str,
        text: str,
        *,
        concepts: list[str] | None = None,
        metadata: dict | None = None,
        created_at: float | None = None,
    ) -> None:
        concepts = list(dict.fromkeys(concepts or []))
        vector = self.encoder.encode(text, concepts)
        created = created_at or time.time()
        meta = metadata or {}
        self.memory.save_embedding(
            item_id=item_id,
            kind=kind,
            text=text,
            vector=vector,
            concepts=concepts,
            metadata=meta,
            created_at=created,
        )
        self._entries[item_id] = {
            "item_id": item_id,
            "kind": kind,
            "text": text,
            "vector": vector,
            "concepts": concepts,
            "metadata": meta,
            "created_at": created,
        }

    def index_fact(self, atom: Atom) -> None:
        item_id = "fact:" + hashlib.sha1(repr(atom.key()).encode("utf-8")).hexdigest()[:20]
        self.index(
            item_id, "fact", atom.render(), concepts=[atom.subject, atom.object],
            metadata={"confidence": atom.confidence, "source": atom.source, "negated": atom.negated},
        )

    def index_event(self, event: Event) -> None:
        concepts = [event.actor] + ([event.object] if event.object else [])
        self.index(
            f"event:{event.id}", "event", event.render(), concepts=concepts,
            metadata={"confidence": event.confidence, "source": event.source, "modality": event.modality},
            created_at=event.created_at,
        )

    def index_thought(self, thought: Thought) -> None:
        self.index(
            f"thought:{thought.id}", "thought", thought.content, concepts=thought.concepts,
            metadata={"confidence": thought.confidence, "cycle": thought.cycle, "kind": thought.kind},
            created_at=thought.created_at,
        )

    def query(
        self,
        text: str,
        *,
        concepts: list[str] | None = None,
        top_k: int = 6,
        min_score: float = 0.12,
        kinds: set[str] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[MemoryMatch]:
        if top_k < 1:
            return []
        self._refresh_if_needed()
        q = self.encoder.encode(text, concepts or [])
        exclude_ids = exclude_ids or set()
        matches: list[MemoryMatch] = []
        wanted = set(concepts or [])
        for row in self._entries.values():
            if row["item_id"] in exclude_ids:
                continue
            if kinds and row["kind"] not in kinds:
                continue
            score = self.encoder.cosine(q, row["vector"])
            if wanted:
                overlap = len(wanted.intersection(row["concepts"]))
                if overlap:
                    score = min(1.0, score + min(0.24, overlap * 0.08))
            if score < min_score:
                continue
            matches.append(MemoryMatch(
                item_id=row["item_id"], kind=row["kind"], text=row["text"],
                score=score, concepts=row["concepts"], metadata=row["metadata"],
            ))
        matches.sort(key=lambda m: (m.score, m.item_id), reverse=True)
        return matches[:top_k]

    def count(self) -> int:
        self._refresh_if_needed()
        return len(self._entries)
