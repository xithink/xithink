from __future__ import annotations

from typing import Callable, Protocol

from xithink.models import KnowledgeClaim
from xithink.modules.memory import MemoryStore


class GeneralKnowledgeProvider(Protocol):
    """Foundation-model boundary: provide common-knowledge candidates only."""

    name: str

    def lookup(self, query: str) -> list[KnowledgeClaim]: ...


class NullKnowledgeProvider:
    name = "none"

    def lookup(self, query: str) -> list[KnowledgeClaim]:
        return []


class CallableKnowledgeProvider:
    """Adapter for a local Qwen/RWKV/Gemma wrapper without hard dependency."""

    def __init__(self, name: str, fn: Callable[[str], list[KnowledgeClaim]]) -> None:
        self.name = name
        self.fn = fn

    def lookup(self, query: str) -> list[KnowledgeClaim]:
        claims = self.fn(query)
        out: list[KnowledgeClaim] = []
        for claim in claims:
            claim.provider = self.name
            claim.query = query
            claim.status = "quarantined"
            out.append(claim)
        return out


class KnowledgeGateway:
    """Quarantine for foundation-model/common-knowledge output.

    `consult()` never mutates the symbolic fact/rule base. Promotion is an
    explicit second action performed by the engine after verification.
    """

    def __init__(self, store: MemoryStore, provider: GeneralKnowledgeProvider | None = None) -> None:
        self.store = store
        self.provider = provider or NullKnowledgeProvider()

    def consult(self, query: str) -> list[KnowledgeClaim]:
        claims = self.provider.lookup(query)
        for claim in claims:
            claim.provider = getattr(self.provider, "name", claim.provider)
            claim.query = query
            claim.status = "quarantined"
            self.store.save_knowledge_claim(claim)
        return claims

    def pending(self, limit: int = 50) -> list[KnowledgeClaim]:
        return self.store.list_knowledge_claims("quarantined", limit)

    def get(self, claim_id: str) -> KnowledgeClaim | None:
        return self.store.load_knowledge_claim(claim_id)

    def mark_verified(self, claim_id: str) -> KnowledgeClaim | None:
        claim = self.get(claim_id)
        if claim is None:
            return None
        self.store.update_knowledge_claim_status(claim_id, "verified")
        claim.status = "verified"
        return claim

    def reject(self, claim_id: str) -> bool:
        return self.store.update_knowledge_claim_status(claim_id, "rejected")
