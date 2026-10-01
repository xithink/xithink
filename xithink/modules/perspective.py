from __future__ import annotations

from xithink.models import MemoryMatch, PerspectiveResonance


class PerspectiveResonanceEngine:
    """Use analogous personal experiences to shape attention ('感同身受').

    This is intentionally *not* emotion mind-reading. It says which of XThink's
    own past experiences resonate with the current concepts and lets that change
    search priority, while never asserting another person's private mental state.
    """

    def assess(
        self,
        concepts: list[str],
        recalled: list[MemoryMatch],
        *,
        empathy_weight: float = 0.75,
        limit: int = 5,
    ) -> PerspectiveResonance:
        wanted = set(concepts)
        matches = [m for m in recalled if m.kind == "experience"][:limit]
        if not matches:
            return PerspectiveResonance(
                0.0, note="没有足够相似的自身经历形成经验共鸣。"
            )
        shared: set[str] = set()
        activation: dict[str, float] = {}
        weighted_score = 0.0
        weighted_valence = 0.0
        total = 0.0
        ids: list[str] = []
        for match in matches:
            overlap = wanted.intersection(match.concepts)
            shared.update(overlap)
            weight = max(0.0, min(1.0, match.score))
            context = match.metadata.get("context") or {}
            affect_valence = float(context.get("affect_valence", 0.0))
            weighted_score += weight
            weighted_valence += affect_valence * weight
            total += weight
            ids.append(match.item_id)
            for concept in match.concepts:
                if concept in wanted:
                    continue
                activation[concept] = max(
                    activation.get(concept, 0.0),
                    min(0.50, weight * max(0.1, empathy_weight) * 0.45),
                )
        score = min(1.0, weighted_score / max(1, len(matches)))
        valence = weighted_valence / total if total else 0.0
        return PerspectiveResonance(
            score=score,
            shared_concepts=sorted(shared),
            experience_ids=ids,
            valence=max(-1.0, min(1.0, valence)),
            activation=dict(sorted(activation.items(), key=lambda kv: kv[1], reverse=True)[:10]),
            note="基于自身相似经历形成的经验共鸣；valence 只是经历文本的粗粒度显著性线索。它仅影响理解与搜索优先级，不代表对他人主观感受的事实判断。",
        )
