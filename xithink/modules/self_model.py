from __future__ import annotations

from xithink.models import Decision, SelfState


class SelfModel:
    """Updates persistent cognitive preferences slowly and within bounds."""

    def update(self, state: SelfState, decision: Decision | None) -> None:
        state.thought_count += 1
        if decision is None:
            return

        tags = set(decision.hypothesis.tags)
        if "question" in tags or "association" in tags or "reflection" in tags:
            state.values["curiosity"] = state.values.get("curiosity", 0.5) + 0.001
        if "conflict" in tags:
            state.traits["skepticism"] = state.traits.get("skepticism", 0.5) + 0.001
        if decision.hypothesis.uncertainty > 0.65:
            q = decision.hypothesis.statement
            if q not in state.unresolved_questions:
                state.unresolved_questions.append(q)
                state.unresolved_questions = state.unresolved_questions[-50:]
        state.normalize()
