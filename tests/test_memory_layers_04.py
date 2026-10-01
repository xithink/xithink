from xithink.engine import XThinkEngine
from xithink.models import Experience, KnowledgeClaim
from xithink.modules.knowledge import CallableKnowledgeProvider


def test_three_memory_layers_have_distinct_authority(tmp_path):
    e = XThinkEngine(tmp_path / "layers.db")
    e.think("自由意味着选择。选择导致风险。什么是自由？", cycles=1)
    snap = e.snapshot()["memory_layers"]
    assert snap["long_term"]["facts"] >= 3
    assert snap["long_term"]["authority"] == "symbolic-premise"
    assert snap["experience"]["count"] == 1
    assert snap["experience"]["authority"] == "case-context-only"
    assert snap["subconscious"]["traces"] > 0
    assert snap["subconscious"]["authority"] == "retrieval-priority-only"
    e.close()


def test_experience_memory_does_not_become_symbolic_fact(tmp_path):
    e = XThinkEngine(tmp_path / "experience.db")
    e.experience_memory.remember(Experience(
        content="月亮意味着奶酪",
        concepts=["月亮", "奶酪"],
        confidence=1.0,
    ))
    frame = e.semantic.parse("什么是月亮？")
    answer = e.answer_frame(frame)
    assert answer.answered is False
    assert not any(f.subject == "月亮" and f.object == "奶酪" for f in e.reasoner.facts)
    e.close()


def test_subconscious_activation_never_creates_fact(tmp_path):
    e = XThinkEngine(tmp_path / "sub.db")
    for _ in range(20):
        e.subconscious_memory.reinforce(["火", "危险"], reward=1.0)
    activation = e.subconscious_memory.activate(["火"])
    assert activation.get("危险", 0) > 0
    frame = e.semantic.parse("什么是火？")
    answer = e.answer_frame(frame)
    assert answer.answered is False
    assert not any(f.subject == "火" and f.object == "危险" for f in e.reasoner.facts)
    e.close()


def test_general_knowledge_is_quarantined_and_not_used_by_think(tmp_path):
    calls = []

    def provider_fn(query):
        calls.append(query)
        return [KnowledgeClaim("地球", "是", "行星", confidence=0.99)]

    provider = CallableKnowledgeProvider("fake-llm", provider_fn)
    e = XThinkEngine(tmp_path / "knowledge.db", knowledge_provider=provider)

    # Thinking itself never calls the model/general-knowledge provider.
    e.think("什么是地球？", cycles=1)
    assert calls == []
    assert e.answer_frame(e.semantic.parse("什么是地球？")).answered is False

    claims = e.consult_general_knowledge("什么是地球？")
    assert calls == ["什么是地球？"]
    assert len(claims) == 1
    assert claims[0].status == "quarantined"
    assert e.answer_frame(e.semantic.parse("什么是地球？")).answered is False
    assert not any(f.subject == "地球" for f in e.reasoner.facts)

    # Only explicit verification promotes it into long-term symbolic memory.
    assert e.verify_general_knowledge(claims[0].id)
    answer = e.answer_frame(e.semantic.parse("什么是地球？"))
    assert answer.answered is True
    assert "地球 是 行星" in answer.text
    e.close()
