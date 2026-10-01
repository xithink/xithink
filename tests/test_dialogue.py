from xithink.dialogue import DialogueService
from xithink.engine import XThinkEngine


def test_dialogue_service_returns_semantic_and_reply(tmp_path):
    engine = XThinkEngine(tmp_path / "dialogue.db")
    service = DialogueService(engine)
    result = service.respond("自由意味着选择。选择导致风险。什么是自由？", cycles=2)
    assert result["reply"]
    assert "semantic" in result
    assert "自由" in result["semantic"]["concepts"]
    assert result["thoughts"]
    assert "自由 间接导致 风险" in result["reply"]
    engine.close()
