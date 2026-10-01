from xithink.engine import XThinkEngine


def test_end_to_end_thought_loop(tmp_path):
    e = XThinkEngine(tmp_path / "x.db")
    thoughts = e.think("自由意味着选择。选择导致风险。什么是自由？", cycles=3)
    assert 1 <= len(thoughts) <= 3
    assert e.state.thought_count == len(thoughts)
    assert len(e.memory.recent(10)) == len(thoughts)
    assert any("自由 间接导致 风险" == f.render() for f in e.reasoner.facts)
    e.close()


def test_state_remains_bounded(tmp_path):
    e = XThinkEngine(tmp_path / "x.db")
    for _ in range(30):
        e.think("什么是自由？", cycles=1)
    assert all(0.0 <= x <= 1.0 for x in e.state.values.values())
    assert all(0.0 <= x <= 1.0 for x in e.state.traits.values())
    e.close()


def test_internal_thought_does_not_pollute_fact_base(tmp_path):
    e = XThinkEngine(tmp_path / "pollution.db")
    e.think("自由意味着选择。选择导致风险。什么是自由？", cycles=5)
    assert not any(f.subject.startswith("由现有规则") for f in e.reasoner.facts)
    assert not any(f.subject.startswith("探索假设") for f in e.reasoner.facts)
    e.close()


def test_self_state_persists_across_restart(tmp_path):
    path = tmp_path / "persist.db"
    e1 = XThinkEngine(path)
    e1.think("什么是自由？", cycles=1)
    count = e1.state.thought_count
    e1.close()

    e2 = XThinkEngine(path)
    assert e2.state.thought_count == count
    assert e2.state.values["curiosity"] >= 0.9
    e2.close()


def test_knowledge_persists_across_restart(tmp_path):
    path = tmp_path / "knowledge.db"
    e1 = XThinkEngine(path)
    e1.think("自由意味着选择。选择导致风险。", cycles=1)
    e1.close()

    e2 = XThinkEngine(path)
    rendered = {f.render() for f in e2.reasoner.facts}
    assert "自由 意味着 选择" in rendered
    assert "选择 导致 风险" in rendered
    assert "自由 间接导致 风险" in rendered
    e2.close()
