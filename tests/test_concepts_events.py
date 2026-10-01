from xithink.engine import XThinkEngine


def test_concepts_have_stable_ids_and_persist(tmp_path):
    path = tmp_path / "concepts.db"
    e1 = XThinkEngine(path)
    e1.think("自由意味着选择。", cycles=1)
    c1 = e1.concepts.resolve("自由")
    assert c1 is not None
    cid = c1.id
    e1.close()

    e2 = XThinkEngine(path)
    c2 = e2.concepts.resolve("自由")
    assert c2 is not None
    assert c2.id == cid
    e2.close()


def test_events_persist(tmp_path):
    path = tmp_path / "events.db"
    e = XThinkEngine(path)
    e.think("今天机器人学习逻辑。", cycles=1)
    events = e.memory.recent_events()
    assert events
    assert events[0]["actor"] == "机器人"
    assert events[0]["action"] == "学习"
    assert events[0]["time_ref"] == "今天"
    e.close()
