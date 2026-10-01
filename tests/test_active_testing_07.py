from xithink.engine import XThinkEngine


def _teach_split(engine: XThinkEngine):
    for text in [
        "甲1导致乙1。乙1导致丙1。甲1间接导致丙1。甲1是条件甲。",
        "甲2导致乙2。乙2导致丙2。甲2间接导致丙2。甲2是条件甲。",
        "丁1导致戊1。戊1导致己1。丁1不间接导致己1。丁1是条件乙。",
        "丁2导致戊2。戊2导致己2。丁2不间接导致己2。丁2是条件乙。",
    ]:
        engine.think(text, cycles=1, environment="测试场")


def test_active_hypothesis_testing_proposes_discriminating_observation_without_fabricating_fact():
    e = XThinkEngine(":memory:")
    _teach_split(e)
    before = len(e.reasoner.facts)
    tests = e.active_testing.plan(environment="测试场", max_tests=10)
    assert tests
    assert any("条件甲" in " ".join(t.discriminating_conditions) or "条件乙" in " ".join(t.discriminating_conditions) for t in tests)
    assert all(t.status == "proposed" for t in tests if t.resolved_experience_id is None)
    assert len(e.reasoner.facts) == before
    e.close()


def test_active_hypothesis_test_resolution_persists(tmp_path):
    db = tmp_path / "active.db"
    e = XThinkEngine(db)
    _teach_split(e)
    test = e.active_testing.plan(environment="测试场", max_tests=1)[0]
    e.active_testing.resolve(test.id, outcome="positive-supported", experience_id="exp-1")
    e.close()
    e2 = XThinkEngine(db)
    restored = {t.id: t for t in e2.memory.load_active_hypothesis_tests()}[test.id]
    assert restored.status == "resolved"
    assert restored.resolved_experience_id == "exp-1"
    e2.close()
