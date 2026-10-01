from xithink.engine import XThinkEngine


def _teach_split(engine: XThinkEngine, env: str = "测试场") -> None:
    for text in [
        "甲1导致乙1。乙1导致丙1。甲1间接导致丙1。甲1是条件甲。",
        "甲2导致乙2。乙2导致丙2。甲2间接导致丙2。甲2是条件甲。",
        "丁1导致戊1。戊1导致己1。丁1不间接导致己1。丁1是条件乙。",
        "丁2导致戊2。戊2导致己2。丁2不间接导致己2。丁2是条件乙。",
    ]:
        engine.think(text, cycles=1, environment=env)


def test_rule_genealogy_links_base_rule_to_conditional_children():
    e = XThinkEngine(":memory:")
    _teach_split(e)
    snap = e.rule_genealogy.snapshot()
    roots = [n for n in snap["nodes"] if n["rule_kind"] == "learned_candidate"]
    children = [n for n in snap["nodes"] if n["rule_kind"] == "evolved_variant"]
    assert roots and children
    root_ids = {n["id"] for n in roots}
    assert any(root_ids.intersection(c["parent_ids"]) for c in children)
    assert snap["max_generation"] >= 1
    e.close()


def test_rule_genealogy_appends_revision_instead_of_overwriting():
    e = XThinkEngine(":memory:")
    for text in [
        "甲导致乙。乙导致丙。甲间接导致丙。",
        "丁导致戊。戊导致己。丁间接导致己。",
        "庚导致辛。辛导致壬。庚间接导致壬。",
    ]:
        e.think(text, cycles=1, environment="A")
    before = e.rule_genealogy.snapshot(limit=200)
    for text in [
        "子导致丑。丑导致寅。子不间接导致寅。",
        "卯导致辰。辰导致巳。卯不间接导致巳。",
        "午导致未。未导致申。午不间接导致申。",
    ]:
        e.think(text, cycles=1, environment="A")
    after = e.rule_genealogy.snapshot(limit=300)
    assert len(after["nodes"]) > len(before["nodes"])
    assert any(n["mutation"] in {"evidence_revision", "superseded_by_evidence"} for n in after["nodes"])
    e.close()
