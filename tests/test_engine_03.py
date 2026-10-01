from xithink.engine import XThinkEngine


def test_engine_trace_contains_graph_recall_and_metacognition(tmp_path):
    path = tmp_path / "x.db"
    e = XThinkEngine(path)
    e.think("自由意味着选择。选择导致风险。", cycles=2)
    thoughts = e.think("自由不间接导致风险。什么是自由？", cycles=2)

    trace = e.last_trace
    assert trace["recall"]
    assert trace["cycles"]
    candidates = [c for cycle in trace["cycles"] for c in cycle["candidates"]]
    assert any("metacognition" in c for c in candidates)
    assert thoughts[0].content.startswith("存在冲突")
    assert trace["cycles"][0]["selected"]["metacognition"]["needs_verification"] is False

    snap = e.snapshot()
    assert snap["concept_graph"]["edges"] >= 3
    assert snap["embedding_memory"]["items"] >= 3
    e.close()
