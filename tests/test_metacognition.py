from xithink.models import Atom, Hypothesis
from xithink.modules.concept_graph import ConceptGraph
from xithink.modules.memory import MemoryStore
from xithink.modules.metacognition import MetacognitionEngine
from xithink.modules.symbolic import SymbolicReasoner


def test_counterexample_reduces_candidate_confidence(tmp_path):
    store = MemoryStore(tmp_path / "meta.db")
    graph = ConceptGraph(store)
    reasoner = SymbolicReasoner()
    positive = Atom("自由", "导致", "风险", confidence=0.85)
    negative = Atom("自由", "导致", "风险", negated=True, confidence=0.92)
    reasoner.add_fact(positive)
    reasoner.add_fact(negative)
    graph.ingest_fact(positive)
    graph.ingest_fact(negative)

    h = Hypothesis(
        "由现有规则可推出：自由 导致 风险",
        supporting_atoms=[positive], evidence=0.85, consistency=0.95,
        uncertainty=0.15, tags=["inference"],
    )
    reviewed, assessment = MetacognitionEngine().assess(h, reasoner, graph, [])
    assert assessment.counterevidence >= 0.8
    assert reviewed.consistency < h.consistency * 0.5
    assert reviewed.uncertainty > h.uncertainty
    assert assessment.needs_verification is True
    assert assessment.counterexamples
    store.close()


def test_counterexample_can_become_explicit_reflection(tmp_path):
    store = MemoryStore(tmp_path / "meta-reflect.db")
    graph = ConceptGraph(store)
    reasoner = SymbolicReasoner()
    positive = Atom("机器", "需要", "联网", confidence=0.72)
    negative = Atom("机器", "需要", "联网", negated=True, confidence=0.95)
    reasoner.add_fact(positive)
    reasoner.add_fact(negative)
    graph.ingest_fact(positive)
    graph.ingest_fact(negative)
    h = Hypothesis("机器需要联网", [positive], evidence=0.72, consistency=0.9, uncertainty=0.2)
    meta = MetacognitionEngine()
    reviewed, assessments = meta.review([h], reasoner, graph, [])
    reflections = meta.counterexample_reflections(reviewed, assessments)
    assert reflections
    assert reflections[0].statement.startswith("反例审查")
    assert "机器 不需要 联网" in reflections[0].statement
    store.close()
