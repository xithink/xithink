from xithink.models import Atom
from xithink.modules.concept_graph import ConceptGraph
from xithink.modules.memory import MemoryStore


def test_concept_graph_multihop_activation_and_path(tmp_path):
    store = MemoryStore(tmp_path / "graph.db")
    graph = ConceptGraph(store)
    graph.ingest_fact(Atom("自由", "意味着", "选择", confidence=0.9))
    graph.ingest_fact(Atom("选择", "导致", "风险", confidence=0.8))

    activation = graph.activate(["自由"], max_depth=3)
    assert activation["选择"] > activation["风险"] > 0

    path = graph.strongest_path("自由", "风险")
    assert [(x.source, x.relation, x.target) for x in path] == [
        ("自由", "意味着", "选择"),
        ("选择", "导致", "风险"),
    ]
    store.close()


def test_graph_persists(tmp_path):
    path = tmp_path / "graph.db"
    s1 = MemoryStore(path)
    g1 = ConceptGraph(s1)
    g1.ingest_fact(Atom("记忆", "影响", "判断", confidence=0.75))
    s1.close()

    s2 = MemoryStore(path)
    g2 = ConceptGraph(s2)
    assert g2.strongest_path("记忆", "判断")
    assert s2.graph_edge_count() == 1
    s2.close()
