from xithink.modules.embedding_memory import EmbeddingMemory
from xithink.modules.memory import MemoryStore


def test_embedding_memory_recalls_related_wording(tmp_path):
    store = MemoryStore(tmp_path / "emb.db")
    emb = EmbeddingMemory(store)
    emb.index("m1", "thought", "自由与选择之间存在紧密关系", concepts=["自由", "选择"])
    emb.index("m2", "thought", "天气温度影响降雨", concepts=["天气", "降雨"])

    matches = emb.query("选择和自由是什么关系", concepts=["自由", "选择"], top_k=2, min_score=0.0)
    assert matches[0].item_id == "m1"
    assert matches[0].score > matches[1].score
    store.close()


def test_embedding_memory_persists(tmp_path):
    path = tmp_path / "emb.db"
    s1 = MemoryStore(path)
    e1 = EmbeddingMemory(s1)
    e1.index("x", "fact", "机器需要能源", concepts=["机器", "能源"])
    s1.close()

    s2 = MemoryStore(path)
    e2 = EmbeddingMemory(s2)
    result = e2.query("机器和能源", concepts=["机器", "能源"], min_score=0.0)
    assert result and result[0].item_id == "x"
    s2.close()
