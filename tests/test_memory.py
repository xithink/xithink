from xithink.models import Thought
from xithink.modules.memory import MemoryStore


def test_memory_persistence(tmp_path):
    path = tmp_path / "memory.db"
    m = MemoryStore(path)
    t = Thought(0, "decision", "探索自由", 0.7, ["自由"])
    m.add_thought(t)
    m.close()

    m2 = MemoryStore(path)
    rows = m2.search_by_concepts(["自由"])
    assert rows[0]["content"] == "探索自由"
    m2.close()
