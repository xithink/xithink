from pprint import pprint
from xithink.engine import XThinkEngine


e = XThinkEngine(":memory:")

for text in [
    "甲1导致乙1。乙1导致丙1。甲1间接导致丙1。甲1是条件甲。",
    "甲2导致乙2。乙2导致丙2。甲2间接导致丙2。甲2是条件甲。",
    "丁1导致戊1。戊1导致己1。丁1不间接导致己1。丁1是条件乙。",
    "丁2导致戊2。戊2导致己2。丁2不间接导致己2。丁2是条件乙。",
]:
    e.think(text, cycles=1, environment="测试场")

print("\n=== evolved rules ===")
pprint(e.rule_evolution.snapshot())

e.think(
    "新甲导致新乙。新乙导致新丙。新甲是条件甲。为什么新甲间接导致新丙？",
    cycles=1,
    environment="测试场",
)

print("\n=== symbolic answer ===")
pprint(e.last_trace["symbolic_answer"])
print("\n=== multi perspective ===")
pprint(e.last_trace["multi_perspective"])

e.close()
