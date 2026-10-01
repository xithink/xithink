# XThink · X思维决策引擎 0.7.0

XThink 是一个开源神经符号认知与决策引擎。0.7.0 在 0.6 的经验规则、正反例、条件化规则和多视角逻辑链基础上，加入三块面向“认知成长”的能力：

1. **Rule Genealogy / 规则家谱**：规则从形成、验证、分裂、修正到失效都保留不可覆盖的谱系节点。
2. **Active Hypothesis Testing / 主动验证理论**：面对相反理论时，不只报告冲突，而是选择最有信息量的未知条件并提出下一步观察需求。
3. **Life Experience Simulation / 年龄-经历模拟**：可用 0/10/20/30/40 岁等检查点模拟经验增长，并调节怀疑度、探索度、规则验证阈值、主动验证强度等“思维模型”参数。

> 核心原则仍不变：大模型只提供隔离的通识知识候选；XThink 的事实性结论必须来自 `Fact + Rule + SymbolicProof`。经验、Embedding、潜意识、个人符号和 LLM 都不能绕过证明边界。

## 0.7 Thought Loop

```text
现实经历 / 语音 / 文字
        ↓
Experience + Worldview
        ↓
Experience Rule Learner
        ↓
Positive / Counterexample
        ↓
Rule Evolution
        ↓
Conditional Rules
        ↓
Rule Genealogy
  origin → revision → split → refinement → superseded
        ↓
Competing Theories
        ↓
Active Hypothesis Testing
        ↓
“观察什么最能区分理论 A / B？”
        ↓
真实外部观察 / 人工输入 / 传感器
        ↓
新 Experience
        ↓
重新验证规则
        ↓
Symbolic Reasoner + Proof Tree
```

## 规则家谱

0.6 只保存当前 `EvolvedRuleVariant` 状态；0.7 额外保存不可覆盖的历史快照：

```text
Rule v0: X→Z
   │  新反例
   ↓
Rule v1: X + 条件A → Z
   │  新正/反证据
   ↓
Rule v2: X + 条件A + 条件B → Z
   │  条件失效
   ↓
Rule v3: superseded
```

每个 `RuleGenealogyNode` 包含：

```text
rule_ref
base_signature
parent_ids
generation
mutation
status
premises / conditions / conclusion
support / counterexamples / confidence
environments
rationale
```

家谱本身只用于解释和审计，**不直接具有推理权限**。

## 主动验证理论

当系统同时有：

```text
条件甲 → C
条件乙 → not-C
```

`ActiveHypothesisTester` 会比较两套理论的条件、支持数量和置信度，输出：

```text
ActiveHypothesisTest
├─ hypothesis_a
├─ hypothesis_b
├─ discriminating_conditions
├─ expected_information_gain
├─ priority
├─ question
└─ status
```

例如：

```text
为了区分两套相反理论，优先观察：
?x 是 条件甲；?x 是 条件乙。
随后检验目标结论是否成立。
```

生产路径中，这只是**观察请求**。XThink 不会自己生成验证结果。

1000 次人生模拟使用独立的 `SyntheticLifeWorld` 作为测试 Oracle；它与生产推理路径隔离，只用于自动验证学习闭环。

## 噪声与经验规律

0.7 修正了早期“规则必须零反例才能验证”的限制。真实长期经验中即使规律稳定，也可能存在测量误差、偶发例外或环境噪声。

因此条件规则现在同时检查：

```text
support
counterexamples
confidence
counterexample_rate
```

平衡型默认允许最大 12% 的明确例外；例外仍完整保留并降低 Proof confidence，不会被删除。

## 可调思维模型

`CognitiveProfile` 目前可调：

```python
CognitiveProfile(
    skepticism=0.65,
    exploration=0.75,
    rule_min_support=3,
    rule_verification_confidence=0.74,
    reject_counterexamples=2,
    condition_min_support=2,
    condition_verification_confidence=0.72,
    condition_max_counterexample_rate=0.12,
    active_test_aggressiveness=0.70,
    active_test_budget=3,
    max_condition_terms=2,
    environment_diversity=0.70,
    experience_noise=0.06,
)
```

内置：

- `balanced`：平衡型。
- `conservative`：更高验证门槛、更低例外容忍、更慢接受规则。
- `exploratory`：更积极验证假设、更快形成候选规则、允许更多例外后再修正。

## 1000 次经历模拟

默认：

```text
0岁  = 0 次经历
10岁 = 250 次经历
20岁 = 500 次经历
30岁 = 750 次经历
40岁 = 1000 次经历
```

映射不是生物学年龄断言，而是用于比较“经验累积阶段”的实验刻度；`total_experiences`、`max_age` 和 checkpoint 都可修改。

Python：

```python
from xithink.engine import XThinkEngine
from xithink.modules.life_simulation import profile_from_name

engine = XThinkEngine(":memory:")
report = engine.simulate_life(
    total_experiences=1000,
    max_age=40,
    age_checkpoints=[0, 10, 20, 30, 40],
    seed=20261001,
    profile=profile_from_name("balanced"),
)
print(report.to_dict())
```

示例：

```bash
python examples/demo_07_life_simulation.py --experiences 1000 --profile balanced
```

## Web / 实时语音

```bash
python -m pip install -e ".[dev]"
xithink-web
```

打开：

```text
http://127.0.0.1:8000
```

0.7 页面新增：

- 思维模型选择：平衡 / 谨慎 / 探索。
- 1000 次经历模拟参数。
- 怀疑度、探索度、主动验证强度。
- **规则家谱/成长** 面板。
- 年龄检查点结果。
- Active Hypothesis Test 列表。

模拟使用独立内存 Engine，不污染正常对话数据库。

## 测试

```bash
python -m pytest -q
python -m compileall -q xithink tests examples
```

0.7 覆盖包括：

- 0.1–0.6 全部回归。
- 规则家谱父子关系与版本追加。
- 主动验证不制造 Fact。
- 主动验证结果跨重启保持。
- 思维模型参数实际改变学习阈值。
- 年龄检查点模拟。
- Web 0.7 与模拟 API。

完整 1000 次实测见 `docs/SIMULATION_1000.md`。
