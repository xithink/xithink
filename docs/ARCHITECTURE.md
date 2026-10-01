# XThink 0.7 Architecture

## 1. Authority layers

```text
Long-Term Fact / verified Rule -> may prove
Experience Memory              -> case evidence only
Worldview                       -> environment-scoped evidence
Personal Symbol                 -> thought vocabulary only
Subconscious                    -> retrieval priority only
LLM KnowledgeClaim              -> quarantine until verified
Rule Genealogy                  -> audit/explanation only
Active Hypothesis Test          -> observation request only
```

事实性回答仍只能由 `SymbolicReasoner` 产生 `SymbolicProof`。

## 2. Rule genealogy

`RuleGenealogyTracker.sync()` 在每轮经验学习之后把当前候选规则和条件化规则同步成不可覆盖的谱系节点。

### Root

```text
LearnedRuleCandidate
  -> RuleGenealogyNode(generation=0, mutation=experience_rule_origin)
```

### Evidence revision

同一规则在支持数、反例数、置信度或状态变化时：

```text
node N
  ↓ parent
node N+1 mutation=evidence_revision
```

### Counterexample split

新条件化分支：

```text
base candidate
    ↓
conditional variant
mutation=counterexample_split
```

### Refinement

相同结论方向出现新的条件版本：

```text
old variant + base
       ↓
new variant
mutation=condition_refinement
```

### Supersede

条件后来失去区分力：

```text
active variant
    ↓
superseded_by_evidence
```

历史保留但推理权限撤销。

## 3. Active Hypothesis Testing

`ActiveHypothesisTester` 对同一 `base_signature` 下的正/反条件规则进行配对。

```text
positive variants × negative variants
        ↓
condition symmetric difference
        ↓
support balance / confidence balance
        ↓
expected_information_gain
        ↓
priority
```

它只输出应观察的信息：

```text
question
conditions
target
```

`resolve()` 只能由外部观察结果调用。规划器没有 API 可以直接创建 Fact。

## 4. Rule evolution under noisy experience

0.6 要求 `counterexamples == 0` 才能 verified；1000 次模拟证明这在含噪长期经验下过严。

0.7 改为：

```text
support >= min_support
AND confidence >= threshold
AND counterexample_rate <= max_counterexample_rate
```

反例仍保留在规则中并影响置信度。

## 5. Life experience simulator

### Separation

```text
Production:
ActiveHypothesisTester -> request observation -> external world/human/sensor

Simulation only:
ActiveHypothesisTester -> SyntheticLifeWorld Oracle -> structured Experience
```

模拟 Oracle 不可被生产 Thought Loop 调用。

### Age mapping

```text
experiences(age) = round(total_experiences * age / max_age)
```

所以默认 `total=1000, max_age=40` 时：

```text
0 / 250 / 500 / 750 / 1000
```

### Held-out cognitive tests

每个年龄 checkpoint 使用同一留出场景测：

- `symbolic_accuracy`
- `abstention_accuracy`
- `contradiction_rate`
- verified/evolved rule counts
- genealogy depth
- proposed/resolved active tests
- worldview/personal symbol maturity

## 6. Cognitive profiles

Profile 改的是“如何学习和验证”，不是偷偷修改事实：

```text
skepticism
exploration
min support
verification confidence
counterexample tolerance
active test aggressiveness
condition complexity
environment diversity
experience noise
```

相同经历流在不同 profile 下可以产生不同的理论成熟速度。
