# XThink 0.7 Verification

## Automated regression

```bash
python -m pytest -q
```

结果：

```text
62 passed
```

覆盖：

- 0.1–0.6 全回归。
- RuleGenealogy：base rule -> conditional child。
- RuleGenealogy：证据变化追加新版本而非覆盖。
- ActiveHypothesisTester：产生区分条件，但不增加 Symbolic Fact。
- Active test resolve 跨重启保持。
- CognitiveProfile 改变 learner/evolution/tester 参数。
- 0/10/20/30/40 岁经历检查点。
- Web 0.7 health/index/WebSocket。
- `/api/simulate` 参数真实传入模拟器。

## Compile verification

```bash
python -m compileall -q xithink tests examples
```

## 1000-experience simulation

使用固定 seed `20261001`，每 25 个 episode 执行一次批量规则学习与主动验证规划。

三套 profile 都跑了完整 1000 次经历，详见 `SIMULATION_1000.md` 和 `data/simulation_1000_*.json`。

## Important finding from first 1000-run

第一轮 1000 次模拟暴露出 0.6 的限制：

```text
verified conditional rule required zero counterexamples
```

在 6% experience noise 下，长期样本几乎必然出现反例，因此规则永远只能 provisional。

0.7 修复为显式 `counterexample_rate` 门槛。随后相同 seed 重跑，平衡型在 20 岁/500 次经历时已经能验证完整的 6 个隐藏条件分支，同时保持未知条件场景的 100% abstention。
