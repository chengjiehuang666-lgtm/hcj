# 接口契约（Interface Contracts）

> 本目录是**契约先行**纪律的落地物：Schema 先冻结，再写实现。
> 定义与优先级见《系统架构设计》2.5 接口契约章节；与本节冲突的内容以架构文档为准。

## 文件清单

| 文件 | 契约 | 状态 |
| :--- | :--- | :--- |
| `task_event.schema.json` | TaskEvent — 外部渠道消息归一化后的标准任务事件 | v1.0.0 已冻结 |
| `subtask_dag.schema.json` | SubtaskDAG — Planner 输出的子任务图 | v1.0.0 已冻结 |
| `blackboard_entry.schema.json` | BlackboardEntry — 黑板条目（Agent 间唯一交换介质） | v1.0.0 已冻结（2026-10-17 枚举补 `planner`） |
| `agent_io.schema.json` | AgentIO — 统一 Agent 调用契约（`AgentInvocation` / `AgentResult`） | v1.0.0 已冻结（2026-10-17 枚举补 `planner`） |
| （待定） | AgentCard — A2A 风格能力声明 | 延后至阶段二/三 |

格式：JSON Schema **draft 2020-12**。

## 三条贯穿约定（与架构文档 2.5 一致）

1. **全局标识** — 必带 `task_id`（UUID）；子任务级消息另带 `node_id` + `attempt`。三者构成幂等键。
2. **溯源强制** — 产出结论的契约（`BlackboardEntry.sources`、`AgentResult.sources`）**不得为空**。这是「每个结论可回溯」要求的技术落点。
3. **预算单调收紧** — `TaskEvent.budget` → `SubtaskDAG` 节点 → `AgentInvocation.budget` 逐级传递，**只能收紧，不得放宽**。

## 引用方式

```
# 引用顶层结构
task_event.schema.json#
subtask_dag.schema.json#
blackboard_entry.schema.json#

# 引用 agent_io 内的子结构（该文件用 $defs 组织两个结构）
agent_io.schema.json#/$defs/AgentInvocation
agent_io.schema.json#/$defs/AgentResult
```

## 版本策略

`schema_version` 采用语义化版本（当前一律 `1.0.0`）：

- **小版本**（1.0.0 → 1.1.0）：仅新增可选字段，向后兼容，无需迁移。
- **大版本**（1.0.0 → 2.0.0）：字段语义变更或删除。**必须同步迁移既有评测集**，否则历史结果不可比。

## 修订记录

| 日期 | 契约 | 变更 | 性质 |
| :--- | :--- | :--- | :--- |
| 2026-10-17 | `agent_io`、`blackboard_entry` | `agent_role` 枚举补充 `planner` | **兼容性修正**：Planner 一直通过 `AgentInvocation` 调用（图外产出者），枚举从未包含它——W6 独立核验确认「每次规划都发出不合冻结 Schema 的消息」。补充枚举值不改变既有字段语义，不需要迁移评测集；`schema_version` 保持 `1.0.0` |

## 变更流程

1. 改 Schema 文件 → 2. 更新本表状态 → 3. 记录到对应文档的修订记录 → 4. Git 提交（契约变更必须单独提交，不与业务代码混在一起，便于回滚）。

## 待办

- [x] W1 前：把四份 Schema 与 Agent 基类实现对齐，确认可自动生成 Pydantic 模型（W1 完成）
- [x] W4 前：依据评测集实测结果校准 `TaskEvent.budget.max_tokens` 默认值（D2：W4 定 40,000、W6 修订为 60,000，见路线图 §3.6/§3.11）
- [ ] 阶段二：补 `agent_card.schema.json`
