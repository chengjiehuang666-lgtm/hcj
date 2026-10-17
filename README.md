# 多 Agent 接入协作项目

企业级多 Agent 智能协作平台。当前处于**阶段二（动态编排 + 多 Agent 协同 + 异常处理/HITL，W7 完成）**（单人全职开发，计划与范围见《实施路线图》）。

## 唯一事实源

需求、架构、计划三份文档是项目的唯一事实源；代码实现必须与之一致。

| 文档 | 路径 |
| :--- | :--- |
| 产品需求 | `产品需求/产品需求文档.md` |
| 系统架构设计 | `架构设想/系统架构设计.md` |
| 实施路线图与风险 | `当前进度/实施路线图.md` |
| 接口契约（JSON Schema） | `架构设想/接口契约/` |

**代码中所有跨模块的数据结构都必须能追溯到 `架构设想/接口契约/` 下的某份契约。** 契约变更须单独提交，不与业务代码混在一起。

## 目录结构

```
├── 产品需求/                  # 文档（唯一事实源）
├── 架构设想/
│   ├── 系统架构设计.md
│   └── 接口契约/              # JSON Schema，draft 2020-12
├── 当前进度/
│   └── 实施路线图.md
├── src/multiagent/            # 应用代码
│   ├── config.py              # 配置（pydantic-settings，读 .env）
│   ├── contracts/             # 契约的 Pydantic 实现，与接口契约一一对应
│   ├── gateway/               # 大模型网关（供应商无关 + 降级 + 分级路由）
│   ├── agents/                # 专家 Agent：planner / retriever / analyst / reviewer / writer
│   ├── graph/                 # 编排层：由 Planner 的 DAG 在运行时驱动构图与执行
│   ├── tools/                 # SQL 沙箱（只读角色）与 Python 沙箱（容器隔离）
│   ├── rag/                   # 知识库：本地 embedding + pgvector + pg_trgm 混合检索
│   ├── memory/                # 长期记忆：评审门槛后沉淀、跨会话检索（与知识库分表）
│   ├── eval/                  # 评测集与执行器（27 条用例 + 打分器）
│   ├── api/                   # REST + SSE 入口
│   └── cli.py                 # 命令行入口
├── tests/                     # 契约一致性、引擎、Agent、记忆、网关等单测
├── docker/initdb/             # PG 初始化 SQL（扩展 / 演示数据 / 知识库 / 记忆）
├── docker-compose.yml         # PostgreSQL(pgvector) + Redis
├── .env.example               # 环境变量模板（真实密钥放 .env，不入库）
└── pyproject.toml
```

## 环境前置条件

| 依赖 | 当前状态 | 说明 |
| :--- | :--- | :--- |
| Python | ✅ 本机 3.13.7 | 已确认使用 3.13；`requires-python = ">=3.11"` |
| Docker Desktop | ✅ 运行中 | PostgreSQL 16 + pgvector、Redis 7 两个容器 healthy |
| 模型端点 | ✅ 本地端点 127.0.0.1:7863 | reasoning 档主力与 fast 档都走本地；见「已知限制」 |

## 已知限制

1. **模型端点拓扑与质量边界**（2026-10-16 W6 收尾更新）：
   - reasoning 档：主 = 本地 `cn:deepseek-v4.1-flash`（推理型），备 = 本地 `glm-5.3`
   - fast 档：主 = 本地 `glm-5.3`，备 = 云端 `deepseek-flash`

   原先的云端 deepseek 主模型**余额耗尽**（HTTP 402），已整体切到本地端点；
   **切换只改了 `.env`，没有改代码**。备选端点 glm-5.3 的真实质量边界已被观测：
   它在**规划类**任务上不可靠（会产出 node_id 与 role 不一致的畸形 DAG，被规划门禁
   拦下），适合作为快速档/兜底，不适合承担规划与归因。两个档位的主备都指向同一个
   本地端点，因此**本地服务是可用性单点**。
2. **无远端仓库**：回档能力目前仅限本机（见《实施路线图》风险 R12）。
3. **时延与成功率相互牵制，W6/W7 明码选择「保质量」**：
   - **成功率 26/27 = 96.3%**（24 条分析 + 2 条文档 + 1 条跨会话记忆）。W7 唯一失败
     `doc_return_policy_notice` 的原因是 Planner 把「内部通知」误标为需审批 → 任务暂停
     （评测侧自动批准 + 提示词收敛后单跑通过），不是能力问题
   - P50 **53.0 秒**，P95 126.5 秒仍未达标；reviewer 占模型调用耗时约 50%——
     这是「强制必审」的明码代价，如实记录、不追时延
4. **W7 能力的边界（如实记录）**：熔断与 HITL 均为**后端**能力——熔断后任务
   终止于稳定错误码（`BUDGET_EXCEEDED`/`STEP_LIMIT`/`DEADLINE_EXCEEDED`），
   单租户没有审批队列；HITL 只覆盖 `requires_human_approval` 节点的**审批门**，
   审批 UI、条件分支/竞速、多 reviewer 仲裁、`needs_human` 的完整交互均延后。
   动态重规划上限 1 次（`MAX_REPLANS`），复用的是「重新规划」而非旧图状态。
   任务注册表仍是进程内：HITL 恢复可从 checkpoint 重建，但任务列表/历史事件不跨重启。
5. **推理型模型的 `max_tokens` 是「共享额度」**：推理模型会先产生思考 token 再产生
   正文，`max_tokens` 由两者共享。额度偏小时思考会吃满全部额度，导致正文为空且
   `finish_reason=length`（W6 实测：本地 `cn:deepseek-v4.1-flash` 某次调用思考消耗
   满 8192 仍无正文，任务失败；上限已扩至 16384）。

   网关对此**显式报错**（`EmptyCompletionError`），不会静默返回空字符串——
   静默失败会让上游误以为模型「没话说」。`LLMResponse.reasoning_tokens` 单独记录了
   思考 token，供成本计量使用。

## 快速开始

```bash
# 1. 环境变量
cp .env.example .env          # Windows: copy .env.example .env
                              # 默认 LLM_MODE=stub，不需要 API key

# 2. 安装（建议虚拟环境）
python -m venv .venv
# Windows: .venv\Scripts\activate      Git Bash: source .venv/Scripts/activate
pip install -e ".[dev]"

# 3. 跑测试（契约一致性 + 网关降级 + 引擎/Agent/记忆）
#    数据库不可达时相关用例自动跳过；记忆相关用例只操作 w6test- 命名空间，
#    不会动开发库里真实沉淀的数据
pytest -q

# 4. 启动依赖服务（需先有 Docker）
docker compose up -d

# 5. 灌知识库（首次或数据卷重建后）
python -m multiagent.rag.ingest --stats

# 6. 跑一条端到端任务 / 评测回归
python -m multiagent.cli "分析上月华东区销售额下滑的原因"
python -m multiagent.eval.runner --limit 3        # 全量 27 条约 45 分钟
python -m multiagent.memory --stats               # 长期记忆维护（--search / --clear）
```

### HITL 审批（W7）

高风险请求（对外发送 / 签发 / 资金或数据变更）会被 Planner 标上
`requires_human_approval`，任务执行到该节点后**暂停**（状态 `awaiting_approval`），
审批后从同一 checkpoint 恢复：

```bash
# 提交（命中高风险关键词会暂停）
curl -s -X POST localhost:8000/tasks -H "Content-Type: application/json"      -d '{"question":"写一份对外公告并发送给客户"}'
# → {"task_id":"...","status":"pending",...}

# 查询：status 变为 awaiting_approval 时带了 approval_requests（节点、产物、验收标准）
curl -s localhost:8000/tasks/<task_id>

# 批准 / 驳回 / 改写（改写需提供替换后的节点产物）
curl -s -X POST localhost:8000/tasks/<task_id>/approval      -H "Content-Type: application/json"      -d '{"decision":"approved","comment":"同意"}'
# decision: rejected → 任务终态 rejected；rewritten → 需带 outputs
```

> 已有数据卷的环境**不会**自动执行新增的初始化 SQL（initdb 只在空卷跑一次）。
> 新增脚本需手工执行，例如长期记忆表：
> `docker exec -i ma-postgres psql -U multiagent -d multiagent < docker/initdb/04-memory-schema.sql`

## 开发约定

1. **契约先行**：改动数据结构前先改 `架构设想/接口契约/` 的 Schema，再改 `src/multiagent/contracts/`。
2. **空值不输出**：可选字段为空时序列化会省略（`exclude_none=True`），以匹配契约中非 nullable 字段的定义。
3. **降级优先**：任何外部依赖调用都必须有降级路径；模型调用统一走 `gateway`，不要在业务代码里直接发 HTTP。
4. **每个稳定功能立即提交**，并用 `git tag` 标记里程碑，保证可回档。
