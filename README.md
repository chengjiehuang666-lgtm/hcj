# 多 Agent 接入协作项目

企业级多 Agent 智能协作平台。**W1–W12 已完成、M1/M2/M3 达成；W13 控制台 + W13.1 知识库扩容；W14 安全/运维加固（HTTPS/限流/审计）**。

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
│   ├── observability.py       # W10：结构化日志、task_id 贯穿、Token/成本计量
│   ├── contracts/             # 契约的 Pydantic 实现，与接口契约一一对应
│   ├── gateway/               # 大模型网关（供应商无关 + 降级 + 分级路由）
│   ├── agents/                # 专家 Agent：planner / retriever / analyst / reviewer / writer
│   ├── graph/                 # 编排层：由 Planner 的 DAG 在运行时驱动构图与执行
│   ├── tools/                 # SQL 沙箱（只读角色）与 Python 沙箱（容器隔离）
│   ├── rag/                   # 知识库：本地 embedding + pgvector + pg_trgm 混合检索
│   ├── memory/                # 长期记忆：评审门槛后沉淀、跨会话检索（与知识库分表）
│   ├── eval/                  # 评测集与执行器（50 条用例）+ 三层压测工具
│   ├── security/              # W9：输入过滤、输出 DLP、API Key 角色权限
│   ├── api/                   # REST + SSE 入口
│   │   └── console/           # W13：轻量控制台（无构建链单页）
│   └── cli.py                 # 命令行入口
├── tests/                     # 契约一致性、引擎、Agent、记忆、网关等单测
├── docker/initdb/             # PG 初始化 SQL（扩展 / 演示数据 / 知识库 / 记忆）
├── docker/app/Dockerfile      # W12 应用镜像
├── docker-compose.yml         # PostgreSQL(pgvector) + Redis
├── docker-compose.app.yml     # W12 应用服务 overlay
├── scripts/                   # deploy.sh / rollback.sh / smoke.py
├── 运维说明.md                # W12 部署与回滚手册
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
2. **远端仓库**：`https://github.com/chengjiehuang666-lgtm/hcj`——2026-10-19 已推送 `main` 与全部 11 个标签；
   本机仍是唯一开发环境（R12 的「唯一开发者中断」风险仍在，但代码不再是单副本）。
3. **时延与成功率相互牵制，W6–W8 明码选择「保质量」**：
   - **W11 全量复跑 50/50 = 100%**（分析 29/29、文档 20/20、记忆 1/1）；P50 **57.5 秒**首次在 50 条进入 60 秒线，
     P95 133.1 秒仍未达标；reviewer 占模型调用耗时约 41%——这是「强制必审」的明码代价，如实记录、不追时延
   - **D1 已定**：单机推荐 **12 并发任务**，持续吞吐 **0.18 任务/s（≈648/小时）**；端点容量 20，留 8 个调用余量
   - **D2 已定**：任务 Token 熔断 **80,000**（W11 两轮最大 48,051 × 1.5，向上取整到万）
   - **瓶颈已在 W8 定位并扩容**：应用编排层可撑 32 并发（P50 5.6 秒，65 条 DB 连接）；
     本地模型端点原来每账号在途 3（2 个 cn 账号 → 硬上限 6），2026-10-19 将
     `pool.max_in_flight` 调至 10 后实测容量 **20**；全链路 12 并发任务吞吐
     **0.12 任务/s**（扩容前 0.06），再往上只增时延不增吞吐
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
6. **W8 的边界（如实记录）**：评测到 50 条，但**只有一轮全量**（单条失败重跑即过，
   波动方差未估计）；评审 verdict 中文档体裁 8/20 fail（只影响记忆沉淀，不影响交付），
   **按体裁校准仍然未做**；第 3 场景按 D6 决议（2026-10-19）延后到扩员后；压测只覆盖
   单进程，任务注册表与 SSE 不跨进程；端点备选接管演练（O12）留待 W9。

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
# 示例语料（5 篇）
python -m multiagent.rag.ingest
# Obsidian 仓库批量灌库（Markdown 标题感知分片 + docx；默认排除项目记忆/Agent 指令）
python -m multiagent.rag.vault_ingest --vault "G:/第二大脑obsidian仓库/obsidian仓库" --dry-run
python -m multiagent.rag.vault_ingest --vault "G:/第二大脑obsidian仓库/obsidian仓库"

# 6. 跑一条端到端任务 / 评测回归
python -m multiagent.cli "分析上月华东区销售额下滑的原因"
python -m multiagent.eval.runner --limit 3        # 全量 50 条约 55 分钟
python -m multiagent.eval.runner --scenario document   # 只跑文档子集（20 条）
python -m multiagent.eval.loadtest --mode pipeline --levels 1,4,8,16,32 --tasks-per-level 32
                                                  # W8 压测：不烧 token 的编排层瓶颈定位
python -m multiagent.eval.loadtest --mode model --levels 12,16,20 --tasks-per-level 20
                                                  # 模型端点并发膝点（tasks 必须 ≥ 档位）
python -m multiagent.memory --stats               # 长期记忆维护（--search / --clear）
```

## 单机部署（W12）

```bash
cp .env.example .env           # 填 API_KEYS 与模型 Key
mkdir -p .model_cache          # embedding 缓存（容器只读挂载）
bash scripts/deploy.sh         # 构建 app + PG/Redis，健康检查 + 冒烟（HTTP，仅本机）
bash scripts/deploy.sh --tls   # 同上 + HTTPS 8443（自签证书，浏览器会告警）
bash scripts/rollback.sh v0.14-w11-calibration   # 按 Git 标签回滚并冒烟
```

- 应用容器通过 `host.docker.internal:7863` 访问宿主机模型端点；Linux 需改地址。
- 容器内无 Docker socket，Python 沙箱降级为 subprocess（结果里标注 isolation）。
- 完整步骤、日志、指标、备份与常见问题见 [运维说明.md](运维说明.md)。
- 回滚演练记录：部署 → 回滚到 `v0.14-w11-calibration` → 恢复，三次冒烟均通过。

### HITL 审批（W7）

高风险请求（对外发送 / 签发 / 资金或数据变更）会被 Planner 标上
`requires_human_approval`，任务执行到该节点后**暂停**（状态 `awaiting_approval`），
审批后从同一 checkpoint 恢复：

```bash
# 以下示例假设已在 .env 配置 API_KEYS，并导出操作员/管理员的 Key：
#   export API_KEY=<operator 或 admin 的 key>
# 提交（命中高风险关键词会暂停）
curl -s -X POST localhost:8000/tasks \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"question":"写一份对外公告并发送给客户"}'
# → {"task_id":"...","status":"pending",...}

# 查询：status 变为 awaiting_approval 时带了 approval_requests（节点、产物、验收标准）
curl -s localhost:8000/tasks/<task_id> -H "Authorization: Bearer $API_KEY"

# 批准 / 驳回 / 改写（改写需提供替换后的节点产物）
# 审批只有 admin 能做；operator/viewer 会收到 403
curl -s -X POST localhost:8000/tasks/<task_id>/approval \
  -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"decision":"approved","comment":"同意"}'
# decision: rejected → 任务终态 rejected；rewritten → 需带 outputs
```

## 安全基线（W9）

1. **API Key + 三角色**（`src/multiagent/security/auth.py`）：`API_KEYS=key:role,...`，
   `admin` = 读取/提交/审批，`operator` = 读取/提交，`viewer` = 只读。
   `API_AUTH_ENABLED=true` 但未配置任何 Key 时**服务拒绝启动**（fail closed）。
   `/health` 是公开探活端点。
2. **输入过滤**：6 类 Prompt Injection 规则 + 敏感词；NFKC、零宽字符、汉字间空格归一化。
   命中返回 `400 {"detail":{"code":"INPUT_BLOCKED",...}}`，不建任务、不调模型；
   输入中的身份证/手机号/银行卡/密钥会先脱敏再送模型。
3. **输出 DLP**：身份证 / 手机号 / 银行卡（Luhn 校验）/ 密钥与令牌正则脱敏，
   覆盖 API 结果、SSE、CLI 与长期记忆；422 校验错误不回显原始输入。
4. **SSE 的例外**：浏览器 `EventSource` 不能自定义 Header，因此只有
   `/tasks/{id}/events` 允许 `?api_key=`；Key 可能进入访问日志，生产应置于 HTTPS 反代后。
5. **未做**：CORS、Vault、OIDC/RBAC+ABAC、外部检索内容隔离标注、财务数字与模型层 DLP、
   Key 轮换——均在《实施路线图》§3.14 记录。

## 运维加固（W14）

- **HTTPS**：`bash scripts/deploy.sh --tls` 生成自签证书并以 HTTPS（8443）启动，
  HTTP 仅绑定 `127.0.0.1:8000`；生产请换正式证书（Caddy/nginx + Let's Encrypt）。
- **限流**：按有效 API Key 指纹（无 Key/非法 Key 按 IP）令牌桶，默认 60 次/分、突发 10；
  超限返回 `429` + `Retry-After`。单进程口径，多副本需换 Redis。
- **审计日志**：`audit/audit.jsonl` 只追加，记录认证失败、任务提交/结束、审批、限流；
  只记 actor/task_id/hash/长度，不落原始问题与 Key。
- **安全响应头**：`nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer`；
  HTTPS 下加 HSTS。
- 配置见 `.env.example` 的 `RATE_LIMIT_*` / `AUDIT_*` / `SSL_*`。

代码里的 `API_AUTH_ENABLED` 默认 `false`（兼容本地 stub 与单测）；`.env.example`
默认 `true` 并放占位 Key。**共享或对外环境必须开启鉴权。**

> 已有数据卷的环境**不会**自动执行新增的初始化 SQL（initdb 只在空卷跑一次）。
> 新增脚本需手工执行，例如长期记忆表：
> `docker exec -i ma-postgres psql -U multiagent -d multiagent < docker/initdb/04-memory-schema.sql`

## 轻量控制台（W13）

服务启动后打开 **http://localhost:8000/console/**：

- 填写 API Key（保存在浏览器 `sessionStorage`，不落服务端）；
- 提交任务、任务列表、SSE 实时进度、结论/关键发现/文档、评审；
- **DAG 可视化**（节点按依赖分层，状态着色）、节点耗时与 Token；
- Token/成本指标（总览 + 按 Agent + 按调用点）；
- 待审批任务可直接批准/驳回（只有 admin Key 能成功）。

无前端构建链（原生 HTML/CSS/JS），静态文件在 `src/multiagent/api/console/`。
完整 React + React Flow 版本仍延后到扩员后。

## 可观测（W10）

- **结构化日志**：`multiagent` 命名空间输出 JSON（`LOG_FORMAT=json`，可切 `text`）。
  每条日志带 `task_id`；Agent/工具日志还带 `node_id`/`agent_role`。
  **不记录原始 prompt/输出/密钥**，只记长度、SQL 哈希、token、耗时。
- **成本计量**：每次模型调用按 `LLM_COST_PER_1K` 的单价汇总到任务结果
  （`cost` / `cost_by_route` / `cost_by_agent`）和评测报告；单位是本地端点**积分**，不是货币。
- **进程内指标**：`GET /metrics`（需要 `tasks:read`）返回任务状态、Token 与成本：

```bash
curl -s localhost:8000/metrics -H "Authorization: Bearer $API_KEY"
```

单进程口径、重启清零；OTel / Prometheus / 跨进程持久化指标留到部署阶段。

## 开发约定

1. **契约先行**：改动数据结构前先改 `架构设想/接口契约/` 的 Schema，再改 `src/multiagent/contracts/`。
2. **空值不输出**：可选字段为空时序列化会省略（`exclude_none=True`），以匹配契约中非 nullable 字段的定义。
3. **降级优先**：任何外部依赖调用都必须有降级路径；模型调用统一走 `gateway`，不要在业务代码里直接发 HTTP。
4. **每个稳定功能立即提交**，并用 `git tag` 标记里程碑，保证可回档。
