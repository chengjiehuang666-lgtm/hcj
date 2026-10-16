# 多 Agent 接入协作项目

企业级多 Agent 智能协作平台。当前处于**阶段一 MVP**（单人全职开发，计划与范围见《实施路线图》）。

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
│   └── gateway/               # 大模型网关（供应商无关 + 降级）
├── tests/                     # 契约一致性测试、网关测试
├── docker-compose.yml         # PostgreSQL(pgvector) + Redis
├── .env.example               # 环境变量模板（真实密钥放 .env，不入库）
└── pyproject.toml
```

## 环境前置条件

| 依赖 | 当前状态 | 说明 |
| :--- | :--- | :--- |
| Python | ✅ 本机 3.13.7 | 已确认使用 3.13（文档原写 3.11）；`requires-python = ">=3.11"` |
| Docker Desktop | ⚠️ **已安装，但 engine 无法运行** | 4.95.0，用户级安装于 `%LOCALAPPDATA%\Programs\DockerDesktop`；CLI v29.8.2 正常且在 PATH 中 |
| WSL2 | ⚠️ **无法启动** | 报 `HCS_E_HYPERV_NOT_INSTALLED`，见下方「阻塞项」 |

### 🚧 阻塞项：Docker engine 起不来（需管理员 + 重启）

诊断结论：**CPU 的 BIOS 虚拟化已经开启**（`VirtualizationFirmwareEnabled=True`），
但 Windows 的虚拟化功能未启用（`HypervisorPresent=False`），因此 WSL2 无法启动，
Docker Desktop 的 WSL2 后端也无法工作。Docker CLI 本身是好的，只是连不上 daemon
（`failed to connect to the docker API at npipe:////./pipe/docker_engine`）。

修复步骤（**以管理员身份**打开 PowerShell 或 CMD）：

```bat
dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

然后**重启电脑**，启动 Docker Desktop，再用 `docker info` 确认 daemon 已就绪。

> 在 Docker 未就绪前，`docker-compose.yml` 无法启动。但**契约测试与网关测试都不需要它**
> （`pytest -q` 或 `python tests/test_gateway.py` 可直接跑），模型调用走 `LLM_MODE=live` 也不需要数据库。

## 已知限制

1. **本地端点同时承担两个角色，是当前的可用性单点**：`.env` 中
   - reasoning 档：主 = DeepSeek `deepseek-flash`（推理型），备 = 本地 `glm-5.3`（非推理型）
   - fast 档：主 = 本地 `glm-5.3`，备 = DeepSeek `deepseek-flash`

   即本地端点既是 reasoning 档的备用，又是 fast 档的主力。
   **本地服务不可用时不会失败**：fast 档会自动降级到 DeepSeek，只是时延回退到推理型水平。

   另注：智谱**官方**端点（`https://open.bigmodel.cn/api/paas/v4`）的账号**余额为 0**，
   调用返回 `HTTP 429 / code 1113「余额不足或无可用资源包」`（key 本身有效，
   `GET /models` 返回 200）。因此备用没有指向官方端点，而是指向了本地服务。
   若想恢复官方端点作为备用，需先充值。
2. **无远端仓库**：回档能力目前仅限本机（见《实施路线图》风险 R12）。
3. **时延与成功率相互牵制，本轮选择「保质量」**：
   - **✅ 成功率 24/24 = 100%**（引入知识库 RAG 后由 85% 回升），远超 M1 的 80% 门槛
   - ⚠️ **P95 端到端时延 74.3 秒**，重新超过 M1 的 60 秒目标（P50 30.8 秒仍达标）
   - 原因：知识片段同时进入取数与归因两个提示词，平均 token 从 5,659 升到 8,903；
     但正是这个变化治好了原先失败的用例——**RAG 用「时延」换「质量」**

   三轮实测对比与超时用例清单见《实施路线图》3.7 与 3.9。

   **下一步候选手段**：压缩知识片段预算（4 段 × 400 字符 × 两个提示词）、
   降低 SQL 修补轮触发率。遵循「先量后改」，不盲调。
4. **推理型模型的 `max_tokens` 是「共享额度」**：`deepseek-flash` 会先产生
   `reasoning_content` 再产生 `content`，而 `max_tokens` **由两者共享**。额度偏小时
   思考会吃掉全部额度，导致 `content` 为空且 `finish_reason=length`
   （实测同一句请求：`max_tokens=20` 全被思考消耗、输出为空；`512` 时才正常返回）。

   网关对此**显式报错**（`EmptyCompletionError`，提示调大 `max_tokens`），
   不会静默返回空字符串——静默失败会让上游误以为模型「没话说」。
   `LLMResponse.reasoning_tokens` 单独记录了思考 token，供成本计量使用。

## 快速开始

```bash
# 1. 环境变量
cp .env.example .env          # Windows: copy .env.example .env
                              # 默认 LLM_MODE=stub，不需要 API key

# 2. 安装（建议虚拟环境）
python -m venv .venv
# Windows: .venv\Scripts\activate      Git Bash: source .venv/Scripts/activate
pip install -e ".[dev]"

# 3. 跑测试（契约一致性 + 网关降级）
pytest -q

# 4. 启动依赖服务（需先有 Docker）
docker compose up -d
```

## 开发约定

1. **契约先行**：改动数据结构前先改 `架构设想/接口契约/` 的 Schema，再改 `src/multiagent/contracts/`。
2. **空值不输出**：可选字段为空时序列化会省略（`exclude_none=True`），以匹配契约中非 nullable 字段的定义。
3. **降级优先**：任何外部依赖调用都必须有降级路径；模型调用统一走 `gateway`，不要在业务代码里直接发 HTTP。
4. **每个稳定功能立即提交**，并用 `git tag` 标记里程碑，保证可回档。
