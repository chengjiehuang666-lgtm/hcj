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

1. **主备模型同源**：当前 `.env` 中主力与备用都是 `deepseek-flash`（同一供应商、同一端点）。
   此时降级只对「模型级」故障有效（如配额、模型下线），**对供应商整体故障无效**——
   DeepSeek 一旦不可用，两者会一起失败。真正的供应商级failover需要第二个厂商的 key
   （通义千问 / 智谱 GLM，均为 OpenAI 兼容接口，只需改 `.env` 三行）。
   同端点下可退而求其次把备用设为 `deepseek-v4-pro`，至少能规避单模型级的容量问题。
2. **无远端仓库**：回档能力目前仅限本机（见《实施路线图》风险 R12）。

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
