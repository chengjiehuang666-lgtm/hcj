"""配置：集中读取环境变量 / .env。

约定：
- 真实密钥只放在 `.env`（已被 .gitignore 忽略），仓库里只有 `.env.example`。
- `LLM_MODE=stub` 时不需要任何 API key，用于本地跑通链路与测试。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    """单个模型供应商的连接配置。"""

    provider: str
    model: str
    base_url: str
    api_key: SecretStr | None = None

    @property
    def has_key(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value().strip())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 模型网关 ----
    llm_mode: Literal["stub", "live"] = "stub"

    llm_primary_provider: str = "deepseek"
    llm_primary_model: str = "deepseek-chat"
    llm_primary_base_url: str = "https://api.deepseek.com/v1"
    llm_primary_api_key: SecretStr | None = None

    llm_fallback_provider: str = "qwen"
    llm_fallback_model: str = "qwen-plus"
    llm_fallback_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_fallback_api_key: SecretStr | None = None

    # ---- fast 档：机械步骤用的低延迟模型（分级路由）----
    # 上面 primary/fallback 构成 reasoning 档（强推理模型，用于规划与归因）；
    # 本组构成 fast 档，用于「生成 SQL」这类不需要深度思考的调用点。
    # 默认指向非推理型的本地端点——实测一次简单调用 3.39 秒 / 87 token，
    # 比推理型低一到两个数量级；本地不可用时自动降级到 fast 档的备模型。
    llm_fast_primary_provider: str = "zhipu"
    llm_fast_primary_model: str = "glm-5.3"
    llm_fast_primary_base_url: str = "http://127.0.0.1:7863/v1"
    llm_fast_primary_api_key: SecretStr | None = None

    llm_fast_fallback_provider: str = "deepseek"
    llm_fast_fallback_model: str = "deepseek-flash"
    llm_fast_fallback_base_url: str = "https://api.deepseek.com/v1"
    llm_fast_fallback_api_key: SecretStr | None = None

    # ---- 分级路由表 ----
    # 调用点 -> 档位（reasoning | fast）。改路由只需改这里，不用改代码。
    # 调用点名见 gateway/router.py 的 ROUTE_* 常量。
    # 默认全部走 reasoning，即与引入分级路由之前的行为完全一致——先建机制、
    # 再用测量数据决定哪些调用点切到 fast（见《实施路线图》3.7）。
    llm_default_tier: str = "reasoning"
    llm_routing: dict[str, str] = Field(
        default_factory=lambda: {
            "planner": "reasoning",
            "analyst_sql": "reasoning",
            "analyst_conclude": "reasoning",
            "reviewer": "reasoning",
            "writer": "reasoning",
        }
    )

    # ---- 基础设施（与 docker-compose.yml 默认值一致）----
    postgres_dsn: str = "postgresql://multiagent:multiagent@localhost:5432/multiagent"
    # SQL 沙箱专用：只读角色，Agent 执行查询一律用它，不用上面的应用主账号。
    # 该角色的权限由 docker/initdb/02-demo-data.sql 授予（仅 SELECT + 5s 语句超时）。
    postgres_ro_dsn: str = "postgresql://analyst_ro:analyst_ro@localhost:5432/multiagent"
    redis_url: str = "redis://localhost:6379/0"

    # ---- SQL 沙箱约束 ----
    sql_max_rows: int = 500
    sql_timeout_seconds: int = 10

    # ---- Python 沙箱约束（见 tools/python.py）----
    # auto：能用容器就用容器，否则降到子进程（隔离强度显著更弱，结果里会标注）
    # container：强制容器，不可用则直接报错——评测/生产建议用它，避免静默降级
    python_sandbox_mode: Literal["auto", "container", "subprocess"] = "auto"
    python_sandbox_image: str = "multiagent-python-sandbox:latest"
    # Windows 上 Docker Desktop 多为用户级安装、不在 PATH 里，
    # 此时在此填 docker 可执行文件的完整路径。
    python_sandbox_docker_cli: str = "docker"
    python_sandbox_timeout_seconds: int = 30
    python_sandbox_memory_mb: int = 512
    python_sandbox_cpus: float = 1.0

    # ---- 表结构自省的白名单/黑名单 ----
    # 排除框架自身建的表。LangGraph 的 PostgresSaver 会在 public schema 下建
    # checkpoints / checkpoint_blobs / checkpoint_writes / checkpoint_migrations 四张表；
    # 把它们喂给模型既是噪音（占表结构提示词约 70% 的字符），也可能诱导模型去查框架表。
    schema_exclude_table_prefixes: list[str] = Field(
        default_factory=lambda: ["checkpoint", "langgraph", "pg_", "sql_", "memory"]
    )

    # ---- 知识库 / RAG ----
    # embedding 用**本地**模型（fastembed + ONNX），不依赖在线服务：
    # 检索每次都要用，走本地没有按次成本，也不受供应商可用性影响。
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    # 模型缓存目录。**不要落在系统临时目录**：被清理后要重新下载，
    # 且会占系统盘（本机 C 盘仅剩 2.8G）。默认放在项目内，已 gitignore。
    embedding_cache_dir: str = ".model_cache"
    # 模型下载镜像。直连 HuggingFace 在当前网络不可达，必须走镜像。
    hf_endpoint: str = "https://hf-mirror.com"
    rag_chunk_size: int = 320
    rag_chunk_overlap: int = 60
    rag_top_k: int = 4
    # 混合检索里向量分的权重，其余给关键词（pg_trgm）分
    rag_vector_weight: float = 0.7

    # ---- 长期记忆（W6）----
    #: 是否把「任务成功 + 评审未否决」的结论自动沉淀进长期记忆。
    #: 沉淀是 best-effort：失败只记录错误，不影响任务本身。
    memory_sink_enabled: bool = True
    #: 记忆检索条数上限（混合加权与知识库同一套公式与权重）
    memory_top_k: int = 3

    # ---- 安全基线（W9）----
    #: 输入侧 Prompt Injection / 敏感词过滤（默认开启；规则误伤时可整体关闭）
    security_enabled: bool = True
    #: 输出侧 DLP 脱敏（身份证 / 手机号 / 银行卡 / 密钥）
    security_dlp_enabled: bool = True
    #: 敏感词列表（逗号分隔）。默认只收凭据/身份标识类词，避免误伤正常分析问题。
    security_blocked_terms: str = (
        "密码,口令,私钥,api key,apikey,secret key,access token,"
        "身份证号,银行卡号,信用卡号,cvv"
    )
    #: 是否启用 API Key 鉴权。代码默认关闭以兼容本地 stub / 单测；
    #: `.env.example` 默认开启——**共享或对外环境必须开启**。
    api_auth_enabled: bool = False
    #: `key:role,key:role`；角色 = admin | operator | viewer。
    #: 启用鉴权但未配置任何 Key 时，API 启动会直接失败（fail closed）。
    api_keys: str = ""

    # ---- 可观测（W10）----
    #: 日志级别与格式；json 便于采集，text 便于本地阅读。
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    #: 模型单价（每 1k token）。本地端点的单位是**积分**，不是货币；
    #: 用于相对成本比较，不参与计费。值为 0 表示未配置（只记 token）。
    llm_cost_per_1k: dict[str, float] = Field(
        default_factory=lambda: {
            "deepseek-v4.1-flash": 0.0355,
            "glm-5.3": 0.1699,
            "deepseek-v4-pro": 0.1309,
        }
    )
    cost_unit: str = "credits"

    # ---- 执行预算默认值（对应契约 TaskEvent.budget）----
    default_max_steps: int = 30
    # 任务级 Token 熔断阈值，即《实施路线图》待定项 D2。
    # 定值规则：评测实测最大 × 1.5 安全系数、向上取整到万。
    #   W4（无 reviewer，20 条）：最大 20,550 → 40,000
    #   W6（含 reviewer，27 条）：最大 37,842 → 60,000（实测见路线图 §3.11）
    default_max_tokens: int = 60_000
    default_max_retries: int = 2
    default_task_timeout_seconds: int = 300
    #: 动态重规划次数上限（W7）。触发条件=节点重试耗尽或 Agent 返回 needs_replan；
    #: 超出后任务以最后一次失败终止。1 次是「先量后改」下的保守起点。
    max_replans: int = 1

    # 单次模型调用的输出上限（reasoning 档）。
    # max_tokens 只是上限、不用不收费，所以给足以免被截断。
    # 实测（云端 deepseek-flash）：生成结构化 DAG 这类重推理任务需 ~2,400 输出 token，
    # 4096 会因思考吃满而截断成空内容。
    # W6 扩到 16384：本地 cn:deepseek-v4.1-flash 思考更重——实测记忆回归用例的某次
    # 调用思考消耗满 8192 仍无正文（finish_reason=length、内容为空），任务失败。
    # 推理档的截断放宽重试**帮不上忙**（requested == cap 时不会放宽），所以上限本身
    # 必须留足。
    llm_max_output_tokens: int = 16384

    # fast 档的输出上限。**这是 fast 档延迟的主要控制点**，不是提示词长度。
    # 实测（输入完全相同，仅改上限）：
    #   max_tokens=8192 -> 输出 599 token（其中思考 453）、耗时 10.7 秒
    #   max_tokens=2048 -> 输出 117 token（思考 0）、耗时 4.5 秒
    # 即：给本地模型过大的上限等于邀请它无限思考。取数只需一条 SQL + 一句话理由，
    # 2048 足够，同时把「思考失控」这个失效模式从根上掐掉。
    llm_fast_max_output_tokens: int = 2048

    # ---- 派生配置 ----
    @property
    def primary(self) -> ProviderConfig:
        return ProviderConfig(
            provider=self.llm_primary_provider,
            model=self.llm_primary_model,
            base_url=self.llm_primary_base_url,
            api_key=self.llm_primary_api_key,
        )

    @property
    def fallback(self) -> ProviderConfig:
        return ProviderConfig(
            provider=self.llm_fallback_provider,
            model=self.llm_fallback_model,
            base_url=self.llm_fallback_base_url,
            api_key=self.llm_fallback_api_key,
        )

    @property
    def fast_primary(self) -> ProviderConfig:
        return ProviderConfig(
            provider=self.llm_fast_primary_provider,
            model=self.llm_fast_primary_model,
            base_url=self.llm_fast_primary_base_url,
            api_key=self.llm_fast_primary_api_key,
        )

    @property
    def fast_fallback(self) -> ProviderConfig:
        return ProviderConfig(
            provider=self.llm_fast_fallback_provider,
            model=self.llm_fast_fallback_model,
            base_url=self.llm_fast_fallback_base_url,
            api_key=self.llm_fast_fallback_api_key,
        )

    def tier_for(self, route: str) -> str:
        """查询某个调用点走哪个档位（与 ModelRouter.tier_for 的判定保持一致）。"""
        return self.llm_routing.get(route, self.llm_default_tier)

    @property
    def blocked_terms(self) -> tuple[str, ...]:
        """敏感词列表（逗号分隔 → 元组）。"""
        return tuple(
            term.strip() for term in self.security_blocked_terms.split(",") if term.strip()
        )

    @property
    def api_key_map(self) -> dict[str, str]:
        """解析后的 {API Key: 角色}；格式错误时抛 ValueError。"""
        from .security.auth import parse_api_keys

        return parse_api_keys(self.api_keys)

    @property
    def default_client_key(self) -> str | None:
        """第一个可用客户端 Key（优先 operator，其次 admin/viewer）。

        给压测/脚本用，不用于对外暴露。未配置或格式错误时返回 None。
        """
        try:
            keys = self.api_key_map
        except ValueError:
            return None
        for role in ("operator", "admin", "viewer"):
            for key, value in keys.items():
                if value == role:
                    return key
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例，避免重复解析 .env。"""
    return Settings()
