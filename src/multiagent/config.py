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
        default_factory=lambda: ["checkpoint", "langgraph", "pg_", "sql_"]
    )

    # ---- 执行预算默认值（对应契约 TaskEvent.budget）----
    default_max_steps: int = 30
    default_max_tokens: int = 200_000  # 待评测集校准，见《实施路线图》待定项 D2
    default_max_retries: int = 2
    default_task_timeout_seconds: int = 300

    # 单次模型调用的输出上限（reasoning 档）。
    # max_tokens 只是上限、不用不收费，所以给足以免被截断。
    # 实测：推理型模型在「生成结构化 DAG」这类重推理任务上需 ~2,400 输出 token，
    # 4096 会因思考吃满而截断成空内容。
    llm_max_output_tokens: int = 8192

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例，避免重复解析 .env。"""
    return Settings()
