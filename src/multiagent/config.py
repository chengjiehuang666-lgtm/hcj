"""配置：集中读取环境变量 / .env。

约定：
- 真实密钥只放在 `.env`（已被 .gitignore 忽略），仓库里只有 `.env.example`。
- `LLM_MODE=stub` 时不需要任何 API key，用于本地跑通链路与测试。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
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

    # ---- 基础设施（与 docker-compose.yml 默认值一致）----
    postgres_dsn: str = "postgresql://multiagent:multiagent@localhost:5432/multiagent"
    # SQL 沙箱专用：只读角色，Agent 执行查询一律用它，不用上面的应用主账号。
    # 该角色的权限由 docker/initdb/02-demo-data.sql 授予（仅 SELECT + 5s 语句超时）。
    postgres_ro_dsn: str = "postgresql://analyst_ro:analyst_ro@localhost:5432/multiagent"
    redis_url: str = "redis://localhost:6379/0"

    # ---- SQL 沙箱约束 ----
    sql_max_rows: int = 500
    sql_timeout_seconds: int = 10

    # ---- 执行预算默认值（对应契约 TaskEvent.budget）----
    default_max_steps: int = 30
    default_max_tokens: int = 200_000  # 待评测集校准，见《实施路线图》待定项 D2
    default_max_retries: int = 2
    default_task_timeout_seconds: int = 300

    # 单次模型调用的输出上限。
    # max_tokens 只是上限、不用不收费，所以给足以免被截断；
    # 真正的成本控制由上面的任务级 default_max_tokens 熔断负责。
    # 实测：推理型模型（deepseek-flash）在「生成结构化 DAG」这类重推理任务上
    # 需 ~2400 输出 token（其中思考 ~2100），4096 会因思考吃满而截断成空内容。
    llm_max_output_tokens: int = 8192

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例，避免重复解析 .env。"""
    return Settings()
