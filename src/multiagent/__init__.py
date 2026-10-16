"""多 Agent 智能协作平台。

分层：
- `contracts` 契约的 Pydantic 实现（对应 架构设想/接口契约/）
- `gateway`   LLM 网关（供应商无关 + 主备降级）
- `config`    配置（读 .env）
"""

__version__ = "0.1.0"
