"""Writer Agent 测试：产物形状、素材进提示词、无素材时的降级与诚实标注。

用假网关注入固定文档，不调用真实模型。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiagent.agents import WriterAgent  # noqa: E402
from multiagent.agents.base import AgentContext  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.contracts import AgentInvocation, InvocationBudget, ResultStatus  # noqa: E402
from multiagent.gateway import LLMResponse  # noqa: E402
from multiagent.tools import SqlSandbox  # noqa: E402

settings = Settings(_env_file=None)


class CapturingGateway:
    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)
        self.prompts: list[str] = []
        self.routes_seen: list[str | None] = []

    async def complete(self, messages, *, max_tokens=None, temperature=None, route=None):
        self.routes_seen.append(route)
        self.prompts.append("\n".join(m.content for m in messages))
        if not self._contents:
            raise AssertionError("假网关的预设响应已用尽")
        return LLMResponse(
            content=self._contents.pop(0),
            provider="fake",
            model="fake-model",
            input_tokens=100,
            output_tokens=200,
            finish_reason="stop",
        )


def _ctx(gateway) -> AgentContext:
    return AgentContext(settings=settings, gateway=gateway, sandbox=SqlSandbox(settings))


_DOC_JSON = json.dumps(
    {
        "title": "Q2 复盘简报",
        "document_markdown": (
            "# Q2 复盘简报\n\n摘要：Q2 全区域达成率 96%（来源：《2026 年 Q2 销售复盘纪要》）。\n\n"
            "## 区域表现\n华南超额 4%，华东缺口 7%。\n\n## 建议\n加强会员权益。"
        ),
        "used_sources": ["《2026 年 Q2 销售复盘纪要》"],
        "material_gaps": [],
    },
    ensure_ascii=False,
)

_KNOWLEDGE = [
    {
        "doc_id": "q2-2026-review",
        "title": "2026 年 Q2 销售复盘纪要",
        "chunk_index": 0,
        "content": "Q2 全区域销售额达成率 96%，其中华南区超额 4%、华东区缺口 7%。",
        "score": 0.9,
    }
]


def _inv(knowledge: list[dict] | None = None) -> AgentInvocation:
    return AgentInvocation(
        task_id="t-1",
        node_id="writer_1",
        attempt=1,
        agent_role="writer",
        trace_id="t-1",
        input={
            "question": "写一份面向管理层的 Q2 复盘简报",
            "acceptance": ["文档包含达成率这一事实"],
            "node_id": "writer_1",
            "upstream_nodes": ["retriever_1"],
            "knowledge": knowledge if knowledge is not None else _KNOWLEDGE,
        },
        budget=InvocationBudget(max_tokens=8192, timeout_seconds=300),
    )


def test_writer_outputs_shape() -> None:
    gateway = CapturingGateway(_DOC_JSON)
    result = asyncio.run(WriterAgent().invoke(_inv(), _ctx(gateway)))

    assert result.status is ResultStatus.ok
    out = result.outputs
    assert out["title"] == "Q2 复盘简报"
    assert out["section_count"] == 2
    assert out["char_count"] > 50
    assert out["used_sources"] == ["《2026 年 Q2 销售复盘纪要》"]
    assert "material_gaps" in out
    # writer 不得顶替业务结论（summary 键留给 analyst 链路）
    assert "summary" not in out
    assert result.blackboard_writes == ["writer_1.document"]
    assert gateway.routes_seen == ["writer"]


def test_writer_prompt_carries_materials_and_rule() -> None:
    gateway = CapturingGateway(_DOC_JSON)
    asyncio.run(WriterAgent().invoke(_inv(), _ctx(gateway)))

    prompt = gateway.prompts[0]
    assert "达成率 96%" in prompt  # 素材正文
    assert "《2026 年 Q2 销售复盘纪要》" in prompt  # 素材出处
    assert "文档包含达成率这一事实" in prompt  # 验收标准
    assert "不得编造" in prompt  # 数字纪律


def test_writer_without_materials_is_honest() -> None:
    gateway = CapturingGateway(_DOC_JSON)
    result = asyncio.run(WriterAgent().invoke(_inv(knowledge=[]), _ctx(gateway)))

    assert result.status is ResultStatus.ok
    assert "（未检索到任何素材）" in gateway.prompts[0]
    # 无素材时置信度显著降低 + 来源里如实记录「无命中」
    assert result.confidence == 0.3
    assert any("无命中" in s.ref for s in result.sources)


def test_writer_sources_reference_material_chunks() -> None:
    gateway = CapturingGateway(_DOC_JSON)
    result = asyncio.run(WriterAgent().invoke(_inv(), _ctx(gateway)))

    refs = [s.ref for s in result.sources]
    assert any("知识库:q2-2026-review#0" in r for r in refs)


def test_writer_invalid_output_fails_with_schema_error() -> None:
    gateway = CapturingGateway("不是 JSON", "还不是 JSON")  # 两次机会都用掉
    result = asyncio.run(WriterAgent().invoke(_inv(), _ctx(gateway)))

    assert result.status is ResultStatus.failed
    assert result.error.code == "SCHEMA_MISMATCH"
    assert result.error.retryable is True
