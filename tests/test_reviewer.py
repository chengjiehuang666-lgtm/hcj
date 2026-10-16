"""Reviewer Agent 测试：核对输入、verdict → 条目状态映射、乐观锁闭环。

用假网关注入固定 verdict，不调用真实模型。重点钉住：

- `fail` 才拒绝条目（长期记忆的沉淀门槛语义），`pass_with_issues` 仍算通过；
- `blackboard_updates` 的 expected_version 取自**本轮实际读到的版本**（乐观锁闭环）；
- reviewer 不产出 `summary` 键——最终结论必须仍是 analyst 的产物。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiagent.agents import ReviewerAgent  # noqa: E402
from multiagent.agents.base import AgentContext  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.contracts import AgentInvocation, InvocationBudget, ResultStatus  # noqa: E402
from multiagent.gateway import LLMResponse  # noqa: E402
from multiagent.tools import SqlSandbox  # noqa: E402

settings = Settings(_env_file=None)


class CapturingGateway:
    """返回预设内容，并把提示词留档——用于断言「材料真的进了提示词」。"""

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


def _entry(node_id: str, payload: dict, *, version: int = 1) -> dict:
    return {
        "task_id": "t-1",
        "schema_version": "1.0.0",
        "key": f"{node_id}.report",
        "version": version,
        "payload": payload,
        "written_by": {
            "agent_role": "analyst",
            "agent_version": "0.3.0",
            "node_id": node_id,
            "attempt": 1,
        },
        "written_at": "2026-10-16T12:00:00+08:00",
        "status": "draft",
        "sources": [{"type": "sql", "ref": "sha256:abc"}],
    }


def _inv(
    *,
    blackboard: dict | None = None,
    upstream_nodes: list[str] | None = None,
) -> AgentInvocation:
    return AgentInvocation(
        task_id="t-1",
        node_id="reviewer_1",
        attempt=1,
        agent_role="reviewer",
        trace_id="t-1",
        input={
            "question": "分析上月华东区销售额下滑原因",
            "acceptance": ["核对每条结论的数字都有查询结果支撑"],
            "node_id": "reviewer_1",
            "upstream_nodes": upstream_nodes if upstream_nodes is not None else ["analyst_1"],
            "blackboard": (
                blackboard
                if blackboard is not None
                else {
                    "analyst_1.report": _entry(
                        "analyst_1",
                        {"summary": "环比下滑 20.85%", "key_findings": ["线上数码降 37.6%"]},
                    )
                }
            ),
            "upstream_sources": {
                "analyst_1": [{"type": "sql", "ref": "sha256:abc"}]
            },
        },
        budget=InvocationBudget(max_tokens=8192, timeout_seconds=300),
    )


def _verdict_json(verdict: str, *, severity: str | None = None) -> str:
    issues = (
        [{"severity": severity, "description": "数字出处不清晰"}]
        if severity
        else []
    )
    return json.dumps(
        {
            "verdict": verdict,
            "checks": [
                {
                    "criterion": "核对每条结论的数字都有查询结果支撑",
                    "result": "met" if verdict != "fail" else "unmet",
                    "evidence": "20.85% 出现在结果行" if verdict != "fail" else "",
                }
            ],
            "issues": issues,
            "confidence": 0.9,
        },
        ensure_ascii=False,
    )


def test_pass_validates_entries() -> None:
    gateway = CapturingGateway(_verdict_json("pass"))
    result = asyncio.run(ReviewerAgent().invoke(_inv(), _ctx(gateway)))

    assert result.status is ResultStatus.ok
    out = result.outputs
    assert out["verdict"] == "pass"
    assert out["blackboard_updates"] == [
        {"key": "analyst_1.report", "expected_version": 1, "status": "validated"}
    ]
    assert out["reviewed_nodes"] == ["analyst_1"]
    # reviewer 不得顶替业务结论：汇点取数会退化到 analyst 的 summary
    assert "summary" not in out
    # 调用点登记为 reviewer（分级路由依据）
    assert gateway.routes_seen == ["reviewer"]
    assert result.blackboard_writes == ["reviewer_1.review"]


def test_fail_rejects_entries() -> None:
    gateway = CapturingGateway(_verdict_json("fail", severity="blocker"))
    result = asyncio.run(ReviewerAgent().invoke(_inv(), _ctx(gateway)))

    assert result.outputs["blackboard_updates"][0]["status"] == "rejected"
    assert result.outputs["issue_count"] == 1


def test_pass_with_issues_still_validates() -> None:
    """沉淀门槛的语义边界：只有 fail 才拒绝，pass_with_issues 仍算通过。"""
    gateway = CapturingGateway(_verdict_json("pass_with_issues", severity="minor"))
    result = asyncio.run(ReviewerAgent().invoke(_inv(), _ctx(gateway)))

    assert result.outputs["verdict"] == "pass_with_issues"
    assert result.outputs["blackboard_updates"][0]["status"] == "validated"


def test_update_submits_version_actually_read() -> None:
    """乐观锁闭环：expected_version 必须来自本轮读到的条目版本，而不是恒为 1。"""
    blackboard = {
        "analyst_1.report": _entry("analyst_1", {"summary": "x"}, version=3),
    }
    gateway = CapturingGateway(_verdict_json("pass"))
    result = asyncio.run(
        ReviewerAgent().invoke(_inv(blackboard=blackboard), _ctx(gateway))
    )
    assert result.outputs["blackboard_updates"] == [
        {"key": "analyst_1.report", "expected_version": 3, "status": "validated"}
    ]


def test_prompt_carries_material_acceptance_and_evidence() -> None:
    gateway = CapturingGateway(_verdict_json("pass"))
    asyncio.run(ReviewerAgent().invoke(_inv(), _ctx(gateway)))

    prompt = gateway.prompts[0]
    assert "环比下滑 20.85%" in prompt  # 上游产物
    assert "核对每条结论的数字都有查询结果支撑" in prompt  # 验收标准
    assert "sha256:abc" in prompt  # 证据明细
    assert "analyst_1" in prompt


def test_missing_upstream_fails_without_model_call() -> None:
    gateway = CapturingGateway()  # 没有预设响应：一旦调用即 AssertionError
    result = asyncio.run(
        ReviewerAgent().invoke(
            _inv(upstream_nodes=[], blackboard={}), _ctx(gateway)
        )
    )

    assert result.status is ResultStatus.failed
    assert result.error.code == "BAD_INPUT"
    assert result.error.retryable is False
    assert gateway.prompts == []
