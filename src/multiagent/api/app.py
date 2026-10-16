"""最小交互入口：REST 提交任务 + SSE 流式进度。

对应《产品需求》1.2.1 的「多渠道接入网关」在 MVP 阶段的形态——只做 REST 一种渠道，
IM / Webhook / Web 控制台延后（见《实施路线图》W3 范围收敛）。

设计说明：
- **提交与执行解耦**：`POST /tasks` 立即返回 task_id，分析在后台跑；
  客户端用 `GET /tasks/{id}` 轮询结果、或用 `GET /tasks/{id}/events` 订阅进度。
  这是 SSE 能实时推送的前提。
- **事件先缓冲再补发**：晚连上来的订阅者也能看到完整过程（见 tasks.py）。
- **健康检查分项报告**：数据库、Redis、模型网关分别探测。整体 ok/fail 会掩盖
  「到底哪一项挂了」，排障时需要分项。

未做（如实记录）：鉴权、限流、CORS、任务状态持久化。前者属阶段三安全加固，
后者属 W7。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..graph import AnalysisPipeline
from ..graph.pipeline import PipelineOutcome, new_task_event
from .tasks import TaskRecord, TaskRegistry, _now_iso

#: 流水线工厂签名：(settings, on_event) -> 具备 run(task_event) 的对象。
#: 抽出来是为了让测试注入假流水线，不必真的调模型。
PipelineFactory = Callable[..., Any]


class SubmitRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="自然语言分析需求")


class SubmitResponse(BaseModel):
    task_id: str
    status: str
    poll_url: str
    events_url: str


def _default_pipeline_factory(settings: Settings, on_event) -> AnalysisPipeline:
    return AnalysisPipeline(settings, on_event=on_event)


def _outcome_to_result(outcome: PipelineOutcome) -> dict[str, Any]:
    # 按 agent_role 取值：动态 DAG 下 node_id 可变，写死 "analyst_1" 会取不到
    outputs = outcome.merged_outputs_for_role("analyst")
    result: dict[str, Any] = {
        "status": outcome.status,
        "summary": outcome.summary,
        "key_findings": outputs.get("key_findings", []),
        "insufficient_data": outputs.get("insufficient_data"),
        "sql": outputs.get("sql"),
        "sql_hash": outputs.get("sql_hash"),
        "sql_attempts": outputs.get("sql_attempts"),
        "row_count": outputs.get("row_count"),
        "total_tokens": outcome.total_tokens,
        "duration_ms": outcome.duration_ms,
        "tokens_by_route": outcome.tokens_by_route,
        "nodes": [
            {
                "node_id": r.get("node_id"),
                "agent_role": r.get("agent_role"),
                "status": r.get("status"),
                "duration_ms": r.get("duration_ms"),
                "tokens": (r.get("tokens_used") or {}).get("total"),
            }
            for r in outcome.node_results
        ],
    }
    # 评审结论单独暴露（有 reviewer 节点时）：verdict 是「能不能采信」的答案
    review = outcome.merged_outputs_for_role("reviewer")
    if review:
        result["review"] = {
            "verdict": review.get("verdict"),
            "issue_count": review.get("issue_count"),
            "issues": review.get("issues", []),
            "checks": review.get("checks", []),
        }
    # 文档场景（第 2 场景）的产物：写作节点产出的是文档本体，不是摘要
    document = outcome.merged_outputs_for_role("writer")
    if document:
        result["document"] = {
            "title": document.get("title"),
            "markdown": document.get("document_markdown"),
            "material_gaps": document.get("material_gaps", []),
        }
    return result


async def _probe_dependencies(settings: Settings) -> dict[str, Any]:
    """分项探测依赖。每项独立超时，单项挂掉不影响其它项的报告。"""

    async def probe_postgres() -> tuple[str, str]:
        import psycopg

        def sync() -> str:
            with psycopg.connect(settings.postgres_ro_dsn, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return "ok"

        return "postgres", await asyncio.wait_for(asyncio.to_thread(sync), timeout=15)

    async def probe_redis() -> tuple[str, str]:
        import redis

        def sync() -> str:
            client = redis.from_url(settings.redis_url, socket_connect_timeout=5)
            return "ok" if client.ping() else "unexpected"

        return "redis", await asyncio.wait_for(asyncio.to_thread(sync), timeout=15)

    async def probe_gateway() -> tuple[str, str]:
        return "llm_gateway", f"{settings.llm_mode}" + (
            ":stub" if settings.llm_mode == "stub" else ""
        )

    results: dict[str, Any] = {}
    for probe in (probe_postgres, probe_redis, probe_gateway):
        try:
            name, value = await probe()
            results[name] = {"status": "ok", "detail": value}
        except Exception as exc:
            results[probe.__name__.removeprefix("probe_")] = {
                "status": "fail",
                "detail": f"{type(exc).__name__}: {exc}"[:200],
            }
    return results


def create_app(
    *,
    settings: Settings | None = None,
    pipeline_factory: PipelineFactory | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    factory = pipeline_factory or _default_pipeline_factory
    registry = TaskRegistry()

    app = FastAPI(
        title="多 Agent 智能协作平台（MVP 入口）",
        version="0.1.0",
        description="REST 提交分析任务 + SSE 订阅执行进度。",
    )

    async def _execute(record: TaskRecord) -> None:
        async def on_event(name: str, payload: dict[str, Any]) -> None:
            record.publish(name, **payload)
            registry.trim(record)

        try:
            record.status = "running"
            record.publish("task_started", question=record.question)
            pipeline = factory(app_settings, on_event)
            outcome = await pipeline.run(new_task_event(record.question))
            record.result = _outcome_to_result(outcome)
            record.status = "ok" if outcome.status == "ok" else "failed"
            if outcome.status != "ok":
                record.error = outcome.error
        except Exception as exc:  # 兜底：任务异常不得让服务崩掉
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            record.finished_at = _now_iso()
            record.publish(
                "task_finished",
                status=record.status,
                error=record.error,
                summary=(record.result or {}).get("summary", ""),
            )

    # ------------------------------------------------------------------ 路由

    @app.get("/health")
    async def health() -> dict[str, Any]:
        dependencies = await _probe_dependencies(app_settings)
        overall = "ok" if all(d["status"] == "ok" for d in dependencies.values()) else "degraded"
        return {
            "status": overall,
            "dependencies": dependencies,
            "routing": dict(app_settings.llm_routing),
        }

    @app.post("/tasks", response_model=SubmitResponse, status_code=202)
    async def submit(request: SubmitRequest) -> SubmitResponse:
        task_id = str(uuid.uuid4())
        record = registry.create(task_id, request.question)
        # 后台执行，不阻塞提交响应——这是 SSE 能实时推送的前提
        asyncio.create_task(_execute(record))
        return SubmitResponse(
            task_id=task_id,
            status=record.status,
            poll_url=f"/tasks/{task_id}",
            events_url=f"/tasks/{task_id}/events",
        )

    @app.get("/tasks")
    async def list_tasks(limit: int = 20) -> dict[str, Any]:
        return {"tasks": registry.list_recent(limit=limit)}

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> dict[str, Any]:
        record = registry.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        return record.summary()

    @app.get("/tasks/{task_id}/events")
    async def stream_events(task_id: str) -> StreamingResponse:
        record = registry.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        # 补发历史事件：晚连上来的客户端也能看到完整过程
        for past in list(record.events):
            queue.put_nowait(past)
        if record.finished:
            # 已结束：补发完历史（其中含 task_finished）再收尾，不订阅
            queue.put_nowait({"event": "end", "at": _now_iso(), "reason": "already_finished"})
        else:
            record.subscribers.append(queue)

        async def event_stream():
            try:
                while True:
                    item = await queue.get()
                    yield _sse(item)
                    if item.get("event") in ("task_finished", "end"):
                        return
            finally:
                if queue in record.subscribers:
                    record.subscribers.remove(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _sse(payload: dict[str, Any]) -> str:
    """按 SSE 格式编码一条事件。"""
    name = payload.get("event", "message")
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
