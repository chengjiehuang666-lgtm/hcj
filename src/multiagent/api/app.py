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

未做（如实记录）：限流、CORS、任务状态持久化。鉴权与输入/输出安全基线见 W9
（API Key + admin/operator/viewer 三角色、输入过滤、输出 DLP 脱敏）。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..graph import AnalysisPipeline
from ..graph.pipeline import PipelineOutcome, new_task_event
from ..security import (
    PERMISSIONS,
    Principal,
    authenticate,
    parse_api_keys,
    redact_structure,
    redact_text,
    scan_input,
)
from .tasks import TERMINAL_STATUSES, TaskRecord, TaskRegistry, _now_iso

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


class ApprovalBody(BaseModel):
    """HITL 审批决策（POST /tasks/{id}/approval 的请求体）。"""

    decision: Literal["approved", "rejected", "rewritten"]
    comment: str = ""
    #: decision=rewritten 时必填：替换后的节点产物
    outputs: dict[str, Any] | None = None
    approver: str = "local_dev"


def _default_pipeline_factory(settings: Settings, on_event) -> AnalysisPipeline:
    return AnalysisPipeline(settings, on_event=on_event)


def _outcome_to_result(outcome: PipelineOutcome, *, dlp: bool = True) -> dict[str, Any]:
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
        # 失败/驳回的稳定错误码与可重试性（熔断：BUDGET_EXCEEDED/STEP_LIMIT/...）
        "error": outcome.error,
        "error_code": outcome.error_code,
        # DAG 版本（1 = 首版；重规划后递增）
        "revision": outcome.revision,
        # HITL：等待审批时的请求明细（节点、产物、验收标准）
        "approval_requests": outcome.approval_requests,
        "approvals": outcome.approvals,
        # 本次任务沉淀进长期记忆的条目 id（未沉淀为 None；沉淀失败只影响该字段）
        "memory_sunk": outcome.memory_sunk,
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
    # 输出侧 DLP：只处理用户可见边界，内部 node_results 保持原值供执行/评测
    return redact_structure(result) if dlp else result


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
    #: 后台任务必须保存强引用：fire-and-forget 的 Task 存在被 GC 回收的风险
    background_tasks: set = set()

    # ---- 鉴权（W9）：启用时 fail closed -------------------------------------
    auth_enabled = bool(app_settings.api_auth_enabled)
    key_map: dict[str, str] = {}
    if auth_enabled:
        try:
            key_map = parse_api_keys(app_settings.api_keys)
        except ValueError as exc:
            raise RuntimeError(f"API_KEYS 配置无效：{exc}") from exc
        if not key_map:
            raise RuntimeError(
                "API_AUTH_ENABLED=true 但 API_KEYS 为空——拒绝启动（fail closed）"
            )

    def require(permission: str):
        """FastAPI 依赖：校验 Bearer Key 与角色权限。"""

        async def dependency(request: Request) -> Principal:
            if not auth_enabled:
                return Principal(key_id="anonymous", role="admin")
            token = _bearer_token(request)
            # SSE 的 EventSource 不能自定义 Header；只对该端点允许 query 传递。
            # 代价是 Key 可能进入访问日志——本地单租户基线的明确取舍。
            if token is None and request.url.path.endswith("/events"):
                token = request.query_params.get("api_key")
            if not token:
                raise HTTPException(
                    status_code=401,
                    detail="缺少 API Key（Authorization: Bearer <key>）",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            principal = authenticate(token, key_map)
            if principal is None:
                raise HTTPException(
                    status_code=401,
                    detail="API Key 无效",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if permission not in PERMISSIONS.get(principal.role, frozenset()):
                raise HTTPException(
                    status_code=403,
                    detail=f"角色 {principal.role} 无权执行 {permission}",
                )
            return principal

        return dependency

    def _spawn(coro) -> None:
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    app = FastAPI(
        title="多 Agent 智能协作平台（MVP 入口）",
        version="0.2.0",
        description="REST 提交分析任务 + SSE 订阅执行进度（W9：API Key + 输入/输出安全基线）。",
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        """校验错误不回显原始输入。

        FastAPI 默认的 422 会带 `input` 字段——超长/畸形请求里可能含手机号、
        身份证等原文；安全基线要求错误响应也不泄露。
        """
        errors = [
            {k: v for k, v in item.items() if k not in ("input", "ctx", "url")}
            for item in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    async def _execute(record: TaskRecord, *, resume_value: dict[str, Any] | None = None) -> None:
        async def on_event(name: str, payload: dict[str, Any]) -> None:
            safe = (
                redact_structure(payload)
                if app_settings.security_dlp_enabled
                else payload
            )
            record.publish(name, **safe)
            registry.trim(record)

        try:
            record.status = "running"
            record.error = None
            record.publish(
                "task_resumed" if resume_value is not None else "task_started",
                question=record.question,
            )
            pipeline = factory(app_settings, on_event)
            outcome = await pipeline.run(
                record.task_event,
                resume=resume_value is not None,
                resume_value=resume_value,
            )
            result = _outcome_to_result(
                outcome, dlp=app_settings.security_dlp_enabled
            )
            if outcome.status == "awaiting_approval":
                # HITL 暂停：非终态。不发 task_finished，SSE 流保持打开等审批。
                record.status = "awaiting_approval"
                record.approval_requests = (
                    redact_structure(outcome.approval_requests)
                    if app_settings.security_dlp_enabled
                    else outcome.approval_requests
                )
                record.result = result
                record.publish(
                    "approval_required",
                    requests=record.approval_requests,
                )
                return
            record.result = result
            # ok / rejected 原样传递；failed 也保持 failed
            record.status = outcome.status
            record.approval_requests = []
            if outcome.status != "ok":
                record.error = (
                    redact_text(outcome.error or "")[0]
                    if app_settings.security_dlp_enabled
                    else outcome.error
                )
        except Exception as exc:  # 兜底：任务异常不得让服务崩掉
            record.status = "failed"
            error = f"{type(exc).__name__}: {exc}"[:500]
            record.error = (
                redact_text(error)[0]
                if app_settings.security_dlp_enabled
                else error
            )
        finally:
            if record.status in TERMINAL_STATUSES:
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
    async def submit(
        request: SubmitRequest,
        _principal: Principal = Depends(require("tasks:submit")),
    ) -> SubmitResponse:
        # 输入侧安全过滤（W9）：命中注入/敏感词直接 400，不建任务、不调模型。
        verdict = scan_input(
            request.question,
            blocked_terms=app_settings.blocked_terms,
            enabled=app_settings.security_enabled,
        )
        if not verdict.allowed:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INPUT_BLOCKED",
                    "reason": verdict.blocked_reason,
                    "rule": verdict.rule,
                },
            )
        question = verdict.text
        task_id = str(uuid.uuid4())
        record = registry.create(task_id, question)
        # TaskEvent 只创建一次：task_id 同时是 checkpointer 的 thread_id，
        # HITL 恢复/续跑必须能找回来（此前每次 run 都新生成 UUID，恢复无从下手）。
        record.task_event = new_task_event(
            question, settings=app_settings, task_id=task_id
        )
        # 后台执行，不阻塞提交响应——这是 SSE 能实时推送的前提
        _spawn(_execute(record))
        return SubmitResponse(
            task_id=task_id,
            status=record.status,
            poll_url=f"/tasks/{task_id}",
            events_url=f"/tasks/{task_id}/events",
        )

    async def _recover_record(task_id: str) -> TaskRecord | None:
        """进程重启后从 checkpoint 重建任务记录（thread_id == API task_id）。"""
        pipeline = factory(app_settings, None)
        try:
            event = await pipeline.load_saved_task(task_id)
        except Exception:  # noqa: BLE001 — 恢复失败按 404 处理
            return None
        if event is None:
            return None
        record = registry.create(task_id, event.payload.text)
        record.task_event = event
        # 能走到恢复流程的前提是任务停在审批门；无权停留的任务会在 resume 时自然收尾
        record.status = "awaiting_approval"
        record.publish("task_recovered", task_id=task_id)
        return record

    @app.post("/tasks/{task_id}/approval", status_code=202)
    async def submit_approval(
        task_id: str,
        body: ApprovalBody,
        principal: Principal = Depends(require("tasks:approve")),
    ) -> dict[str, Any]:
        """HITL 后端阻塞点：批准 / 驳回 / 改写，然后从暂停处恢复执行。"""
        record = registry.get(task_id)
        if record is None:
            record = await _recover_record(task_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"任务不存在或无法从 checkpoint 恢复: {task_id}"
            )
        if record.status != "awaiting_approval":
            raise HTTPException(
                status_code=409, detail=f"任务当前状态不可审批: {record.status}"
            )
        if body.decision == "rewritten" and not body.outputs:
            raise HTTPException(
                status_code=422, detail="rewritten 决策必须提供 outputs（替换后的节点产物）"
            )
        record.status = "running"
        # 审计：启用鉴权时审批人取 Key 指纹，调用方无法自称 approver
        approver = principal.key_id if auth_enabled else body.approver
        record.publish(
            "approval_received",
            decision=body.decision,
            comment=redact_text(body.comment)[0] if app_settings.security_dlp_enabled else body.comment,
            approver=approver,
        )
        resume_value = {
            "decision": body.decision,
            "comment": body.comment,
            "outputs": body.outputs,
        }
        _spawn(_execute(record, resume_value=resume_value))
        return {"task_id": task_id, "status": "running"}

    @app.get("/tasks")
    async def list_tasks(
        limit: int = 20,
        _principal: Principal = Depends(require("tasks:read")),
    ) -> dict[str, Any]:
        body = {"tasks": registry.list_recent(limit=limit)}
        return redact_structure(body) if app_settings.security_dlp_enabled else body

    @app.get("/tasks/{task_id}")
    async def get_task(
        task_id: str,
        _principal: Principal = Depends(require("tasks:read")),
    ) -> dict[str, Any]:
        record = registry.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        body = record.summary()
        return redact_structure(body) if app_settings.security_dlp_enabled else body

    @app.get("/tasks/{task_id}/events")
    async def stream_events(
        task_id: str,
        _principal: Principal = Depends(require("tasks:read")),
    ) -> StreamingResponse:
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
                    yield _sse(item, dlp=app_settings.security_dlp_enabled)
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


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _sse(payload: dict[str, Any], *, dlp: bool = True) -> str:
    """按 SSE 格式编码一条事件。"""
    body = redact_structure(payload) if dlp else payload
    name = body.get("event", "message")
    return f"event: {name}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
