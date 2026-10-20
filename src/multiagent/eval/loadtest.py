"""基础压测（W8）：单机并发瓶颈定位。

三层负载，分别回答不同问题：

| 模式 | 负载 | 回答的问题 |
| :--- | :--- | :--- |
| `model` | 直接打本地模型端点（固定 prompt，N 并发） | 模型端点本身的并发膝点（最可能的瓶颈，且主备同端点） |
| `pipeline` | 进程内真 PostgreSQL + 假 Agent（固定延迟） | 连接池 / checkpoint / 编排开销的天花板（不烧 token） |
| `api` | HTTP 提交到运行中的服务、轮询到终态（真实模型） | 全链路端到端劣化与阶段归因（需先启动服务） |

用法：
    python -m multiagent.eval.loadtest --mode model --levels 1,2,4,8
    python -m multiagent.eval.loadtest --mode pipeline --levels 1,4,8,16
    python -m multiagent.eval.loadtest --mode api --levels 1,2,4 --tasks-per-level 3

口径（与评测一致）：时延是**单任务端到端**（客户端 perf_counter）；同一档位内并发提交，
全部到达终态后进入下一档。W8 只定位瓶颈，不设 QPS 目标（D1 基线归 W11）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TZ_CN = timezone(timedelta(hours=8))


@dataclass
class Sample:
    ok: bool
    latency_ms: int
    error: str | None = None
    server_duration_ms: int | None = None
    tokens: int | None = None


@dataclass
class LevelResult:
    concurrency: int
    mode: str
    samples: list[Sample] = field(default_factory=list)
    wall_s: float = 0.0
    peak_connections: int | None = None

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.samples if s.ok)

    @property
    def effective_concurrency(self) -> int:
        """实际可能达到的并发 = min(信号量档位, 任务数)。

        `--tasks-per-level` 小于档位时，并发被任务数封顶——首轮压测
        曾因此把「N=8 只有 4 个任务」误读成「端点可扩到 8」。
        """
        return min(self.concurrency, len(self.samples))

    def percentile(self, p: float) -> float:
        """最近秩法（nearest-rank），**只统计成功样本**。

        失败样本的「时延」是快速失败的耗时（常为 0.0s），混进分位数会把
        P50 变成 0——读起来像「很快」，实际是「一半请求被拒」。失败率由
        成功率列单独呈现。

        不用 `round((p/100)*(N-1))`：偶数样本时会把 P50 推到第 3 小
        （4 个样本 [1,2,3,4] 的 P50 应是 2，不是 3）。
        """
        values = sorted(s.latency_ms for s in self.samples if s.ok)
        if not values:
            return 0.0
        import math

        rank = max(1, math.ceil((p / 100) * len(values)))
        return values[min(rank, len(values)) - 1] / 1000.0

    @property
    def throughput(self) -> float:
        return len(self.samples) / self.wall_s if self.wall_s else 0.0


def _conn_count(settings) -> int:
    """当前数据库连接数（瓶颈证据）。"""
    import psycopg

    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                )
                return int(cur.fetchone()[0])
    except Exception:  # noqa: BLE001
        return -1


# --------------------------------------------------------------------- model


async def _run_model_level(settings, concurrency: int, per_task: int, timeout: float) -> LevelResult:
    """直接打本地模型端点：固定 prompt、短输出，只测端点并发。"""
    import httpx

    provider = settings.primary
    headers = {"Authorization": f"Bearer {provider.api_key.get_secret_value()}"}
    payload = {
        "model": provider.model,
        "messages": [{"role": "user", "content": "用一句话说明什么是销售额。"}],
        "max_tokens": 256,
    }
    result = LevelResult(concurrency=concurrency, mode="model")
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0),
        limits=httpx.Limits(max_connections=concurrency + 10),
    ) as client:

        async def one() -> None:
            async with sem:
                started = time.perf_counter()
                try:
                    response = await client.post(
                        f"{provider.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    ok = response.status_code == 200
                    error = None if ok else f"HTTP {response.status_code}: {response.text[:200]}"
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    error = f"{type(exc).__name__}: {exc}"[:200]
                result.samples.append(
                    Sample(ok=ok, latency_ms=int((time.perf_counter() - started) * 1000), error=error)
                )

        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(per_task)))
        result.wall_s = time.perf_counter() - started
    return result


# ------------------------------------------------------------------ pipeline


class _FakeAgent:
    """固定延迟的假 Agent：只验证编排/连接/checkpoint 的开销，不烧 token。"""

    def __init__(self, role: str, *, delay: float, version: str = "loadtest") -> None:
        self.role = role
        self.version = version
        self._delay = delay

    async def invoke(self, inv, ctx):
        from ..contracts import AgentResult, SourceRef, SourceType

        await asyncio.sleep(self._delay)
        if self.role == "planner":
            dag = {
                "task_id": inv.task_id,
                "goal": "loadtest",
                "created_at": datetime.now(TZ_CN).isoformat(timespec="seconds"),
                "nodes": [
                    {"node_id": "retriever_1", "name": "r1", "agent_role": "retriever",
                     "depends_on": [], "inputs": {"mode": "metrics"}, "outputs": {},
                     "acceptance": ["x"]},
                    {"node_id": "retriever_2", "name": "r2", "agent_role": "retriever",
                     "depends_on": [], "inputs": {"mode": "knowledge"}, "outputs": {},
                     "acceptance": ["x"]},
                    {"node_id": "analyst_1", "name": "a1", "agent_role": "analyst",
                     "depends_on": ["retriever_1", "retriever_2"], "inputs": {},
                     "outputs": {}, "acceptance": ["x"]},
                    {"node_id": "reviewer_1", "name": "v1", "agent_role": "reviewer",
                     "depends_on": ["retriever_1", "retriever_2", "analyst_1"],
                     "inputs": {}, "outputs": {}, "acceptance": ["x"]},
                ],
            }
            return AgentResult(
                task_id=inv.task_id, node_id=inv.node_id, attempt=inv.attempt,
                status="ok", tokens_used={"input": 1, "output": 1, "total": 2},
                duration_ms=int(self._delay * 1000),
                outputs={"dag": dag, "node_count": 4},
                sources=[SourceRef(type=SourceType.model_inference, ref="loadtest")],
            )
        return AgentResult(
            task_id=inv.task_id, node_id=inv.node_id, attempt=inv.attempt,
            status="ok", tokens_used={"input": 1, "output": 1, "total": 2},
            duration_ms=int(self._delay * 1000),
            outputs={"summary": f"{inv.node_id} ok", "knowledge": [], "knowledge_count": 0},
            sources=[SourceRef(type=SourceType.model_inference, ref="loadtest")],
            blackboard_writes=[f"{inv.node_id}.out"],
        )


async def _run_pipeline_level(
    settings, concurrency: int, per_task: int, delay: float, timeout: float
) -> LevelResult:
    """进程内：真 PostgreSQL + 假 Agent（需要 monkeypatch 注册表）。"""
    from ..graph import pipeline as pipeline_module
    from ..graph.pipeline import AnalysisPipeline, new_task_event

    registry = {
        role: _FakeAgent(role, delay=delay)
        for role in ("planner", "retriever", "analyst", "reviewer")
    }
    original = pipeline_module.build_agent_registry
    pipeline_module.build_agent_registry = lambda: registry
    cfg = settings.model_copy(update={"memory_sink_enabled": False})
    pipeline = AnalysisPipeline(cfg)

    result = LevelResult(concurrency=concurrency, mode="pipeline")
    sem = asyncio.Semaphore(concurrency)
    peak = 0
    stop = asyncio.Event()

    async def sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, _conn_count(cfg))
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

    async def one() -> None:
        async with sem:
            event = new_task_event("loadtest 销售额", settings=cfg)
            started = time.perf_counter()
            try:
                outcome = await asyncio.wait_for(pipeline.run(event), timeout=timeout)
                ok = outcome.status == "ok"
                error = outcome.error if not ok else None
            except Exception as exc:  # noqa: BLE001
                ok = False
                error = f"{type(exc).__name__}: {exc}"[:200]
            result.samples.append(
                Sample(
                    ok=ok,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error=error,
                )
            )

    sampler_task = asyncio.create_task(sampler())
    started = time.perf_counter()
    try:
        await asyncio.gather(*(one() for _ in range(per_task)))
    finally:
        result.wall_s = time.perf_counter() - started
        stop.set()
        await sampler_task
        pipeline_module.build_agent_registry = original
    result.peak_connections = peak
    return result


# ----------------------------------------------------------------------- api


async def _run_api_level(
    base_url: str,
    questions: list[str],
    concurrency: int,
    per_task: int,
    timeout: float,
    api_key: str | None = None,
) -> LevelResult:
    """HTTP 提交 + 轮询到终态（需要先启动 python -m multiagent.api）。"""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    result = LevelResult(concurrency=concurrency, mode="api")
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:

        async def one(index: int) -> None:
            async with sem:
                question = questions[index % len(questions)]
                started = time.perf_counter()
                try:
                    response = await client.post(
                        f"{base_url}/tasks", json={"question": question}, headers=headers
                    )
                    response.raise_for_status()
                    task_id = response.json()["task_id"]
                    while True:
                        body = (
                            await client.get(
                                f"{base_url}/tasks/{task_id}", headers=headers
                            )
                        ).json()
                        if body["status"] in ("ok", "failed", "rejected"):
                            break
                        if time.perf_counter() - started > timeout:
                            raise TimeoutError("轮询超时")
                        await asyncio.sleep(0.5)
                    detail = body.get("result") or {}
                    result.samples.append(
                        Sample(
                            ok=body["status"] == "ok",
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            error=body.get("error"),
                            server_duration_ms=detail.get("duration_ms"),
                            tokens=detail.get("total_tokens"),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    result.samples.append(
                        Sample(
                            ok=False,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            error=f"{type(exc).__name__}: {exc}"[:200],
                        )
                    )

        started = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(per_task)))
        result.wall_s = time.perf_counter() - started
    return result


# -------------------------------------------------------------------- report


def render_report(results: list[LevelResult], *, mode: str, extra: dict[str, Any]) -> str:
    lines = [
        f"# 压测报告（{mode}，{datetime.now(TZ_CN).isoformat(timespec='seconds')}）",
        "",
        "## 环境",
        "",
        f"- 主机：{platform.processor() or platform.machine()}；逻辑核 {os.cpu_count()}",
        f"- 部署：单进程单 worker（进程内任务表/SSE，多 worker 不可用）",
        f"- 模型：`{extra.get('model')}` @ `{extra.get('base_url')}`",
        f"- 说明：{extra.get('note', '')}",
        "",
        "## 结果",
        "",
        "> 时延分位数只统计**成功样本**；失败样本单独列在下一节（快速失败的 0.0s 不参与分位数）。",
        "",
        "| 并发档位 | 实际并发 | 任务数 | 成功 | 成功率 | 墙钟 | 吞吐(任务/s) | P50 | P95 | P99 | 最大 | 峰值连接 |",
        "| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for level in results:
        latency = sorted(s.latency_ms for s in level.samples)
        max_ms = latency[-1] if latency else 0
        lines.append(
            f"| {level.concurrency} | {level.effective_concurrency} | {len(level.samples)} | "
            f"{level.ok_count} | "
            f"{level.ok_count / len(level.samples) if level.samples else 0:.1%} | "
            f"{level.wall_s:.1f}s | {level.throughput:.2f} | "
            f"{level.percentile(50):.1f}s | {level.percentile(95):.1f}s | "
            f"{level.percentile(99):.1f}s | {max_ms / 1000:.1f}s | "
            f"{level.peak_connections if level.peak_connections is not None else '-'} |"
        )
    lines += ["", "## 失败样本", ""]
    errors = [
        (level.concurrency, sample.error)
        for level in results
        for sample in level.samples
        if not sample.ok
    ]
    if not errors:
        lines.append("（无）")
    else:
        for concurrency, error in errors[:20]:
            lines.append(f"- N={concurrency}: {error}")
    lines += ["", "## 观察", ""]
    for note in extra.get("observations", []):
        lines.append(f"- {note}")
    return "\n".join(lines)


async def main_async(args) -> int:
    from ..config import Settings
    from ..observability import configure_logging

    settings = Settings()
    configure_logging(settings)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    results: list[LevelResult] = []
    extra: dict[str, Any] = {
        "model": settings.primary.model,
        "base_url": settings.primary.base_url,
    }

    if args.mode == "model":
        extra["note"] = "直打模型端点（固定 prompt、max_tokens=256），不经过应用"
    elif args.mode == "pipeline":
        extra["note"] = (
            f"进程内真 PostgreSQL + 假 Agent（每节点 {args.delay}s）；关闭记忆沉淀；"
            "DAG=两路并行检索→分析→评审"
        )
    else:
        extra["note"] = f"HTTP 全链路（服务 {args.base_url}），真实模型，轮询到终态"
        extra["auth"] = "enabled" if settings.api_auth_enabled else "disabled"
        questions = [
            "华东区 2026 年 9 月的销售额是多少？",
            "9 月华东区线上渠道和门店渠道各卖了多少？",
            "分析上月华东区销售额下滑的原因",
        ]
        extra["questions"] = questions

    for level in levels:
        if level > args.tasks_per_level:
            print(
                f"  ⚠ 并发档位 {level} 大于每档任务数 {args.tasks_per_level}："
                f"实际并发只有 {args.tasks_per_level}（测不到 {level} 并发）",
                flush=True,
            )
        print(f"▶ {args.mode} N={level} × {args.tasks_per_level} 任务 …", flush=True)
        if args.mode == "model":
            result = await _run_model_level(
                settings, level, args.tasks_per_level, args.timeout
            )
        elif args.mode == "pipeline":
            result = await _run_pipeline_level(
                settings, level, args.tasks_per_level, args.delay, args.timeout
            )
        else:
            result = await _run_api_level(
                args.base_url,
                extra["questions"],
                level,
                args.tasks_per_level,
                args.timeout,
                api_key=(
                    args.api_key
                    or (settings.default_client_key if settings.api_auth_enabled else None)
                ),
            )
        results.append(result)
        print(
            f"  成功 {result.ok_count}/{len(result.samples)}，"
            f"P50 {result.percentile(50):.1f}s，P95 {result.percentile(95):.1f}s，"
            f"吞吐 {result.throughput:.2f} 任务/s"
            + (f"，峰值连接 {result.peak_connections}" if result.peak_connections is not None else "")
        )

    # 基础观察（自动生成，避免人在报告里臆断）
    observations: list[str] = []
    if results:
        capped = [
            level for level in results
            if level.effective_concurrency < level.concurrency
        ]
        if capped:
            observations.append(
                "以下档位被任务数封顶，不能作为该并发下的结论："
                + ", ".join(
                    f"N={level.concurrency}×{len(level.samples)}任务"
                    for level in capped
                )
            )
        first = results[0]
        last = results[-1]
        if last.percentile(95) > first.percentile(95) * 1.5:
            observations.append(
                f"并发从 {first.concurrency} 升到 {last.concurrency} 时 P95 "
                f"从 {first.percentile(95):.1f}s 恶化到 {last.percentile(95):.1f}s"
            )
        failed = sum(len(level.samples) - level.ok_count for level in results)
        if failed:
            observations.append(f"共 {failed} 个任务失败（见失败样本）")
        else:
            observations.append("各档位全部任务成功")
    extra["observations"] = observations

    markdown = render_report(results, mode=args.mode, extra=extra)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(TZ_CN).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"loadtest-{args.mode}-{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    payload = {
        "mode": args.mode,
        "extra": extra,
        "levels": [
            {
                "concurrency": level.concurrency,
                "effective_concurrency": level.effective_concurrency,
                "tasks": len(level.samples),
                "ok": level.ok_count,
                "wall_s": round(level.wall_s, 3),
                "throughput": round(level.throughput, 3),
                "p50_s": round(level.percentile(50), 3),
                "p95_s": round(level.percentile(95), 3),
                "p99_s": round(level.percentile(99), 3),
                "peak_connections": level.peak_connections,
                "errors": [s.error for s in level.samples if not s.ok][:20],
            }
            for level in results
        ],
    }
    (out_dir / f"loadtest-{args.mode}-{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m multiagent.eval.loadtest")
    parser.add_argument("--mode", choices=["model", "pipeline", "api"], required=True)
    parser.add_argument("--levels", default="1,2,4", help="并发档位，逗号分隔")
    parser.add_argument("--tasks-per-level", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0, help="假 Agent 每节点延迟（秒）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key",
        default=None,
        help="api 模式的 API Key；缺省取 .env 中的第一个 operator/admin/viewer Key",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="eval_results")
    args = parser.parse_args(argv)

    from ..asyncutil import run_async

    return run_async(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
