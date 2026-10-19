"""压测工具的口径测试（W8）。

压测报告最容易出的错不是数字算错，而是**口径说错**：首轮把
「N=8 档位只放了 4 个任务」当成「端点扛住了 8 并发」。这里钉住
「实际并发 = min(档位, 任务数)」这个口径与报告呈现。
"""

from __future__ import annotations

from multiagent.eval.loadtest import LevelResult, Sample, render_report


def _level(concurrency: int, latencies: list[int], ok: bool = True) -> LevelResult:
    return LevelResult(
        concurrency=concurrency,
        mode="model",
        samples=[Sample(ok=ok, latency_ms=value) for value in latencies],
        wall_s=1.0,
    )


def test_effective_concurrency_capped_by_task_count() -> None:
    assert _level(8, [1000] * 4).effective_concurrency == 4
    assert _level(4, [1000] * 8).effective_concurrency == 4


def test_percentile_uses_nearest_rank() -> None:
    level = _level(1, [1000, 2000, 3000, 4000])
    assert level.percentile(50) == 2.0
    assert level.percentile(95) == 4.0


def test_percentile_excludes_failed_samples() -> None:
    """失败样本的 0.0s 不能把 P50 拉成 0——那是「一半被拒」，不是「很快」。"""
    level = LevelResult(
        concurrency=7,
        mode="model",
        samples=[
            *(Sample(ok=True, latency_ms=2000) for _ in range(6)),
            *(Sample(ok=False, latency_ms=0, error="HTTP 503") for _ in range(6)),
        ],
        wall_s=2.0,
    )
    assert level.percentile(50) == 2.0
    assert level.percentile(95) == 2.0
    assert level.ok_count == 6


def test_render_report_shows_effective_concurrency_and_marks_capped_level() -> None:
    results = [
        _level(4, [1000, 1200, 1500, 1800]),
        _level(8, [1000, 1200, 1500, 1800]),  # 只有 4 个任务 → 实际并发 4
    ]
    markdown = render_report(
        results,
        mode="model",
        extra={
            "model": "m",
            "base_url": "u",
            "note": "n",
            "observations": [
                "以下档位被任务数封顶，不能作为该并发下的结论：N=8×4任务"
            ],
        },
    )
    assert "| 并发档位 | 实际并发 |" in markdown
    assert "| 8 | 4 |" in markdown
    assert "被任务数封顶" in markdown
