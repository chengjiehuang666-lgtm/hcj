"""W12 冒烟测试：健康检查 + 一条真实任务跑到 ok。

用法（宿主机）：
    python scripts/smoke.py
    SMOKE_BASE_URL=http://127.0.0.1:8000 python scripts/smoke.py

鉴权开启时自动从 Settings 取第一个 operator/admin Key；不打印 Key。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from multiagent.config import Settings  # noqa: E402

BASE = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
QUESTION = os.getenv("SMOKE_QUESTION", "华东区 2026 年 9 月的销售额是多少？")
TIMEOUT_S = int(os.getenv("SMOKE_TIMEOUT_S", "300"))


def call(method: str, path: str, payload: dict | None = None, key: str | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    settings = Settings()
    key = settings.default_client_key if settings.api_auth_enabled else None

    try:
        status, health = call("GET", "/health")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL health: {type(exc).__name__}: {exc}")
        return 1
    if status != 200 or health.get("dependencies", {}).get("postgres", {}).get("status") != "ok":
        print(f"FAIL health: HTTP {status} {health}")
        return 1
    print("PASS health")

    try:
        status, body = call("POST", "/tasks", {"question": QUESTION}, key)
    except urllib.error.HTTPError as exc:
        print(f"FAIL submit: HTTP {exc.code} {exc.read()[:300]!r}")
        return 1
    if status != 202:
        print(f"FAIL submit: HTTP {status} {body}")
        return 1
    task_id = body["task_id"]
    print(f"PASS submit task_id={task_id}")

    deadline = time.time() + TIMEOUT_S
    detail: dict = {}
    while time.time() < deadline:
        try:
            _, detail = call("GET", f"/tasks/{task_id}", key=key)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL poll: {type(exc).__name__}: {exc}")
            return 1
        if detail.get("status") in ("ok", "failed", "rejected"):
            break
        time.sleep(2)
    else:
        print(f"FAIL timeout after {TIMEOUT_S}s")
        return 1

    if detail.get("status") != "ok":
        print(f"FAIL task status={detail.get('status')} error={detail.get('error')}")
        return 1
    result = json.dumps(detail.get("result") or {}, ensure_ascii=False)
    if "2335000" not in result:
        print("FAIL result does not contain the expected 2,335,000 figure")
        return 1
    print(f"PASS task ok (tokens={detail.get('result', {}).get('total_tokens')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
