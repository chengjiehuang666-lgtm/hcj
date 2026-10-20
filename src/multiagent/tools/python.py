"""Python 代码执行沙箱 —— Agent 执行生成代码的唯一入口。

与 SQL 沙箱同样是**多层防御**，但隔离手段不同（代码执行比只读查询危险得多）：

| 层 | container 后端（首选） | subprocess 后端（降级） |
| :--- | :--- | :--- |
| 网络 | `--network none`（无法出网） | **无限制** ⚠️ |
| 文件系统 | `--read-only` 根 + 仅 `/work` 可写 | 仅临时工作目录，根可读写 ⚠️ |
| 内存 | `--memory` 硬限制 | **无限制** ⚠️ |
| CPU | `--cpus` 配额 | **无限制** ⚠️ |
| 超时 | 墙钟超时 + 主动 kill 容器 | 墙钟超时 + kill 进程 |
| 用户 | 非 root（sandbox） | 当前用户权限 ⚠️ |

**降级后端的隔离强度显著更弱**，因此结果里带 `isolation` 字段如实标注，
避免调用方误以为始终有容器级保护。选择逻辑见 `mode` 配置：
`auto`（默认，能用容器就用）/ `container`（强制，不可用则报错）/ `subprocess`。

本期不做 gVisor 级隔离（见《实施路线图》W3 范围收敛）。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..observability import log_event


class PythonSandboxError(Exception):
    """沙箱自身故障（镜像缺失、Docker 不可用等），区别于「用户代码报错」。"""


ISOLATION_CONTAINER = "container"
ISOLATION_SUBPROCESS = "subprocess"

#: 入口脚本文件名。产物 = 工作目录里除它以外的文件。
ENTRY_FILENAME = "main.py"


@dataclass
class PythonResult:
    """一次代码执行的结果。"""

    ok: bool
    stdout: str
    stderr: str
    exit_code: int | None
    duration_ms: int
    isolation: str
    timed_out: bool = False
    artifacts: list[str] = field(default_factory=list)
    work_dir: str | None = None
    #: 被截断标记：输出过长时仅保留前 N 字符，避免把巨量输出灌进 prompt
    stdout_truncated: bool = False

    def as_prompt_text(self, limit: int = 4000) -> str:
        parts = [f"[isolation={self.isolation} exit={self.exit_code} {self.duration_ms}ms]"]
        if self.timed_out:
            parts.append("!! 执行超时被终止")
        if self.stdout:
            parts.append("stdout:\n" + self.stdout[:limit])
        if self.stderr:
            parts.append("stderr:\n" + self.stderr[:limit])
        if self.artifacts:
            parts.append("产物文件: " + ", ".join(self.artifacts))
        return "\n".join(parts)


class PythonSandbox:
    """在隔离环境里执行 Python 代码。"""

    MAX_OUTPUT_CHARS = 20_000

    def __init__(self, settings: Settings) -> None:
        self._image = settings.python_sandbox_image
        self._timeout = settings.python_sandbox_timeout_seconds
        self._memory_mb = settings.python_sandbox_memory_mb
        self._cpus = settings.python_sandbox_cpus
        self._mode = settings.python_sandbox_mode
        self._docker = settings.python_sandbox_docker_cli

    # ------------------------------------------------------------------ 能力探测

    def _docker_ready(self) -> bool:
        if not shutil.which(self._docker):
            return False
        try:
            probe = subprocess.run(
                [self._docker, "image", "inspect", self._image],
                capture_output=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return probe.returncode == 0

    def resolve_isolation(self) -> str:
        """按配置与能力决定用哪个后端；`container` 模式不可用时直接报错，不静默降级。"""
        ready = self._docker_ready()
        if self._mode == "container":
            if not ready:
                raise PythonSandboxError(
                    f"python_sandbox_mode=container，但 Docker 或镜像 {self._image} 不可用。"
                    f"请先构建：docker build -t {self._image} docker/python-sandbox"
                )
            return ISOLATION_CONTAINER
        if self._mode == "subprocess":
            return ISOLATION_SUBPROCESS
        return ISOLATION_CONTAINER if ready else ISOLATION_SUBPROCESS

    # ------------------------------------------------------------------ 执行

    def run(self, code: str, *, timeout_seconds: int | None = None) -> PythonResult:
        timeout = timeout_seconds or self._timeout
        work_dir = Path(tempfile.mkdtemp(prefix="ma-sandbox-"))
        (work_dir / ENTRY_FILENAME).write_text(code, encoding="utf-8")

        isolation = self.resolve_isolation()
        started = time.perf_counter()
        try:
            if isolation == ISOLATION_CONTAINER:
                result = self._run_in_container(work_dir, timeout)
            else:
                result = self._run_as_subprocess(work_dir, timeout)
        finally:
            # 工作目录保留给调用方取产物；由调用方或 GC 负责清理
            pass

        duration_ms = int((time.perf_counter() - started) * 1000)
        artifacts = sorted(
            p.name for p in work_dir.iterdir() if p.name != ENTRY_FILENAME
        )
        log_event(
            "tool_finished",
            level=20 if result["ok"] else 30,
            tool="python",
            isolation=isolation,
            ok=result["ok"],
            exit_code=result["exit_code"],
            timed_out=result["timed_out"],
            duration_ms=duration_ms,
        )
        return PythonResult(
            ok=result["ok"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            exit_code=result["exit_code"],
            duration_ms=duration_ms,
            isolation=isolation,
            timed_out=result["timed_out"],
            artifacts=artifacts,
            work_dir=str(work_dir),
        )

    # ---- 后端：容器 ----
    def _run_in_container(self, work_dir: Path, timeout: int) -> dict[str, Any]:
        name = f"ma-sbx-{uuid.uuid4().hex[:12]}"
        cmd = [
            self._docker, "run", "--rm",
            "--name", name,
            "--network", "none",
            "--memory", f"{self._memory_mb}m",
            "--memory-swap", f"{self._memory_mb}m",
            "--cpus", str(self._cpus),
            "--pids-limit", "256",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=128m,mode=1777",
            "-v", f"{work_dir}:/work:rw",
            "-w", "/work",
            self._image,
            # 用 python3 而非 python：Alpine 的 python3 包只保证 python3 可执行名
            "python3", "-I", ENTRY_FILENAME,
        ]
        return self._execute(cmd, timeout, killer=[self._docker, "kill", name])

    # ---- 后端：本机子进程（降级） ----
    def _run_as_subprocess(self, work_dir: Path, timeout: int) -> dict[str, Any]:
        import sys

        cmd = [sys.executable, "-I", ENTRY_FILENAME]
        return self._execute(cmd, timeout, cwd=str(work_dir))

    # ---- 公共执行与超时处理 ----
    def _execute(
        self,
        cmd: list[str],
        timeout: int,
        *,
        cwd: str | None = None,
        killer: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            # 容器后端必须显式 kill：只终止 docker run 客户端进程，容器仍会继续跑
            if killer:
                try:
                    subprocess.run(killer, capture_output=True, timeout=20)
                except (OSError, subprocess.SubprocessError):
                    pass
            return {
                "ok": False,
                "stdout": self._clip(exc.stdout),
                "stderr": self._clip(exc.stderr),
                "exit_code": None,
                "timed_out": True,
            }
        except FileNotFoundError as exc:
            raise PythonSandboxError(f"执行器不存在: {cmd[0]}") from exc

        return {
            "ok": proc.returncode == 0,
            "stdout": self._clip(proc.stdout),
            "stderr": self._clip(proc.stderr),
            "exit_code": proc.returncode,
            "timed_out": False,
        }

    @classmethod
    def _clip(cls, text: str | None) -> str:
        if not text:
            return ""
        return text[: cls.MAX_OUTPUT_CHARS]
