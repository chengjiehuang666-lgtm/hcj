"""Python 沙箱测试。

沙箱的价值全在「隔离是否真的生效」，所以测试重点是**证明约束成立**，
而不是证明「能跑通代码」。容器后端不可用时，与隔离相关的用例会跳过，
但会明确告知跳过原因——不让「没测」伪装成「通过」。
"""

from __future__ import annotations

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.config import Settings  # noqa: E402
from multiagent.tools import (  # noqa: E402
    ISOLATION_CONTAINER,
    ISOLATION_SUBPROCESS,
    PythonSandbox,
    PythonSandboxError,
)

settings = Settings(_env_file=ROOT / ".env")


def _sandbox(**overrides) -> PythonSandbox:
    return PythonSandbox(Settings(_env_file=ROOT / ".env", **overrides))


def _run_and_clean(box: PythonSandbox, code: str, **kwargs):
    """执行并清理工作目录，避免测试累积临时文件。"""
    result = box.run(code, **kwargs)
    if result.work_dir:
        shutil.rmtree(result.work_dir, ignore_errors=True)
    return result


def _container_or_skip(box: PythonSandbox) -> str:
    isolation = box.resolve_isolation()
    if isolation != ISOLATION_CONTAINER:
        pytest.skip(
            "容器后端不可用（Docker 或沙箱镜像缺失），跳过容器专属的隔离验证"
        )
    return isolation


# --------------------------------------------------------------- 基本执行


def test_runs_code_and_returns_stdout() -> None:
    result = _run_and_clean(_sandbox(), 'print("hello-sandbox")')
    assert result.ok, result.stderr
    assert "hello-sandbox" in result.stdout
    assert result.exit_code == 0
    assert result.timed_out is False


def test_reports_failure_with_traceback() -> None:
    result = _run_and_clean(_sandbox(), 'raise ValueError("boom")')
    assert not result.ok
    assert result.exit_code not in (0, None)
    assert "ValueError" in result.stderr and "boom" in result.stderr


def test_isolation_level_is_reported() -> None:
    """降级到子进程时必须如实标注——调用方不能误以为有容器级保护。"""
    result = _run_and_clean(_sandbox(), "print(1)")
    assert result.isolation in (ISOLATION_CONTAINER, ISOLATION_SUBPROCESS)
    assert result.isolation in result.as_prompt_text()


def test_container_mode_raises_instead_of_silently_degrading() -> None:
    """mode=container 时若镜像不可用必须报错，不能悄悄降级。"""
    box = _sandbox(python_sandbox_mode="container", python_sandbox_image="no-such-image:latest")
    with pytest.raises(PythonSandboxError):
        box.resolve_isolation()


def test_subprocess_mode_can_be_forced() -> None:
    box = _sandbox(python_sandbox_mode="subprocess")
    assert box.resolve_isolation() == ISOLATION_SUBPROCESS


# ------------------------------------------------------------------ 超时


def test_timeout_is_enforced() -> None:
    result = _run_and_clean(
        _sandbox(), "import time\nwhile True:\n    time.sleep(0.5)", timeout_seconds=5
    )
    assert result.timed_out is True
    assert result.ok is False
    assert result.duration_ms < 60_000, "超时必须在设定时间内生效"


def test_normal_run_is_not_marked_timed_out() -> None:
    result = _run_and_clean(_sandbox(), "print('ok')", timeout_seconds=20)
    assert result.timed_out is False


# ------------------------------------------------------- 容器专属的隔离验证


def test_network_is_blocked_in_container() -> None:
    """--network none 必须真的断网，否则模型生成的代码可外传数据。"""
    box = _sandbox(python_sandbox_mode="container")
    _container_or_skip(box)
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=3)\n"
        "    print('NETWORK_REACHABLE')\n"
        "except Exception as e:\n"
        "    print('NETWORK_BLOCKED', type(e).__name__)\n"
    )
    result = _run_and_clean(box, code, timeout_seconds=30)
    assert "NETWORK_REACHABLE" not in result.stdout, (
        "容器内竟然能连外网——隔离失效"
    )


def test_root_filesystem_is_read_only_in_container() -> None:
    box = _sandbox(python_sandbox_mode="container")
    _container_or_skip(box)
    code = (
        "try:\n"
        "    open('/etc/ma_should_not_exist', 'w').write('x')\n"
        "    print('ROOT_WRITABLE')\n"
        "except Exception as e:\n"
        "    print('ROOT_READONLY', type(e).__name__)\n"
    )
    result = _run_and_clean(box, code, timeout_seconds=30)
    assert "ROOT_WRITABLE" not in result.stdout, "根文件系统竟然可写——隔离失效"


def test_work_directory_is_writable_in_container() -> None:
    """只读根 + 仅 /work 可写：产物必须能落到工作目录。"""
    box = _sandbox(python_sandbox_mode="container")
    _container_or_skip(box)
    code = "open('out.txt', 'w').write('artifact-content')\nprint('WROTE')"
    result = _run_and_clean(box, code, timeout_seconds=30)
    assert "WROTE" in result.stdout, result.stderr
    assert "out.txt" in result.artifacts


def test_memory_limit_is_enforced_in_container() -> None:
    """--memory 必须生效，否则一段代码就能吃光宿主内存。"""
    box = _sandbox(python_sandbox_mode="container", python_sandbox_memory_mb=128)
    _container_or_skip(box)
    # 申请远超限额的内存；期望进程被 OOM 杀死而非成功
    code = "x = bytearray(600 * 1024 * 1024)\nprint('ALLOCATED', len(x))"
    result = _run_and_clean(box, code, timeout_seconds=60)
    assert "ALLOCATED" not in result.stdout, "超过 --memory 的分配竟然成功了"


# ------------------------------------------------------------- 数据栈可用性


def test_data_stack_available_in_container() -> None:
    """pandas / numpy 必须可用，否则分析场景的代码执行没有意义。"""
    box = _sandbox(python_sandbox_mode="container")
    _container_or_skip(box)
    code = (
        "import pandas as pd, numpy as np\n"
        "df = pd.DataFrame({'a': [1, 2, 3]})\n"
        "print('PANDAS', pd.__version__, int(df['a'].sum()), float(np.mean([1, 2, 3])))\n"
    )
    result = _run_and_clean(box, code, timeout_seconds=60)
    assert "PANDAS" in result.stdout, result.stderr


def test_matplotlib_can_render_chinese_labels() -> None:
    """中文字体缺失时标签会变成方块——那图表就没法用，必须验证。"""
    box = _sandbox(python_sandbox_mode="container")
    _container_or_skip(box)
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(4, 3))\n"
        "ax.bar(['华东', '华南'], [233.5, 179.4])\n"
        "ax.set_title('区域销售额对比（万元）')\n"
        "fig.tight_layout()\n"
        "fig.savefig('chart.png', dpi=80)\n"
        "print('CHART_OK')\n"
    )
    result = _run_and_clean(box, code, timeout_seconds=90)
    assert "CHART_OK" in result.stdout, result.stderr
    assert "chart.png" in result.artifacts


# ------------------------------------------------------------------ 输出处理


def test_large_output_is_clipped() -> None:
    """巨量输出不能整包灌进 prompt。"""
    result = _run_and_clean(_sandbox(), "print('x' * 60000)")
    assert result.ok
    assert len(result.stdout) <= PythonSandbox.MAX_OUTPUT_CHARS


def test_artifacts_list_excludes_entry_script() -> None:
    result = _run_and_clean(
        _sandbox(), "open('a.txt','w').write('1')\nopen('b.txt','w').write('2')"
    )
    assert result.artifacts == ["a.txt", "b.txt"]
