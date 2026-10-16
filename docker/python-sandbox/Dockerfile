# Python 代码执行沙箱镜像。
#
# 为什么依赖要预先烘进镜像：沙箱运行时是 `--network none`，
# 不可能在容器内联网装包。
#
# ⚠️ 基础镜像选择说明：曾一度因 Docker Hub 不可达而改用本地已有的 redis:7-alpine
# 作为 Alpine 用户态提供者（那次连接失败是暂时的）。现已恢复可达，改回标准基础镜像。
# 注意：不能混用 apk 的 py3-* 包——它们面向 Alpine 发行版的 python，与
# python:3.13-alpine 自带的 python3.13 不同 ABI；依赖一律用 pip 装。
#
# 隔离由运行时参数承担（见 src/multiagent/tools/python.py）：
#   --network none、--read-only、--memory/--cpus 配额、非 root。
# 本期不做 gVisor 级隔离（见《实施路线图》W3 范围收敛）。
#
# 构建：docker build -t multiagent-python-sandbox:latest docker/python-sandbox

FROM python:3.13-alpine

# 数据分析与绘图依赖。暂不锁小版本（锁错版本会导致构建失败），
# 待沙箱行为稳定后应改为精确 pin 以保证可复现。
RUN pip install --no-cache-dir numpy pandas matplotlib

# 中文字体：缺字体时 matplotlib 画出的中文标签会变成方块（tofu），
# 而本项目的图表场景必然出现中文标签，故必须装。
RUN apk add --no-cache fontconfig font-wqy-zenhei && fc-cache -f

RUN adduser -D -s /sbin/nologin sandbox

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /work
USER sandbox
