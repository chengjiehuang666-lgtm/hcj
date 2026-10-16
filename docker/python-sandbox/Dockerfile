# Python 代码执行沙箱镜像。
#
# 为什么依赖要预先烘进镜像：沙箱运行时是 `--network none`，
# 不可能在容器内联网装包。
#
# ⚠️ 基础镜像说明：正常情况下应该用 `python:3.13-alpine`，但**当前网络环境无法
# 访问 Docker Hub**（registry-1.docker.io 与 auth.docker.io 均不可达，宿主机层面
# 就超时），只能从本地已有镜像出发。`redis:7-alpine` 在此**仅作为 Alpine 用户态
# 的提供者，与 Redis 无关**；它只是恰好已经在本地且足够小。
# 一旦 Docker Hub 可达，把 FROM 换成 `python:3.13-alpine`、并删掉下面 python3 的
# apk 安装即可（其余不变）。
#
# 隔离由运行时参数承担（见 src/multiagent/tools/python.py）：
#   --network none、--read-only、--memory/--cpus 配额、非 root。
# 本期不做 gVisor 级隔离（见《实施路线图》W3 范围收敛）。
#
# 构建：docker build -t multiagent-python-sandbox:latest docker/python-sandbox

FROM redis:7-alpine

# 数据分析与绘图依赖 + 中文字体。
# 缺字体时 matplotlib 画出的中文标签会变成方块（tofu），而本项目的图表场景
# 必然出现中文标签，故必须装。全部用发行版包，不依赖 PyPI。
RUN apk add --no-cache \
        python3 \
        py3-numpy \
        py3-pandas \
        py3-matplotlib \
        py3-pillow \
        fontconfig \
        font-wqy-zenhei \
 && fc-cache -f

RUN adduser -D -s /sbin/nologin sandbox

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /work
USER sandbox
