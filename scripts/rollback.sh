#!/usr/bin/env bash
# W12 回滚演练：按 Git 标签构建旧代码镜像并切回，然后冒烟。
# 用法：bash scripts/rollback.sh <git-tag>
# 例：  bash scripts/rollback.sh v0.14-w11-calibration
#
# 设计：用 git worktree 取旧标签源码，补齐 W12 的部署资产（旧标签可能没有），
# 构建并替换 app 容器；不切换当前工作区，失败也不会污染 main。
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "用法: bash scripts/rollback.sh <git-tag>" >&2
  echo "可用标签: $(git tag --sort=-v:refname | head -5 | tr '\n' ' ')" >&2
  exit 1
fi
git rev-parse "$TARGET" >/dev/null

WORK="$(mktemp -d)"
cleanup() { git worktree remove --force "$WORK" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "▶ 检出 $TARGET 到临时 worktree"
git worktree add --detach "$WORK" "$TARGET" >/dev/null

# 旧标签可能没有 W12 的部署资产：从当前版本补齐（只补文件，不改旧代码）
for f in docker-compose.app.yml docker/app/Dockerfile .dockerignore; do
  if [ ! -f "$WORK/$f" ]; then
    mkdir -p "$(dirname "$WORK/$f")"
    cp "$f" "$WORK/$f"
    echo "   补齐部署资产：$f"
  fi
done

IMAGE="multiagent-app:rollback-$(printf '%s' "$TARGET" | tr '/:' '--')"
echo "▶ 构建回滚镜像 $IMAGE"
docker build -f "$WORK/docker/app/Dockerfile" -t "$IMAGE" "$WORK"

echo "▶ 切换 app 容器到 $IMAGE"
APP_IMAGE="$IMAGE" docker compose \
  -f docker-compose.yml -f docker-compose.app.yml \
  up -d --no-build --force-recreate app

for _ in $(seq 1 90); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null; then
  echo "✗ 回滚后服务未就绪；查看 app 日志" >&2
  exit 1
fi
echo "PASS health"

PY="python"
if [ -x .venv/Scripts/python.exe ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
fi
"$PY" scripts/smoke.py

echo "✅ 已回滚到 $TARGET（镜像 $IMAGE）"
echo "   恢复当前版本：bash scripts/deploy.sh"
