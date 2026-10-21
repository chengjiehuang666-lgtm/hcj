#!/usr/bin/env bash
# W14 自签 TLS 证书（仅本地/内网试用；生产请换正式证书）。
# 用法：bash scripts/gen_cert.sh [--force]
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="deploy/tls"
mkdir -p "$OUT"

if [ -f "$OUT/cert.pem" ] && [ -f "$OUT/key.pem" ] && [ "${1:-}" != "--force" ]; then
  echo "TLS 证书已存在：$OUT/cert.pem（--force 可重签）"
  exit 0
fi

# Git Bash 会把 -subj 的 /CN=... 当路径转换，MSYS_NO_PATHCONV=1 关掉
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
  -keyout "$OUT/key.pem" -out "$OUT/cert.pem" \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
  >/dev/null 2>&1
chmod 600 "$OUT/key.pem"
echo "已生成自签证书：$OUT/cert.pem（CN=localhost，SAN=localhost/127.0.0.1）"
echo "⚠ 自签证书浏览器会告警；生产请换成可信证书（Caddy/nginx + Let's Encrypt）。"
