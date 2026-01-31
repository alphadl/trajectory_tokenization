#!/bin/bash
# 在本机终端执行以推送（需先完成 GitHub 认证）
set -e
cd "$(dirname "$0")"
# 避免大 push（>1MB）时 HTTP 400 / RPC failed：提高 post 缓冲区
git config http.postBuffer 524288000
git branch -M main
git push -u origin main
