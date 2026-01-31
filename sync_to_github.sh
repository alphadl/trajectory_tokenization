#!/bin/bash
# 把当前目录（trajectory_tokenization 新代码）推上去，覆盖远程旧内容
set -e
cd "$(dirname "$0")"
git add -A
git status
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "trajectory_tokenization release"
fi
git branch -M main
git push -f origin main
echo "Done. Check https://github.com/alphadl/trajectory_tokenization"
