#!/bin/bash
# One-click: ReAct vs ReAct+tokenization comparison (HotpotQA + FEVER).
# Requires: OPENAI_API_KEY, pip install openai requests beautifulsoup4
set -e
cd "$(dirname "$0")"
export PYTHONPATH="."
python run_comparison.py --max_examples 5
