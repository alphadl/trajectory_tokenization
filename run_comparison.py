#!/usr/bin/env python3
"""
Comparison: ReAct (baseline) vs ReAct + trajectory tokenization.
Runs HotpotQA and FEVER each with and without tokenization, prints EM table.
Usage:
  export OPENAI_API_KEY=your_key
  python run_comparison.py [--max_examples 5]
"""
import argparse
import os
import sys
from types import SimpleNamespace

_react_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_react_root)
sys.path.insert(0, _react_root)

import run_hotpotqa
import run_fever


def main():
    parser = argparse.ArgumentParser(description="ReAct vs ReAct+tokenization comparison")
    parser.add_argument("--max_examples", type=int, default=5, help="Examples per task (use 500 for paper setting)")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run comparison.")
        sys.exit(1)

    base = SimpleNamespace(
        split="dev",
        max_examples=args.max_examples,
        tokenize=False,
        max_raw_steps=3,
        max_context_chars=32000,
        max_steps=8,
        seed=args.seed,
        verbose=args.verbose,
        prompt_key="webthink_simple6",
    )
    fever_base = SimpleNamespace(
        split="dev",
        max_examples=args.max_examples,
        tokenize=False,
        max_raw_steps=3,
        max_context_chars=32000,
        max_steps=5,
        seed=args.seed,
        verbose=args.verbose,
        prompt_key="webthink_simple3",
    )

    print("=" * 60)
    print("ReAct vs ReAct + trajectory tokenization")
    print("=" * 60)

    print("\n--- HotpotQA dev: ReAct (baseline) ---")
    hotpot_baseline = run_hotpotqa.run_eval(base)
    print("\n--- HotpotQA dev: ReAct + tokenization ---")
    base.tokenize = True
    hotpot_tokenize = run_hotpotqa.run_eval(base)

    print("\n--- FEVER dev: ReAct (baseline) ---")
    fever_baseline = run_fever.run_eval(fever_base)
    print("\n--- FEVER dev: ReAct + tokenization ---")
    fever_base.tokenize = True
    fever_tokenize = run_fever.run_eval(fever_base)

    print("\n" + "=" * 60)
    print("Comparison (EM, n=%d per task)" % args.max_examples)
    print("=" * 60)
    print("%-25s %10s %10s" % ("", "ReAct", "ReAct+Token"))
    print("%-25s %10.4f %10.4f" % ("HotpotQA dev", hotpot_baseline, hotpot_tokenize))
    print("%-25s %10.4f %10.4f" % ("FEVER dev", fever_baseline, fever_tokenize))
    print("=" * 60)


if __name__ == "__main__":
    main()
