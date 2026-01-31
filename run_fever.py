#!/usr/bin/env python3
"""
Run ReAct on FEVER dev set. Supports trajectory tokenization for long context.
Usage:
  python run_fever.py [--max_examples 500] [--tokenize] [--max_raw_steps 3] [--max_context_chars 8000]
"""
import argparse
import json
import os
import random
import sys
import time

_react_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_react_root)
sys.path.insert(0, _react_root)
import wikienv
import wrappers
from react_loop import run_react


def run_eval(args):
    """Run evaluation; returns EM (float). Used by run_comparison.py."""
    env = wikienv.WikiEnv()
    env = wrappers.FeverWrapper(env, split=args.split)
    env = wrappers.LoggingWrapper(env)

    folder = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(folder, "fever.json"), "r") as f:
        prompt_dict = json.load(f)
    if args.prompt_key not in prompt_dict:
        args.prompt_key = "webthink_simple3"
    webthink_examples = prompt_dict[args.prompt_key]
    instruction = (
        "Determine if there is Observation that SUPPORTS or REFUTES a Claim, "
        "or if there is NOT ENOUGH INFORMATION. "
        "Here are some examples.\n"
    )
    instruction += webthink_examples

    n_data = len(env)
    idxs = list(range(n_data))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[: args.max_examples]

    results = []
    infos = []
    t0 = time.time()
    for idx in idxs:
        try:
            r, info = run_react(
                env,
                instruction=instruction,
                question="",
                max_steps=args.max_steps,
                use_tokenization=args.tokenize,
                max_raw_steps=args.max_raw_steps,
                max_context_chars=args.max_context_chars,
                to_print=args.verbose,
                idx=idx,
            )
        except Exception as e:
            print(f"Error idx={idx}: {e}", file=sys.stderr)
            r, info = 0, {"em": 0, "question_idx": idx}
        results.append(info.get("em", r))
        infos.append(info)
        n_done = len(results)
        em_sum = sum(results)
        print(f"Done {n_done}/{len(idxs)} | EM so far: {em_sum}/{n_done} = {em_sum / n_done:.4f} | time: {(time.time() - t0) / n_done:.1f}s/sample")
    total_em = sum(results)
    print(f"\nFEVER {args.split} | n={len(results)} | EM = {total_em}/{len(results)} = {total_em / len(results):.4f}")
    if args.tokenize:
        print("(ReAct + trajectory tokenization)")
    return total_em / len(results) if results else 0.0


def main():
    parser = argparse.ArgumentParser(description="ReAct FEVER (baseline or + tokenization)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--max_raw_steps", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=8000)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prompt_key", type=str, default="webthink_simple3")
    args = parser.parse_args()
    return run_eval(args)


if __name__ == "__main__":
    main()
