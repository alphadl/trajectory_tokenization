#!/usr/bin/env python3
"""
Run ReAct on HotpotQA dev set. Supports trajectory tokenization for long context.
Usage:
  python run_hotpotqa.py [--max_examples 500] [--tokenize] [--max_raw_steps 3] [--max_context_chars 8000]
"""
import argparse
import json
import os
import random
import sys
import time

# ReAct repo root: run from script dir so data/ and prompts/ resolve
_react_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_react_root)
sys.path.insert(0, _react_root)
import wikienv
import wrappers
from react_loop import run_react


def step_env(env, action, max_attempts=10):
    import requests
    attempts = 0
    while attempts < max_attempts:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
    raise RuntimeError("env.step timed out")


def run_eval(args):
    """Run evaluation; returns EM (float). Used by run_comparison.py."""
    env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(env, split=args.split)
    env = wrappers.LoggingWrapper(env)

    folder = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(folder, "prompts_naive.json"), "r") as f:
        prompt_dict = json.load(f)
    if args.prompt_key not in prompt_dict:
        args.prompt_key = "webthink_simple6"
    webthink_examples = prompt_dict[args.prompt_key]
    instruction = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. "
        "Thought can reason about the current situation, and Action can be three types: "
        "(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. "
        "If not, it will return some similar entities to search. "
        "(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage. "
        "(3) Finish[answer], which returns the answer and finishes the task. "
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
    print(f"\nHotpotQA {args.split} | n={len(results)} | EM = {total_em}/{len(results)} = {total_em / len(results):.4f}")
    if args.tokenize:
        print("(ReAct + trajectory tokenization)")
    return total_em / len(results) if results else 0.0


def main():
    parser = argparse.ArgumentParser(description="ReAct HotpotQA (baseline or + tokenization)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train", "test"])
    parser.add_argument("--max_examples", type=int, default=500, help="Max dev examples to run")
    parser.add_argument("--tokenize", action="store_true", help="ReAct + trajectory tokenization")
    parser.add_argument("--max_raw_steps", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=8000)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prompt_key", type=str, default="webthink_simple6")
    args = parser.parse_args()
    return run_eval(args)


if __name__ == "__main__":
    main()
