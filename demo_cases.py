#!/usr/bin/env python3
"""
生成几个 ReAct 轨迹的「完整版 vs 压缩版」样例，便于对比效果。
不调用任何 LLM API，只做 tokenize 并打印。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from trajectory_tokenizer import tokenize_trajectory, parse_react_steps, count_steps_in_prompt

INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], (2) Lookup[keyword], (3) Finish[answer].
Question: Who is Milhouse (The Simpsons) named after?
"""


def make_case(name: str, trajectory_suffix: str, max_raw_steps: int = 2):
    full_prompt = INSTRUCTION.strip() + "\n" + trajectory_suffix.strip()
    instruction_prefix = INSTRUCTION.strip() + "\n"
    tokenized = tokenize_trajectory(
        full_prompt,
        instruction_prefix,
        max_raw_steps=max_raw_steps,
        max_thought=60,
        max_obs=100,
    )
    steps_full = count_steps_in_prompt(full_prompt)
    steps_tok = count_steps_in_prompt(tokenized)
    len_full = len(full_prompt)
    len_tok = len(tokenized)
    return {
        "name": name,
        "full_prompt": full_prompt,
        "tokenized_prompt": tokenized,
        "steps_full": steps_full,
        "steps_tok": steps_tok,
        "len_full": len_full,
        "len_tok": len_tok,
        "saved_chars": len_full - len_tok,
    }


def main():
    # Case 1: 3 步 → 压缩掉第 1 步，保留 2、3 完整
    case1_traj = """
Thought 1: I need to search for Milhouse to find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in The Simpsons. He is named after Richard Nixon's middle name.

Thought 2: The first paragraph says he is named after Richard Nixon's middle name. Let me look up "named after" to get the exact sentence.
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon.

Thought 3: I have found the answer. Milhouse was named after Richard Nixon.
Action 3:
"""
    c1 = make_case("Case1_Milhouse_3steps", case1_traj, max_raw_steps=2)
    # Case 2: 5 步长轨迹，压缩前 3 步
    case2_traj = """
Thought 1: Need to find the capital of France. I'll search Paris.
Action 1: Search[Paris]
Observation 1: Paris is the capital and largest city of France.

Thought 2: That was direct. Let me also confirm the country France.
Action 2: Search[France]
Observation 2: France is a country in Western Europe. Its capital is Paris.

Thought 3: I have two confirmations. Let me look up "capital" in the France passage.
Action 3: Lookup[capital]
Observation 3: (Result 1 / 1) Its capital is Paris.

Thought 4: So the capital of France is Paris. I'll double-check with one more lookup.
Action 4: Lookup[Paris]
Observation 4: (Result 1 / 1) Paris is the capital and largest city of France.

Thought 5: I'm confident. The answer is Paris.
Action 5:
"""
    c2 = make_case("Case2_France_capital_5steps", case2_traj, max_raw_steps=2)
    # Case 3: 4 步，保留最后 2 步完整
    case3_traj = """
Thought 1: Find when the Eiffel Tower was built. Search Eiffel Tower.
Action 1: Search[Eiffel Tower]
Observation 1: The Eiffel Tower is a wrought-iron lattice tower in Paris. It is named after engineer Gustave Eiffel.

Thought 2: The first paragraph doesn't give the date. Look up "built" or "constructed".
Action 2: Lookup[built]
Observation 2: (Result 1 / 1) It was built from 1887 to 1889.

Thought 3: So built 1887-1889. I need to give a single answer; typically we say 1889 when it opened.
Action 3: Lookup[1889]
Observation 3: (Result 1 / 1) The tower was opened on 31 March 1889.

Thought 4: The Eiffel Tower was built and opened in 1889. I'll finish with that.
Action 4:
"""
    c3 = make_case("Case3_Eiffel_4steps", case3_traj, max_raw_steps=2)

    for c in (c1, c2, c3):
        print("=" * 70)
        print(c["name"])
        print("=" * 70)
        print("Steps: full = %d, after tokenize = %d" % (c["steps_full"], c["steps_tok"]))
        print("Length: full = %d chars, tokenized = %d chars (saved %d)" % (c["len_full"], c["len_tok"], c["saved_chars"]))
        print()
        print("--- FULL PROMPT (last 600 chars) ---")
        print(c["full_prompt"][-600:])
        print()
        print("--- TOKENIZED PROMPT (last 600 chars) ---")
        print(c["tokenized_prompt"][-600:])
        print()

    # 输出用于对比的「下一句」提示
    print("=" * 70)
    print("PROMPTS FOR SIDE-BY-SIDE TEST (next step)")
    print("=" * 70)
    print("Below: same case in FULL vs TOKENIZED. Which context is enough to output the next 'Thought N / Action N'?")
    print()
    for c in (c1, c2, c3):
        print("--- %s ---" % c["name"])
        print("[FULL length %d]" % c["len_full"])
        print(c["full_prompt"][-800:])
        print()
        print("[TOKENIZED length %d]" % c["len_tok"])
        print(c["tokenized_prompt"][-800:])
        print()


if __name__ == "__main__":
    main()
