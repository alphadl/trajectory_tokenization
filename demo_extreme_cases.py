#!/usr/bin/env python3
"""
极端长轨迹对比：35k / 50k / 65k / 80k+ 上下文、多步数（阈值 32k）。
对比「完整 ReAct」与「Trajectory Tokenization」的字数、结构与可读性。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap
_bootstrap.setup(__file__)

from trajectory_tokenizer import tokenize_trajectory, parse_react_steps, count_steps_in_prompt

# 固定 instruction + question，约 500 字
INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], (2) Lookup[keyword], (3) Finish[answer].
Here are some examples.
Example 1: Question: What is the capital of France? Thought 1: I need to find the capital. Action 1: Search[France]
Observation 1: France is a country. Its capital is Paris. Thought 2: I have the answer. Action 2: Finish[Paris]
Example 2: ...
Question: Which actor played the main character in the film that won the Academy Award for Best Picture in 2020?
"""

# 生成单步：Thought / Action / Observation 长度可控，用于凑到目标总字数
def make_step(step_num: int, thought_len: int, obs_len: int, action: str = None) -> str:
    thought = (
        f"I need to search for relevant entities and then look up specific facts. "
        f"Step {step_num}: reasoning about the question and previous observations. "
        f"Let me try to find the film first, then the actor. " * (max(0, thought_len // 120) + 1)
    )[:thought_len]
    if action is None:
        if step_num % 3 == 1:
            action = f"Search[entity_{step_num}]"
        elif step_num % 3 == 2:
            action = f"Lookup[keyword_{step_num}]"
        else:
            action = "Lookup[detail]"
    obs = (
        f"(Result 1 / 1) This is a long observation paragraph simulating Wikipedia or document retrieval. "
        f"The relevant sentence might be here. For step {step_num} we have some factual content. "
        f"Sometimes the observation contains multiple sentences and redundant information. " * (max(0, obs_len // 150) + 1)
    )[:obs_len]
    return f"Thought {step_num}: {thought}\nAction {step_num}: {action}\nObservation {step_num}: {obs}\n"


def build_long_trajectory(num_steps: int, thought_len: int = 120, obs_len: int = 280) -> str:
    """生成 num_steps 步的轨迹，每步约 thought_len + 40 + obs_len 字符。"""
    parts = []
    for i in range(1, num_steps + 1):
        parts.append(make_step(i, thought_len, obs_len))
    return "\n".join(parts)


def run_extreme_case(name: str, num_steps: int, target_full_chars: int, max_raw_steps: int = 3, max_context_chars: int = 32000):
    """生成接近 target_full_chars 字数的轨迹，然后做 full vs tokenized 对比。"""
    # 每步大约 40 + thought + obs，反推 thought_len 和 obs_len
    per_step = target_full_chars // num_steps
    obs_len = int(per_step * 0.65)
    thought_len = int(per_step * 0.35) - 50
    thought_len = max(60, thought_len)
    obs_len = max(100, obs_len)

    trajectory = build_long_trajectory(num_steps, thought_len=thought_len, obs_len=obs_len)
    full_prompt = INSTRUCTION.strip() + "\n" + trajectory
    instruction_prefix = INSTRUCTION.strip() + "\n"

    tokenized = tokenize_trajectory(
        full_prompt,
        instruction_prefix,
        max_raw_steps=max_raw_steps,
        max_total_chars=max_context_chars,
        max_thought=60,
        max_obs=100,
    )

    n_full = count_steps_in_prompt(full_prompt)
    len_full = len(full_prompt)
    len_tok = len(tokenized)
    steps_parsed = len(parse_react_steps(trajectory))

    return {
        "name": name,
        "num_steps": num_steps,
        "steps_parsed": steps_parsed,
        "len_full": len_full,
        "len_tok": len_tok,
        "saved": len_full - len_tok,
        "ratio": len_tok / len_full if len_full else 0,
        "full_prompt": full_prompt,
        "tokenized_prompt": tokenized,
        "over_threshold_full": len_full > max_context_chars,
        "over_threshold_tok": len_tok > max_context_chars,
    }


def main():
    threshold = 32_000
    cases = [
        ("~35k 上下文, 45 步", 45, 35_000, 3, threshold),
        ("~50k 上下文, 65 步", 65, 50_000, 3, threshold),
        ("~65k 上下文, 85 步", 85, 65_000, 3, threshold),
        ("~80k 上下文, 100 步", 100, 80_000, 3, threshold),
    ]

    print("=" * 72)
    print("极端长轨迹对比：完整 ReAct vs Trajectory Tokenization")
    print("  max_raw_steps=3, max_context_chars=%d" % threshold)
    print("=" * 72)

    results = []
    for name, num_steps, target, max_raw, max_ctx in cases:
        c = run_extreme_case(name, num_steps, target, max_raw_steps=max_raw, max_context_chars=max_ctx)
        results.append(c)
        print()
        print("-" * 72)
        print(c["name"])
        print("-" * 72)
        print("  步数:           %d" % c["num_steps"])
        print("  完整 ReAct:     %d 字符  (超过 %dk: %s)" % (c["len_full"], threshold // 1000, "是" if c["over_threshold_full"] else "否"))
        print("  Tokenization:   %d 字符  (超过 %dk: %s)" % (c["len_tok"], threshold // 1000, "是" if c["over_threshold_tok"] else "否"))
        print("  节省:           %d 字符  (压缩比: %.2f%%)" % (c["saved"], (1 - c["ratio"]) * 100))
        print()

    # 汇总表
    print("=" * 72)
    print("汇总表")
    print("=" * 72)
    print("%-28s %10s %10s %10s %8s" % ("Case", "Full(字符)", "Token(字符)", "节省", "压缩比"))
    print("-" * 72)
    for c in results:
        print("%-28s %10d %10d %10d %7.1f%%" % (
            c["name"][:28], c["len_full"], c["len_tok"], c["saved"], (1 - c["ratio"]) * 100))
    print("=" * 72)

    # 展示 50k case 的结构片段：完整版中间 vs 压缩版整体结构
    c50 = results[1]
    print()
    print("=" * 72)
    print("【50k 案例】结构对比（片段）")
    print("=" * 72)
    print()
    print("--- 完整 ReAct：中间一段（约第 30–35 步）---")
    mid = len(c50["full_prompt"]) // 2
    snippet = c50["full_prompt"][mid - 400 : mid + 400]
    print(snippet)
    print()
    print("--- Tokenization：压缩后的整体结构（前 1200 字 + 最后 800 字）---")
    tok = c50["tokenized_prompt"]
    print(tok[:1200])
    print("  ... [中间省略] ...")
    print(tok[-800:])
    print()

    # 80k 的“最后几步”对比：模型实际看到的结尾
    print("=" * 72)
    print("【超长 80k 案例】模型看到的「当前上下文结尾」对比")
    print("=" * 72)
    c80 = results[3]
    print()
    print("--- 完整 ReAct 最后 1000 字（若上下文上限 32k，前面会被截断）---")
    print(c80["full_prompt"][-1000:])
    print()
    print("--- Tokenization 最后 1000 字（压到 32k 内，模型看到完整摘要+最近 3 步）---")
    print(c80["tokenized_prompt"][-1000:])
    print()

    # 效果差异总结：假设模型上下文上限 32k
    print("=" * 72)
    print("【效果差异总结】假设上下文上限 = 32k 字符")
    print("=" * 72)
    print("""
  完整 ReAct:
    - 35k 案例: 超出 32k → 前面约 3k 字被截断，模型看不到最早几步。
    - 50k 案例: 超出 32k → 只能保留「最后约 32k 字符」，前约 18k 字（约 20+ 步）被截断。
    - 65k 案例: 前约 33k 字（约 35+ 步）丢失，模型只看到后半段轨迹。
    - 80k 案例: 前约 48k 字（约 50+ 步）被截断，模型几乎只看到最后 30+ 步，推理链易断裂。

  Trajectory Tokenization:
    - 所有案例压缩后均在 32k 内（约 1.5万–2.5万字符）。
    - 模型始终看到：早期步骤的「摘要 token」+ 最近 3 步完整 (Thought/Action/Observation)。
    - 不丢步数：[Step i] token 保留梗概，最后 3 步完整，适合续写下一步或 Finish。

  结论：步数越多、轨迹越长，完整 ReAct 在固定上下文下越容易截断、丢失早期信息；
       Tokenization 把长度压到上限以内，保留全局梗概 + 近期细节，两种方法在极端长轨迹下效果差异显著。
""")
    print("=" * 72)


if __name__ == "__main__":
    main()
