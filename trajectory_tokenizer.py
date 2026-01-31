"""
Trajectory Tokenization for ReAct: compress (Thought, Action, Observation) history
into short tokens to reduce context length while preserving structure.
"""
import re
from typing import List, Tuple, Optional


def _truncate(s: str, max_len: int, suffix: str = "...") -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)].rstrip() + suffix


def parse_react_steps(trajectory_text: str) -> List[Tuple[str, str, str]]:
    """
    Parse ReAct trajectory string into list of (thought, action, obs).
    Expects format: "Thought i: ...\nAction i: ...\nObservation i: ...\n" (repeated).
    """
    steps = []
    # Match blocks: Thought i: ... Action i: ... Observation i: ...
    pattern = re.compile(
        r"Thought\s+(\d+):\s*(.*?)(?=Thought\s+\d+:|$)",
        re.DOTALL,
    )
    for m in pattern.finditer(trajectory_text):
        step_num = m.group(1)
        block = m.group(2).strip()
        action_pattern = re.compile(
            rf"Action\s+{re.escape(step_num)}:\s*(.*?)(?=Observation\s+{re.escape(step_num)}:|$)",
            re.DOTALL,
        )
        obs_pattern = re.compile(
            rf"Observation\s+{re.escape(step_num)}:\s*(.*?)(?=Thought\s+\d+:|$)",
            re.DOTALL,
        )
        action_m = action_pattern.search(block)
        obs_m = obs_pattern.search(block)
        thought = block[: action_m.start()].strip() if action_m else (block.split("\n")[0].strip() if block else "")
        action = action_m.group(1).strip() if action_m else ""
        obs = obs_m.group(1).strip() if obs_m else ""
        steps.append((thought, action, obs))
    return steps


def summarize_step(
    thought: str,
    action: str,
    obs: str,
    max_thought: int = 60,
    max_obs: int = 100,
) -> str:
    """Produce a one-line token summary for one ReAct step (no LLM, deterministic)."""
    t = _truncate(thought, max_thought)
    o = _truncate(obs, max_obs)
    return f"[{t} | {action} | {o}]"


def steps_to_full_text(steps: List[Tuple[str, str, str]], start_idx: int = 1) -> str:
    """Convert list of (thought, action, obs) back to ReAct format string."""
    lines = []
    for i, (thought, action, obs) in enumerate(steps):
        k = start_idx + i
        lines.append(f"Thought {k}: {thought}")
        lines.append(f"Action {k}: {action}")
        lines.append(f"Observation {k}: {obs}")
    return "\n".join(lines) + "\n"


def tokenize_trajectory(
    full_prompt: str,
    instruction_prefix: str,
    max_raw_steps: int = 3,
    max_total_chars: Optional[int] = None,
    max_thought: int = 60,
    max_obs: int = 100,
) -> str:
    """
    Compress trajectory by summarizing older steps into tokens; keep last max_raw_steps in full.
    - full_prompt: current full prompt (instruction + question + Thought 1 / Action 1 / Obs 1 / ...)
    - instruction_prefix: instruction + few-shot + question (so we know where trajectory starts).
    - max_raw_steps: number of most recent steps to keep in full.
    - max_total_chars: if set, try to keep total prompt under this (by increasing summarization).
    Returns rebuilt prompt with compressed history when applicable.
    """
    if not full_prompt.startswith(instruction_prefix):
        return full_prompt
    trajectory_part = full_prompt[len(instruction_prefix) :].lstrip()
    steps = parse_react_steps(trajectory_part)
    if len(steps) <= max_raw_steps:
        return full_prompt
    n_summarize = len(steps) - max_raw_steps
    summarized_tokens = []
    for i in range(n_summarize):
        thought, action, obs = steps[i]
        tok = summarize_step(thought, action, obs, max_thought=max_thought, max_obs=max_obs)
        summarized_tokens.append(f"[Step {i + 1}] {tok}")
    summary_block = "\n".join(summarized_tokens) + "\n\n"
    raw_steps = steps[n_summarize:]
    raw_text = steps_to_full_text(raw_steps, start_idx=n_summarize + 1)
    out = instruction_prefix + summary_block + raw_text
    if max_total_chars and len(out) > max_total_chars:
        return tokenize_trajectory(
            full_prompt,
            instruction_prefix,
            max_raw_steps=max(1, max_raw_steps - 1),
            max_total_chars=max_total_chars,
            max_thought=max(max_thought - 10, 30),
            max_obs=max(max_obs - 20, 50),
        )
    return out


def count_steps_in_prompt(prompt: str) -> int:
    """Count number of Thought/Action/Observation steps in prompt."""
    return len(re.findall(r"Thought\s+\d+:", prompt))
