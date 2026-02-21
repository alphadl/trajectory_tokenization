"""
ReAct loop with optional trajectory tokenization.
Works with HotpotQA and FEVER (Wikipedia env); same interface for both.
"""
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional: use tokenization when enabled
try:
    from trajectory_tokenizer import tokenize_trajectory
except ImportError:
    def tokenize_trajectory(*args, **kwargs):
        return args[0] if args else ""


def llm(prompt: str, stop: List[str], api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> str:
    try:
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e


def run_react(
    env: Any,
    instruction: str,
    question: str,
    max_steps: int = 8,
    llm_fn: Optional[Callable[[str, List[str]], str]] = None,
    use_tokenization: bool = False,
    max_raw_steps: int = 3,
    max_context_chars: Optional[int] = None,
    to_print: bool = True,
    idx: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Run one ReAct episode. Returns (reward, info).
    - env: gym-style env with reset(idx=...) and step(action).
    - instruction: ReAct instruction + few-shot examples (no trailing question).
    - question: current question/claim (e.g. "Question: ..." or "Claim: ...").
    - use_tokenization: if True, compress older steps into tokens when building prompt.
    """
    if llm_fn is None:
        llm_fn = llm
    obs, reward, done, info = None, 0, False, {}
    try:
        obs = env.reset(idx=idx if idx is not None else getattr(env, "data_idx", None))
    except TypeError:
        obs = env.reset()
    if to_print:
        print(obs[:200] + "..." if len(obs) > 200 else obs)
    instruction_prefix = instruction + obs.strip() + "\n"
    full_prompt = instruction_prefix
    n_calls, n_badcalls = 0, 0
    for i in range(1, max_steps + 1):
        if use_tokenization and len(full_prompt) > (max_context_chars or 32000):
            full_prompt = tokenize_trajectory(
                full_prompt,
                instruction_prefix,
                max_raw_steps=max_raw_steps,
                max_total_chars=max_context_chars,
            )
        n_calls += 1
        thought_action = llm_fn(full_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ", 1)
        except Exception:
            if to_print:
                print("parse retry:", thought_action[:150])
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action = llm_fn(full_prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()
        action = action[0].lower() + action[1:] if action else ""
        obs, reward, done, info = env.step(action)
        obs = obs.replace("\\n", "")
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        full_prompt += step_str
        if to_print:
            print(step_str[:300] + "..." if len(step_str) > 300 else step_str)
        if done:
            break
    if not done:
        obs, reward, done, info = env.step("finish[]")
    if to_print:
        print(info, "\n")
    info["n_calls"] = n_calls
    info["n_badcalls"] = n_badcalls
    info["traj"] = full_prompt
    return reward, info
