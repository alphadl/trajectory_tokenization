#!/usr/bin/env python3
"""Unit test for trajectory tokenization (no API call)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from trajectory_tokenizer import (
    parse_react_steps,
    summarize_step,
    steps_to_full_text,
    tokenize_trajectory,
)


def test_parse_and_summarize():
    traj = """
Thought 1: I need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in The Simpsons.

Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon.

Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
"""
    steps = parse_react_steps(traj)
    assert len(steps) == 3
    assert "Milhouse" in steps[0][0]
    assert steps[0][1] == "Search[Milhouse]"
    assert "Finish[Richard Nixon]" in steps[2][1]
    tok = summarize_step(steps[0][0], steps[0][1], steps[0][2], max_thought=40, max_obs=50)
    assert "Search[Milhouse]" in tok
    assert len(tok) < 150
    rebuilt = steps_to_full_text(steps, start_idx=1)
    assert "Thought 1:" in rebuilt and "Observation 3:" in rebuilt
    print("parse + summarize + rebuild OK")


def test_tokenize_trajectory():
    instruction = "Solve QA.\nQuestion: Who is Milhouse named after?\n"
    full = instruction + """
Thought 1: Search Milhouse.
Action 1: Search[Milhouse]
Observation 1: Milhouse is a character on The Simpsons.

Thought 2: Look up named after.
Action 2: Lookup[named after]
Observation 2: Milhouse was named after Richard Nixon.

Thought 3: So answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
Observation 3: Episode finished.
"""
    out = tokenize_trajectory(full, instruction, max_raw_steps=1)
    assert "[Step 1]" in out or "[Step 2]" in out
    assert "Thought 3:" in out and "Action 3:" in out
    print("tokenize_trajectory OK")


if __name__ == "__main__":
    test_parse_and_summarize()
    test_tokenize_trajectory()
    print("All tokenizer tests passed.")
