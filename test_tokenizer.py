#!/usr/bin/env python3
"""Unit test for trajectory tokenization (no API call)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap
_bootstrap.setup(__file__)

from trajectory_tokenizer import (
    parse_react_steps,
    summarize_step,
    steps_to_full_text,
    tokenize_trajectory,
    count_steps_in_prompt,
)


class TestTrajectoryTokenizer(unittest.TestCase):
    """Tests for parse_react_steps, summarize_step, steps_to_full_text, tokenize_trajectory."""

    def test_parse_and_summarize(self):
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
        self.assertEqual(len(steps), 3)
        self.assertIn("Milhouse", steps[0][0])
        self.assertEqual(steps[0][1], "Search[Milhouse]")
        self.assertIn("Finish[Richard Nixon]", steps[2][1])
        tok = summarize_step(steps[0][0], steps[0][1], steps[0][2], max_thought=40, max_obs=50)
        self.assertIn("Search[Milhouse]", tok)
        self.assertLess(len(tok), 150)
        rebuilt = steps_to_full_text(steps, start_idx=1)
        self.assertIn("Thought 1:", rebuilt)
        self.assertIn("Observation 3:", rebuilt)

    def test_tokenize_trajectory(self):
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
        self.assertTrue("[Step 1]" in out or "[Step 2]" in out)
        self.assertIn("Thought 3:", out)
        self.assertIn("Action 3:", out)

    def test_count_steps_and_short_trajectory_unchanged(self):
        """When steps <= max_raw_steps, tokenize_trajectory returns prompt unchanged."""
        instruction = "Q: x?\n"
        full = instruction + "Thought 1: a\nAction 1: b\nObservation 1: c\n"
        self.assertEqual(count_steps_in_prompt(full), 1)
        out = tokenize_trajectory(full, instruction, max_raw_steps=3)
        self.assertEqual(out, full)


if __name__ == "__main__":
    unittest.main()
