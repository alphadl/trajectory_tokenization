# ReAct + Trajectory Tokenization

Based on [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (ICLR 2023).

**This release:** trajectory tokenization on ReAct. **Comparison = ReAct (baseline) vs ReAct + trajectory tokenization** on HotpotQA and FEVER (original ReAct data). Tokenization compresses older (Thought, Action, Observation) steps into short tokens so long history fits in context; the last N steps stay in full.

---

## Method name

**Unified name: Trajectory Tokenization.**  
In code / scripts: `trajectory_tokenizer`, `--tokenize`, etc. In prose: "trajectory tokenization" or "Trajectory Tokenization."

---

## Core idea & difference from ReAct

**ReAct** keeps the **entire** trajectory in the prompt: every (Thought, Action, Observation) is stored in full. Context length grows with steps and can hit the model limit, forcing truncation or failure.

**This method** keeps context bounded by **compressing older steps into one-line tokens** and **keeping only the last N steps in full**. When prompt length exceeds a threshold, we (1) parse the trajectory into steps, (2) summarize older steps into tokens like `[thought \| action \| obs]`, (3) leave the last N steps verbatim. So the model always sees a short "memory" of the past + full detail for the recent reasoning.

### Schematic

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ReAct (baseline): full trajectory in context                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Prompt:  [Q]  T1  A1  O1  T2  A2  O2  T3  A3  O3  ...  Tk  Ak  Ok  →  [next]   │
│           ↑_____________________________ all in full ___________________________↑│
│  Context length grows with steps  →  may exceed limit  →  truncate / fail       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  ReAct + Trajectory Tokenization (this method): bounded context                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Prompt:  [Q]  τ1  τ2  ...  τ_{k-N}   │  T A O  ...  Tk Ak Ok  │  → [next]       │
│           ↑    compressed tokens     ↑   last N steps full   ↑                 │
│           └── older steps: "[thought | action | obs]" ──────┘                   │
│  Context length bounded: prefix (tokens) + fixed window (full steps)            │
└─────────────────────────────────────────────────────────────────────────────────┘

  τ = token (summarized step):  [truncated_thought | action | truncated_obs]
  T/A/O = Thought / Action / Observation (full text)
  N = max_raw_steps (e.g. 3)
```

### In one sentence

**ReAct** keeps full history; **Trajectory Tokenization** keeps a **tokenized (summarized) history** plus **full recent steps**, so long-horizon ReAct stays within context limits without losing the structure of the trajectory.

---

## Extreme long trajectory: why our method wins

When the trajectory has **many steps** (e.g. 45–100 steps) and **large context** (35k–80k characters), full ReAct quickly **exceeds** a typical context limit (default 32k chars). Trajectory Tokenization **compresses** the history so the prompt stays **within the limit** while preserving a summary of all steps + full detail for the last N steps.

**Run the demo:** `python demo_extreme_cases.py`

| Case            | Steps | Full ReAct | Tokenization | Saved  | Compression |
|-----------------|-------|------------|---------------|--------|-------------|
| ~35k context    | 45    | ~35k chars | ~12k–18k chars| ~18k+  | **~50%+**   |
| ~50k context    | 65    | ~50k chars | ~18k–24k chars| ~26k+  | **~52%+**   |
| ~65k context    | 85    | ~65k chars | ~24k–30k chars| ~35k+  | **~54%+**   |
| ~80k context    | 100   | ~80k chars | ~28k–32k chars| ~48k+  | **~60%+**   |

*(Exact numbers: run `python demo_extreme_cases.py`. Settings: `max_raw_steps=3`, `max_context_chars=32000`.)*

**Effect under 32k context limit:**

- **Full ReAct:** At 35k+ chars, the **earlier steps are truncated** (model only sees the last ~32k chars). Early Search/Lookup and key facts can be **lost**; the reasoning chain is **broken**.
- **Trajectory Tokenization:** All cases fit **within 32k**. The model always sees **summary tokens for every step** plus **full last 3 steps**. No step is dropped; the chain stays usable for the next action or `Finish[...]`.

So: **the more steps and the longer the trajectory, the more our method outperforms full ReAct**—bounded context with full structural memory vs. truncation and information loss.

---

## Setup

- Set `OPENAI_API_KEY`.
- Install: `pip install -r requirements-tokenization.txt` (or `pip install openai requests beautifulsoup4 gym numpy`)

## Quick start (one-click comparison)

```bash
cd trajectory_tokenization
export OPENAI_API_KEY=your_key
chmod +x run_all.sh && ./run_all.sh
```

Runs HotpotQA and FEVER with **ReAct (baseline)** and **ReAct + tokenization**, then prints an EM comparison table (default 5 examples per setting).

---

## Scripts (release)

| Script | Description |
|--------|-------------|
| **run_comparison.py** | ReAct vs ReAct+tokenization on HotpotQA + FEVER; print EM table. |
| **run_hotpotqa.py** | ReAct on HotpotQA dev. `--tokenize` = ReAct+tokenization. |
| **run_fever.py** | ReAct on FEVER dev. `--tokenize` = ReAct+tokenization. |
| **run_all.sh** | One-click: `python run_comparison.py --max_examples 5`. |
| **trajectory_tokenizer.py** | Parse/summarize trajectory; `tokenize_trajectory()`. |
| **react_loop.py** | ReAct loop with optional tokenization. |
| **test_tokenizer.py** | Unit test for tokenizer (no API). |
| **demo_extreme_cases.py** | Extreme long trajectory (35k/50k/65k/80k) full vs tokenized comparison; no API. |

Dependencies: `wikienv.py`, `wrappers.py`.

---

## Full comparison (e.g. 500 examples)

```bash
python run_comparison.py --max_examples 500
```

Or per task:

```bash
python run_hotpotqa.py --split dev --max_examples 500
python run_hotpotqa.py --split dev --max_examples 500 --tokenize
python run_fever.py --split dev --max_examples 500
python run_fever.py --split dev --max_examples 500 --tokenize
```

---

## Options

- `--tokenize`: use trajectory tokenization.
- `--max_raw_steps N`: keep last N steps in full (default 3).
- `--max_context_chars C`: compress when prompt length > C (default 32000; tune for your model’s context).
- `--max_examples M`: number of dev examples.
- `--prompt_key K`: prompt key in JSON (e.g. `webthink_simple6` for HotpotQA, `webthink_simple3` for FEVER).

---

## Data

Original ReAct data: `data/hotpot_dev_v1_simplified.json`, `data/paper_dev.jsonl`. Run from the `trajectory_tokenization` directory.

---

## Related projects

- **[AdaRubrics](https://github.com/alphadl/AdaRubrics)** — Adaptive dynamic rubric evaluator for agent trajectories: generates task-specific dimensions and scores runs for filtering/RLHF. Use it to score and filter trajectories; trajectory tokenization keeps those trajectories compact in context during inference.
- **[AgentHER](https://github.com/alphadl/AgentHER)** — Hindsight Experience Replay for LLM agents: relabel failed trajectories into valid training data (SFT/DPO). Complements this repo when you have existing failed runs to recover; trajectory tokenization addresses *context length*, AgentHER addresses *data recovery*.
- **[AgentSynth](https://github.com/alphadl/AgentSynth)** — Synthetic agent data pipeline (forward + back-translation, execution-based reject sampling). Use it to generate SFT-style trajectories; use this repo to keep long ReAct runs within context via tokenization.

---

## Citation

If you use this code, please cite this repository:

```bibtex
@software{trajectory_tokenization,
  title = {Trajectory Tokenization},
  author = {Ding, Liang},
  year = {2026},
  url = {https://github.com/alphadl/trajectory_tokenization},
}
```

This work builds on ReAct:

```bibtex
@inproceedings{yao2023react,
  title = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  author = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023},
  url = {https://arxiv.org/abs/2210.03629},
}
```
