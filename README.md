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

## Setup

- Set `OPENAI_API_KEY`.
- Install: `pip install openai requests beautifulsoup4`

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
- `--max_context_chars C`: compress when prompt length > C (default 8000).
- `--max_examples M`: number of dev examples.

---

## Data

Original ReAct data: `data/hotpot_dev_v1_simplified.json`, `data/paper_dev.jsonl`. Run from the `trajectory_tokenization` directory.

---

## Citation

If you use this code, please cite this repository:

```bibtex
@software{trajectory_tokenization,
  title = {Trajectory Tokenization},
  author = {Ding, Liang},
  year = {2025},
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
