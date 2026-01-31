# Trajectory Tokenization — Introduction / 轨迹 Token 化 — 介绍

---

## 1. Overview / 概述

### English

**Trajectory Tokenization** is a method that keeps ReAct-style agent trajectories within context limits by **compressing older (Thought, Action, Observation) steps into short one-line tokens** and **keeping only the last N steps in full**. It builds on [ReAct](https://arxiv.org/abs/2210.03629) (ICLR 2023): same loop (Thought → Action → Observation), same tasks (e.g. HotpotQA, FEVER), but when the prompt grows too long, we summarize the history instead of truncating it. No extra model, no training—just a deterministic summarization rule applied before each LLM call when the prompt exceeds a character threshold.

This repository provides the implementation, evaluation scripts (ReAct baseline vs ReAct + trajectory tokenization), and one-click comparison on HotpotQA and FEVER using the original ReAct data and prompts.

### 中文

**Trajectory Tokenization（轨迹 Token 化）** 通过把**较早的 (Thought, Action, Observation) 步压缩成简短的单行 token**，并**只保留最近 N 步的完整内容**，使 ReAct 式智能体的轨迹在上下文长度限制内可继续扩展。它在 [ReAct](https://arxiv.org/abs/2210.03629)（ICLR 2023）的基础上工作：同样的循环（Thought → Action → Observation）、同样的任务（如 HotpotQA、FEVER），但当 prompt 过长时，我们对历史做摘要而不是简单截断。无需额外模型或训练，仅在 prompt 超过字符阈值时，在每次 LLM 调用前应用确定性的摘要规则。

本仓库提供实现、评测脚本（ReAct 基线 vs ReAct + trajectory tokenization）以及在 HotpotQA 和 FEVER 上使用原始 ReAct 数据和 prompt 的一键对比。

---

## 2. Motivation / 动机

### English

In **ReAct**, the agent repeatedly produces **Thought** (reasoning), **Action** (e.g. `Search[entity]`), and receives **Observation** (e.g. Wikipedia snippet). The full trajectory—all past Thought, Action, Observation—is concatenated into the prompt for the next turn. So:

- **Context length grows linearly with the number of steps.** Long-horizon tasks (many search/read steps) quickly exceed the model’s context window.
- **Truncating from the left** (dropping oldest steps) loses early reasoning and facts.
- **Truncating from the right** (dropping latest steps) breaks the immediate context the model needs to act.

We want **bounded context** without discarding the *structure* of the trajectory: the model should still see “what was done” in the past, but in a compact form, and keep full detail only for the most recent steps.

### 中文

在 **ReAct** 中，智能体不断产生 **Thought**（推理）、**Action**（如 `Search[entity]`）并得到 **Observation**（如维基片段）。完整轨迹——所有过去的 Thought、Action、Observation——都会拼进下一轮的 prompt，因此：

- **上下文长度随步数线性增长。** 长程任务（多次搜索/阅读）很快会超出模型的上下文窗口。
- **从左侧截断**（丢掉最早几步）会丢失早期推理和事实。
- **从右侧截断**（丢掉最近几步）会破坏模型决策所需的当前上下文。

我们希望在**不破坏轨迹结构**的前提下控制上下文长度：模型仍应“看到”过去做过什么，但以紧凑形式呈现，仅对最近几步保留完整细节。

---

## 3. Core Idea / 核心思想

### English

- **When** the current prompt length exceeds a threshold (e.g. 8000 characters), we **compress** the trajectory:
  1. **Parse** the trajectory part of the prompt into steps: each step = (Thought, Action, Observation).
  2. **Summarize** all but the last N steps into **one-line tokens**:  
     `[Step i] [truncated_thought | action | truncated_obs]`  
     (thought and obs are truncated by character limit; action is kept as-is).
  3. **Keep** the last N steps in **full** ReAct format (Thought k: ... Action k: ... Observation k: ...).
- **Result:** The prompt becomes: `[instruction + question] + [token1] [token2] ... + [full step N-1] [full step N]`. Context length is bounded: a fixed number of tokens for the past + a fixed number of full steps for the present.

So: **ReAct** = full history every step → unbounded growth. **Trajectory Tokenization** = tokenized (summarized) history + full recent steps → bounded context.

### 中文

- **当**当前 prompt 长度超过阈值（如 8000 字符）时，我们对轨迹做**压缩**：
  1. **解析** prompt 中的轨迹部分为若干步，每步 = (Thought, Action, Observation)。
  2. **摘要**除最后 N 步外的所有步为**单行 token**：  
     `[Step i] [截断的 thought | action | 截断的 obs]`  
     （thought 和 obs 按字符数截断，action 保留）。
  3. **保留**最后 N 步的**完整** ReAct 格式（Thought k: ... Action k: ... Observation k: ...）。
- **效果**：prompt 变为 `[指令 + 问题] + [token1] [token2] ... + [完整步 N-1] [完整步 N]`。上下文长度有界：过去用固定数量的 token，当前用固定数量的完整步。

因此：**ReAct** = 每步都保留完整历史 → 长度无界增长。**Trajectory Tokenization** = 摘要化历史 + 近期完整步 → 上下文有界。

---

## 4. Schematic / 示意图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ReAct (baseline): full trajectory in context                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Prompt:  [Q]  T1  A1  O1  T2  A2  O2  T3  A3  O3  ...  Tk  Ak  Ok  →  [next]   │
│           ↑_____________________________ all in full ___________________________↑  │
│  Context length grows with steps  →  may exceed limit  →  truncate / fail        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  ReAct + Trajectory Tokenization: bounded context                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Prompt:  [Q]  τ1  τ2  ...  τ_{k-N}   │  T A O  ...  Tk Ak Ok  │  → [next]        │
│           ↑    compressed tokens      ↑   last N steps full   ↑                  │
│  τ = one-line token per step; T/A/O = full Thought/Action/Observation            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Notes / 实现要点

### English

- **Deterministic, no LLM for summarization:** Each step is turned into a token by truncating thought/obs to a max length and concatenating with the action. No extra API call.
- **Trigger:** Compression runs only when `len(full_prompt) > max_context_chars` (default 8000). Until then, behavior is identical to ReAct.
- **Parameters:**
  - `max_raw_steps`: number of most recent steps to keep in full (default 3).
  - `max_context_chars`: trigger threshold (default 8000).
  - `max_thought` / `max_obs`: max characters kept in each token for thought and observation (default 60, 100). If the rebuilt prompt is still too long, the code can recursively reduce these and/or `max_raw_steps`.

See `trajectory_tokenizer.py`: `parse_react_steps`, `summarize_step`, `tokenize_trajectory`.

### 中文

- **确定性、无需 LLM 做摘要：** 每一步通过截断 thought/obs 到最大长度并与 action 拼接成 token，不增加额外 API 调用。
- **触发条件：** 仅当 `len(full_prompt) > max_context_chars`（默认 8000）时才做压缩，此前与 ReAct 完全一致。
- **参数：**
  - `max_raw_steps`：保留完整内容的最近步数（默认 3）。
  - `max_context_chars`：触发压缩的字符阈值（默认 8000）。
  - `max_thought` / `max_obs`：每个 token 中 thought 和 observation 的最大字符数（默认 60、100）。若压缩后仍超长，代码会递归减小这些值或 `max_raw_steps`。

见 `trajectory_tokenizer.py`：`parse_react_steps`、`summarize_step`、`tokenize_trajectory`。

---

## 6. Comparison with related work / 与相关方法对比

### English

Several recent works also compress or fold agent context (e.g. **ReSum**, **Context-Folding**, **FoldAct**). Main differences from **Trajectory Tokenization**:

| Aspect | ReSum / Context-Folding / FoldAct (typical) | Trajectory Tokenization (this repo) |
|--------|--------------------------------------------|-------------------------------------|
| **Training** | Use **RL / GRPO** (ReSum-GRPO, FoldGRPO, FoldAct) so the agent learns to act under summarized or folded context. | **No training.** Inference-only; same ReAct loop, same model. |
| **Summary production** | Often **LLM-generated** summaries or learned summarization; summary can condition future policy. | **Deterministic rule**: truncate thought/obs by character limit, keep action; **no extra LLM call** for summarization. |
| **Structure** | **Context-Folding**: agent branches into sub-trajectories, then “folds” a subtask into a summary when done. **ReSum**: periodic summarization into “compact reasoning states.” | **Single linear trajectory**: older steps → one-line tokens; last N steps stay full. No branching/folding. |
| **When it applies** | Usually requires **new training** or **new agent design** (summary-conditioned policy, process rewards, etc.). | **Drop-in** for existing ReAct: same prompt, same env; add `--tokenize` when context is long. |

In short: **Trajectory Tokenization** is a **lightweight, training-free** way to bound context: deterministic step-wise tokens + full recent steps, no RL, no LLM summarizer. Methods like ReSum/FoldAct aim for stronger long-horizon performance via **training** and sometimes **procedural folding** (subtask → sub-trajectory → fold); they are complementary (you could add training on top of tokenization later).

### 中文

近年也有不少工作对 agent 的上下文做压缩或折叠（如 **ReSum**、**Context-Folding**、**FoldAct**）。与 **Trajectory Tokenization** 的主要区别如下：

| 维度 | ReSum / Context-Folding / FoldAct（典型做法） | Trajectory Tokenization（本仓库） |
|------|-----------------------------------------------|-----------------------------------|
| **训练** | 使用 **RL / GRPO**（ReSum-GRPO、FoldGRPO、FoldAct）让 agent 在摘要或折叠后的上下文中行动。 | **不训练**。仅推理；同一 ReAct 循环、同一模型。 |
| **摘要如何产生** | 多为 **LLM 生成**摘要或学习式摘要；摘要可条件化后续策略。 | **确定性规则**：按字符截断 thought/obs，保留 action；**不做额外 LLM 调用**做摘要。 |
| **结构** | **Context-Folding**：agent 分支为子轨迹，子任务完成后“折叠”成摘要。**ReSum**：周期性将历史压成“紧凑推理状态”。 | **单条线性轨迹**：较早步 → 单行 token；最近 N 步保持完整。无分支/折叠。 |
| **适用场景** | 通常需要**重新训练**或**新 agent 设计**（摘要条件策略、过程奖励等）。 | 对现有 ReAct **即插即用**：相同 prompt、相同环境；上下文长时加 `--tokenize` 即可。 |

简言之：**Trajectory Tokenization** 是一种**轻量、免训练**的上下文有界方案——确定性的一步一 token + 近期完整步，无需 RL，无需 LLM 做摘要。ReSum/FoldAct 等则通过**训练**和（部分工作的）**过程式折叠**（子任务 → 子轨迹 → 折叠）追求更强的长程表现；二者可互补（后续也可在 tokenization 之上加训练）。

---

## 7. Effect / 效果

### English

We compare **ReAct (baseline)** and **ReAct + trajectory tokenization** on the same dev sets (HotpotQA, FEVER) with the same data and prompts. Metric: **Exact Match (EM)**.

- **Quick comparison (5 examples per task):**  
  `./run_all.sh` or `python run_comparison.py --max_examples 5`  
  Prints a table: HotpotQA dev (ReAct vs ReAct+Token), FEVER dev (ReAct vs ReAct+Token).

- **Larger comparison (e.g. 500 examples):**  
  `python run_comparison.py --max_examples 500`  
  Use this for more stable EM estimates (closer to paper settings).

You can also run each task separately with or without `--tokenize` (see README). The table lets you see whether tokenization helps (e.g. when trajectories are long) or stays on par with baseline when context is short.

### 中文

我们在相同 dev 集（HotpotQA、FEVER）、相同数据和 prompt 下对比 **ReAct（基线）** 与 **ReAct + trajectory tokenization**。指标：**Exact Match (EM)**。

- **快速对比（每任务 5 条）：**  
  `./run_all.sh` 或 `python run_comparison.py --max_examples 5`  
  会打印表格：HotpotQA dev（ReAct vs ReAct+Token）、FEVER dev（ReAct vs ReAct+Token）。

- **更大规模对比（如 500 条）：**  
  `python run_comparison.py --max_examples 500`  
  用于更稳定的 EM 估计（更接近论文设置）。

也可对每个任务单独跑，加或不加 `--tokenize`（见 README）。通过表格可以看出在轨迹较长时 tokenization 是否带来提升，或在上下文较短时是否与基线相当。

---

## 8. How to Use / 如何使用

### English

**Environment**

- Set `OPENAI_API_KEY`.
- Install: `pip install openai requests beautifulsoup4`

**One-click comparison (5 examples per task)**

```bash
cd trajectory_tokenization
export OPENAI_API_KEY=your_key
chmod +x run_all.sh && ./run_all.sh
```

**Run comparison with more examples**

```bash
python run_comparison.py --max_examples 500
```

**Run a single task (HotpotQA or FEVER) with or without tokenization**

```bash
# HotpotQA dev, baseline
python run_hotpotqa.py --split dev --max_examples 500

# HotpotQA dev, with trajectory tokenization
python run_hotpotqa.py --split dev --max_examples 500 --tokenize

# FEVER dev
python run_fever.py --split dev --max_examples 500
python run_fever.py --split dev --max_examples 500 --tokenize
```

**Main options**

| Option | Meaning |
|--------|--------|
| `--tokenize` | Use trajectory tokenization (ReAct+Token). |
| `--max_raw_steps N` | Keep last N steps in full (default 3). |
| `--max_context_chars C` | Compress when prompt length > C (default 8000). |
| `--max_examples M` | Number of dev examples to run. |

**Scripts**

| Script | Description |
|--------|-------------|
| `run_comparison.py` | Run ReAct vs ReAct+tokenization on HotpotQA + FEVER, print EM table. |
| `run_hotpotqa.py` | ReAct on HotpotQA; `--tokenize` = with tokenization. |
| `run_fever.py` | ReAct on FEVER; `--tokenize` = with tokenization. |
| `run_all.sh` | One-click: `python run_comparison.py --max_examples 5`. |
| `trajectory_tokenizer.py` | Core: `tokenize_trajectory()`, `parse_react_steps()`, etc. |
| `react_loop.py` | ReAct loop; when `use_tokenization=True`, calls tokenizer when prompt is long. |
| `test_tokenizer.py` | Unit test for tokenizer (no API). |

Run all commands from the `trajectory_tokenization` directory. Data: `data/hotpot_dev_v1_simplified.json`, `data/paper_dev.jsonl` (original ReAct data).

### 中文

**环境**

- 设置 `OPENAI_API_KEY`。
- 安装：`pip install openai requests beautifulsoup4`

**一键对比（每任务 5 条）**

```bash
cd trajectory_tokenization
export OPENAI_API_KEY=your_key
chmod +x run_all.sh && ./run_all.sh
```

**更多条数对比**

```bash
python run_comparison.py --max_examples 500
```

**单任务（HotpotQA 或 FEVER）是否使用 tokenization**

```bash
# HotpotQA dev，基线
python run_hotpotqa.py --split dev --max_examples 500

# HotpotQA dev，使用 trajectory tokenization
python run_hotpotqa.py --split dev --max_examples 500 --tokenize

# FEVER dev
python run_fever.py --split dev --max_examples 500
python run_fever.py --split dev --max_examples 500 --tokenize
```

**主要参数**

| 参数 | 含义 |
|------|------|
| `--tokenize` | 使用轨迹 token 化（ReAct+Token）。 |
| `--max_raw_steps N` | 保留最近 N 步完整（默认 3）。 |
| `--max_context_chars C` | prompt 长度超过 C 时压缩（默认 8000）。 |
| `--max_examples M` | 运行的 dev 样本数。 |

**脚本**

| 脚本 | 说明 |
|------|------|
| `run_comparison.py` | 在 HotpotQA + FEVER 上跑 ReAct vs ReAct+tokenization，打印 EM 表。 |
| `run_hotpotqa.py` | HotpotQA 上的 ReAct；`--tokenize` 表示使用 tokenization。 |
| `run_fever.py` | FEVER 上的 ReAct；`--tokenize` 表示使用 tokenization。 |
| `run_all.sh` | 一键执行：`python run_comparison.py --max_examples 5`。 |
| `trajectory_tokenizer.py` | 核心：`tokenize_trajectory()`、`parse_react_steps()` 等。 |
| `react_loop.py` | ReAct 主循环；`use_tokenization=True` 时在 prompt 过长时调用 tokenizer。 |
| `test_tokenizer.py` | tokenizer 单元测试（不调用 API）。 |

所有命令均在 `trajectory_tokenization` 目录下执行。数据：`data/hotpot_dev_v1_simplified.json`、`data/paper_dev.jsonl`（原始 ReAct 数据）。

---

## 9. Citation / 引用

If you use this code, please cite this repository / 若使用本代码，请引用本仓库：

```bibtex
@software{trajectory_tokenization,
  title = {Trajectory Tokenization},
  author = {Ding, Liang},
  year = {2026},
  url = {https://github.com/alphadl/trajectory_tokenization},
}
```

This work builds on ReAct / 本工作基于 ReAct：

```bibtex
@inproceedings{yao2023react,
  title = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  author = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023},
  url = {https://arxiv.org/abs/2210.03629},
}
```
