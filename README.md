# Escalate

Routing and context-optimization layer for reducing LLM calls in coding-agent runtimes.

Escalate is a machine learning system that optimizes coding agents by intelligently routing requests between a **local policy model** and expensive **frontier models**. By learning when a cheaper local model is sufficient versus when a frontier model is truly needed, Escalate can dramatically reduce API costs and latency without sacrificing output quality.

---

## Architecture

### Layer 1 — Deterministic Proxy (LiteLLM)

The first layer is a lightweight, deterministic reverse proxy built on [LiteLLM](https://github.com/BerriAI/litellm). It sits in front of all LLM API calls made by the coding agent.

**Responsibilities:**
- Intercept every outbound LLM request before it reaches a frontier provider.
- Log request/response pairs (prompt, completion, token counts, latency, model used) to structured JSON logs.
- Provide a unified OpenAI-compatible API endpoint so the agent requires no code changes.
- Apply hard-coded routing rules (e.g., always use a cheap model for short fill-in-the-middle tasks).

**Configuration:** `proxy/config.yaml`  
**Entry point:** `proxy/proxy_server.py`

---

### Layer 2 — Learned Policy Model (Gemma 4 Fine-tune)

The second layer is a fine-tuned **Gemma 4** model that acts as a learned routing policy. It is trained on the logs collected by Layer 1 and learns to predict, given a request context, whether the local model can handle the task or whether escalation to a frontier model is required.

**Training pipeline:**

| Stage | Script | Description |
|-------|--------|-------------|
| Data formatting | `src/format_data.py` | Converts raw JSON logs into SFT / preference datasets |
| SFT warm-up | `src/train_sft.py` | Supervised fine-tuning with [Unsloth](https://github.com/unslothai/unsloth) + [TRL `SFTTrainer`](https://huggingface.co/docs/trl) |
| DPO alignment | `src/train_dpo.py` | Direct Preference Optimization with `DPOTrainer` to align routing decisions with human preference |

Experiment tracking is handled by [Weights & Biases](https://wandb.ai) (`wandb`).

---

## Repository Structure

```
Escalate/
├── data/
│   ├── raw/            # Raw JSON log files captured by the proxy (git-ignored)
│   └── processed/      # Formatted HuggingFace datasets (git-ignored)
├── models/             # Saved model checkpoints (git-ignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_unsloth_sft_test.ipynb
├── proxy/
│   ├── config.yaml     # LiteLLM proxy configuration
│   └── proxy_server.py # Proxy entry point
├── src/
│   ├── format_data.py  # Data formatting pipeline
│   ├── train_sft.py    # SFT training script
│   └── train_dpo.py    # DPO training script
├── .env                # API keys (git-ignored, copy from .env.example)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, WANDB_API_KEY, ...)
```

### 3. Start the proxy

```bash
litellm --config proxy/config.yaml
```

### 4. Point your coding agent at the proxy

Set your agent's `OPENAI_BASE_URL` (or equivalent) to `http://localhost:4000`.

### 5. Collect logs and train

Once you have collected enough logs in `data/raw/`, run the training pipeline:

```bash
python src/format_data.py
python src/train_sft.py
python src/train_dpo.py
```

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list. Key dependencies:

- **[unsloth](https://github.com/unslothai/unsloth)** — fast Gemma fine-tuning
- **[litellm](https://github.com/BerriAI/litellm)** — unified LLM proxy
- **[trl](https://huggingface.co/docs/trl)** — SFT and DPO trainers
- **[wandb](https://wandb.ai)** — experiment tracking
- **[datasets](https://huggingface.co/docs/datasets)** — dataset loading and processing
- **[transformers](https://huggingface.co/docs/transformers)** — model loading
- **[torch](https://pytorch.org)** — deep learning backend

---

# Escalate Router

Learned compute arbitration router for AI coding agents.

## What it does
Predicts the number of LLM calls required to solve a coding task,
enabling threshold-based routing decisions that reduce frontier model invocations.

## Results
- CV MAE: 15.8 ± 0.6 calls (on 26–100 call range)
- Held-out accuracy: 99% (91/92 instances)
- Length-only baseline: 50% — router learns real complexity signal

## Architecture
- Backbone: Gemma 8B (4-bit, frozen) for task embeddings
- Head: Ridge regression (2560-dim → call count prediction)
- Threshold: 49 calls (33rd percentile of training distribution)

## HuggingFace Models
- SFT: vedevpatel/escalate-router-sft-v1
- IPO: vedevpatel/escalate-router-ipo-v2
- Regression: vedevpatel/escalate-router-regression-v1

## Training Data
929 SWE-bench trajectories with LLM call counts from OpenHands agent runs.
DPO pairs labeled by relative efficiency (chosen_llm_calls vs rejected_llm_calls).



## License

MIT
