"""
train_dpo.py

IPO (Identity Preference Optimization) alignment of the SFT model on
OpenHands trajectory pairs labeled by LLM call efficiency.
Prefers trajectories that solved the task with fewer frontier model invocations.
"""

import json
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothDPOTrainer, UnslothDPOConfig as DPOConfig
from huggingface_hub import login
import wandb


SFT_MODEL = "vedevpatel/escalate-router-sft-v1"
OUTPUT_HF_REPO = "vedevpatel/escalate-router-ipo-v2"
DATA_PATH = "data/dpo_pairs.jsonl"
MAX_SEQ_LENGTH = 2048
ROUTING_THRESHOLD = 49  # 33rd percentile of chosen_llm_calls


def is_successful(text):
    markers = ["## Summary", "successfully implemented", "successfully fixed",
               "have successfully", "changes have been", "\u2705"]
    return any(m.lower() in text.lower() for m in markers) and len(text) > 300


def extract_last_turn(text):
    parts = text.split("<start_of_turn>model")
    return ("<start_of_turn>model" + parts[-1]) if len(parts) > 1 else text


def build_dpo_dataset(pairs_path: str):
    with open(pairs_path) as f:
        raw_pairs = [json.loads(l) for l in f]

    relabeled = []
    skipped = 0
    for p in raw_pairs:
        chosen_trimmed = extract_last_turn(p["chosen"])
        rejected_trimmed = extract_last_turn(p["rejected"])
        chosen_ok = is_successful(chosen_trimmed)
        rejected_ok = is_successful(rejected_trimmed)
        chosen_calls = p["chosen_llm_calls"]
        rejected_calls = p["rejected_llm_calls"]

        if chosen_ok and not rejected_ok:
            relabeled.append({"prompt": p["prompt"],
                               "chosen": chosen_trimmed, "rejected": rejected_trimmed})
        elif rejected_ok and not chosen_ok:
            relabeled.append({"prompt": p["prompt"],
                               "chosen": rejected_trimmed, "rejected": chosen_trimmed})
        elif chosen_ok and rejected_ok:
            if chosen_calls < rejected_calls:
                relabeled.append({"prompt": p["prompt"],
                                   "chosen": chosen_trimmed, "rejected": rejected_trimmed})
            elif rejected_calls < chosen_calls:
                relabeled.append({"prompt": p["prompt"],
                                   "chosen": rejected_trimmed, "rejected": chosen_trimmed})
            else:
                skipped += 1
        else:
            skipped += 1

    print(f"Clean pairs: {len(relabeled)}, Skipped: {skipped}")
    return Dataset.from_list(relabeled).shuffle(seed=42)


def train(hf_token: str):
    login(hf_token)
    wandb.init(project="escalate-router", name="gemma4-ipo-v2")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    dataset = build_dpo_dataset(DATA_PATH)

    trainer = UnslothDPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer.tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=5e-5,
            beta=0.5,
            loss_type="ipo",
            output_dir="outputs/ipo-v2",
            report_to="wandb",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=1024,
            remove_unused_columns=False,
            precompute_ref_log_probs=True,
        ),
    )
    trainer.train()

    model.push_to_hub(OUTPUT_HF_REPO)
    tokenizer.tokenizer.push_to_hub(OUTPUT_HF_REPO)
    print(f"Model pushed to {OUTPUT_HF_REPO}")


if __name__ == "__main__":
    import os
    train(os.environ["HF_TOKEN"])
