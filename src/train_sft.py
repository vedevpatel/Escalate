"""
train_sft.py

Supervised fine-tuning (SFT) of Gemma 4 E4B on OpenHands agent trajectories.
Trains the model to follow efficient coding-agent behavior before IPO alignment.
"""

import json
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
from huggingface_hub import login
import wandb


BASE_MODEL = "unsloth/gemma-4-e4b-it"
OUTPUT_HF_REPO = "vedevpatel/escalate-router-sft-v1"
DATA_PATH = "data/dpo_pairs.jsonl"
MAX_SEQ_LENGTH = 1024


def is_successful(text):
    markers = ["## Summary", "successfully implemented", "successfully fixed",
               "have successfully", "changes have been", "\u2705"]
    return any(m.lower() in text.lower() for m in markers) and len(text) > 300


def extract_last_turn(text):
    parts = text.split("<start_of_turn>model")
    return ("<start_of_turn>model" + parts[-1]) if len(parts) > 1 else text


def build_sft_dataset(pairs_path: str):
    with open(pairs_path) as f:
        raw_pairs = [json.loads(l) for l in f]

    examples = []
    for p in raw_pairs:
        chosen_response = extract_last_turn(p["chosen"])
        if is_successful(chosen_response):
            examples.append({
                "text": (
                    f"<start_of_turn>user\n{p['prompt']}<end_of_turn>\n"
                    f"{chosen_response}<end_of_turn>"
                )
            })
    print(f"SFT examples: {len(examples)}")
    return Dataset.from_list(examples).shuffle(seed=42)


def train(hf_token: str):
    login(hf_token)
    wandb.init(project="escalate-router", name="gemma4-sft-v1")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    dataset = build_sft_dataset(DATA_PATH)

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer.tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            output_dir="outputs/sft-v1",
            report_to="wandb",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
        ),
    )
    trainer.train()

    model.push_to_hub(OUTPUT_HF_REPO)
    tokenizer.tokenizer.push_to_hub(OUTPUT_HF_REPO)
    print(f"Model pushed to {OUTPUT_HF_REPO}")


if __name__ == "__main__":
    import os
    train(os.environ["HF_TOKEN"])
