import json
import pickle
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from unsloth import FastLanguageModel
from huggingface_hub import HfApi


SFT_MODEL = "vedevpatel/escalate-router-sft-v1"
OUTPUT_HF_REPO = "vedevpatel/escalate-router-regression-v1"


def is_successful(text):
    markers = ["## Summary", "successfully implemented", "successfully fixed",
               "have successfully", "changes have been", "\u2705"]
    return any(m.lower() in text.lower() for m in markers) and len(text) > 300


def extract_last_turn(text):
    parts = text.split("<start_of_turn>model")
    return ("<start_of_turn>model" + parts[-1]) if len(parts) > 1 else text


def get_embedding(model, tokenizer, text, max_len=256):
    inputs = tokenizer.tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_len, padding=False
    ).to("cuda")
    with torch.no_grad():
        outputs = model.model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    return ((hidden * mask).sum(1) / mask.sum(1)).squeeze().cpu().float().numpy()


def build_embeddings(pairs_path: str, model, tokenizer):
    with open(pairs_path) as f:
        raw_pairs = [json.loads(l) for l in f]

    X, y = [], []
    for p in raw_pairs:
        if is_successful(extract_last_turn(p["chosen"])):
            emb = get_embedding(model, tokenizer, p["prompt"][:400])
            X.append(emb)
            y.append(p["chosen_llm_calls"])
    return np.array(X), np.array(y)


def train_router(pairs_path: str = "data/dpo_pairs.jsonl"):
    print("Loading backbone...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL, max_seq_length=512, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    print("Extracting embeddings...")
    X, y = build_embeddings(pairs_path, model, tokenizer)
    print(f"Dataset: {X.shape}, y range: {y.min()}-{y.max()}, mean: {y.mean():.1f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"5-fold CV MAE: {-scores.mean():.1f} +/- {scores.std():.1f} calls")

    ridge.fit(X_scaled, y)
    threshold = int(np.percentile(y, 33))
    print(f"Routing threshold (33rd percentile): {threshold} calls")

    state = {"scaler": scaler, "ridge": ridge, "threshold": threshold}
    with open("router_head.pkl", "wb") as f:
        pickle.dump(state, f)
    print("Saved router_head.pkl")

    api = HfApi()
    api.upload_file(
        path_or_fileobj="router_head.pkl",
        path_in_repo="router_head.pkl",
        repo_id=OUTPUT_HF_REPO,
        repo_type="model",
    )
    print(f"Uploaded to HuggingFace: {OUTPUT_HF_REPO}")
    return state


if __name__ == "__main__":
    train_router()
