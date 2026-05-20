import pickle
import json
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download
from unsloth import FastLanguageModel


def load_router(model_name="vedevpatel/escalate-router-regression-v1"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=512, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    head_path = hf_hub_download(repo_id=model_name, filename="router_head.pkl")
    with open(head_path, "rb") as f:
        state = pickle.load(f)
    return model, tokenizer, state["scaler"], state["ridge"], state["threshold"]


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


def route(model, tokenizer, scaler, ridge, threshold, task: str) -> dict:
    emb = get_embedding(model, tokenizer, task)
    predicted_calls = ridge.predict(scaler.transform([emb]))[0]
    return {
        "decision": "HANDLE_LOCALLY" if predicted_calls <= threshold else "ESCALATE",
        "estimated_llm_calls": round(float(predicted_calls), 1),
        "threshold": threshold,
    }


def evaluate(pairs_path: str, model, tokenizer, scaler, ridge, threshold,
             low_threshold=49, high_threshold=61, n=100):
    import re

    def is_successful(text):
        markers = ["## Summary", "successfully implemented", "successfully fixed",
                   "have successfully", "changes have been", "\u2705"]
        return any(m.lower() in text.lower() for m in markers) and len(text) > 300

    def extract_last_turn(text):
        parts = text.split("<start_of_turn>model")
        return ("<start_of_turn>model" + parts[-1]) if len(parts) > 1 else text

    with open(pairs_path) as f:
        raw_pairs = [json.loads(l) for l in f]

    labeled = []
    for p in raw_pairs:
        success = is_successful(extract_last_turn(p["chosen"]))
        calls = p["chosen_llm_calls"]
        if success and calls <= low_threshold:
            labeled.append((p, "EFFICIENT"))
        elif calls >= high_threshold:
            labeled.append((p, "EXPENSIVE"))

    sample = labeled[:n]
    correct = 0
    results = []
    for pair, expected in sample:
        result = route(model, tokenizer, scaler, ridge, threshold, pair["prompt"][:400])
        decision = "EFFICIENT" if result["decision"] == "HANDLE_LOCALLY" else "EXPENSIVE"
        correct += (decision == expected)
        results.append({"expected": expected, "predicted": decision,
                        "estimated_calls": result["estimated_llm_calls"],
                        "true_calls": pair["chosen_llm_calls"]})

    efficient = [r for r in results if r["expected"] == "EFFICIENT"]
    expensive = [r for r in results if r["expected"] == "EXPENSIVE"]
    print(f"Overall accuracy:  {correct}/{len(results)} ({100*correct/len(results):.0f}%)")
    print(f"EFFICIENT recall:  {sum(r['predicted']==r['expected'] for r in efficient)}/{len(efficient)}")
    print(f"EXPENSIVE recall:  {sum(r['predicted']==r['expected'] for r in expensive)}/{len(expensive)}")
    mae = np.mean([abs(r['estimated_calls'] - r['true_calls']) for r in results])
    print(f"MAE:               {mae:.1f} calls")
    return results


if __name__ == "__main__":
    model, tokenizer, scaler, ridge, threshold = load_router()
    evaluate("data/dpo_pairs.jsonl", model, tokenizer, scaler, ridge, threshold)
