import pickle
import torch
import numpy as np
from unsloth import FastLanguageModel

class EscalateRouter:
    def __init__(self, model_name="vedevpatel/escalate-router-regression-v1",
                 head_path="router_head.pkl"):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=512, load_in_4bit=True
        )
        FastLanguageModel.for_inference(self.model)
        with open(head_path, "rb") as f:
            state = pickle.load(f)
        self.scaler = state["scaler"]
        self.ridge = state["ridge"]
        self.threshold = state["threshold"]

    def route(self, task: str) -> dict:
        inputs = self.tokenizer.tokenizer(
            task, return_tensors="pt", truncation=True,
            max_length=256, padding=False
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model.model(
                **inputs, output_hidden_states=True
            )
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = ((hidden * mask).sum(1) / mask.sum(1)).squeeze().cpu().float().numpy()
        predicted_calls = self.ridge.predict(self.scaler.transform([emb]))[0]
        return {
            "decision": "HANDLE_LOCALLY" if predicted_calls <= self.threshold else "ESCALATE",
            "estimated_llm_calls": round(float(predicted_calls), 1),
            "threshold": self.threshold,
        }

if __name__ == "__main__":
    router = EscalateRouter()
    print(router.route("Fix a typo in README.md"))
    print(router.route("Implement OAuth2 from scratch"))