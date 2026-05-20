from datasets import load_dataset
import json
import os

ds = load_dataset("krzysztofwos/CodeAgent-Trajectories", split="train")

def label_waste(messages):
    """
    Scan the messages for signs of orchestration waste.
    Returns a waste label: 'redundant_call', 'clean', or 'verbose_reasoning'
    """
    tool_calls = [m for m in messages if m["role"] == "tool-call"]
    tool_responses = [m for m in messages if m["role"] == "tool-response"]

    # check for duplicate tool calls
    seen_args = []
    for tc in tool_calls:
        if tc["content"] in seen_args:
            return "redundant_call"
        seen_args.append(tc["content"])

    # if it took more than 3 tool-call and response pairs to reach final_answer, flag it
    if len(tool_calls) > 3:
        return "verbose_reasoning"

    return "clean"

def format_for_sft(example):
    """
    Converts a raw trajectory into a clean SFT training example.
    Only keeps: system prompt, user task, and the FIRST successful assistant response.
    """
    messages = example["messages"]
    task = example["task"]
    waste_label = label_waste(messages)

    # get the first assistant turn that leads to a correct tool call response
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    first_assistant = next((m for m in messages if m["role"] == "assistant"), None)
    first_tool_response = next((m for m in messages if m["role"] == "tool-response"), None)

    if not system_msg or not first_assistant or not first_tool_response:
        return None

    return {
        "task": task,
        "waste_label": waste_label,
        "messages": [
            {"role": "system", "content": "You are a compute arbitration router. Given a task and context, decide the most token-efficient action."},
            {"role": "user", "content": task},
            {"role": "assistant", "content": first_assistant["content"]}
        ]
    }

os.makedirs("data/processed", exist_ok=True)

clean_examples = []
verbose_examples = []

for example in ds:
    formatted = format_for_sft(example)
    if not formatted:
        continue
    if formatted["waste_label"] == "clean":
        clean_examples.append(formatted)
    else:
        verbose_examples.append(formatted)

# SFT training data - only clean trajectories
with open("data/processed/sft_train.jsonl", "w") as f:
    for ex in clean_examples:
        f.write(json.dumps(ex) + "\n")

# Rejected examples for DPO pairing later
with open("data/processed/dpo_rejected_candidates.jsonl", "w") as f:
    for ex in verbose_examples:
        f.write(json.dumps(ex) + "\n")

print(f"SFT examples (clean): {len(clean_examples)}")
print(f"DPO rejected candidates (verbose): {len(verbose_examples)}")

# waste label distribution
labels = [label_waste(ex["messages"]) for ex in ds]
from collections import Counter
print(Counter(labels))