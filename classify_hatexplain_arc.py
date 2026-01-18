import os
import json
import time
from typing import Dict, Any

import requests


API_KEY = os.environ.get("ARC_API_KEY")
if API_KEY is None:
    raise RuntimeError(
        "Please set ARC_API_KEY environment variable with your ARC LLM API key "
        "(generate it at https://llm.arc.vt.edu under User profile > Settings > Account > API keys)."
    )

ARC_API_URL = "https://llm-api.arc.vt.edu/api/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "HateXplain-master", "Data")
DATASET_JSON = os.path.join(DATA_DIR, "dataset.json")
SPLIT_JSON = os.path.join(DATA_DIR, "post_id_divisions.json")

DEFAULT_SPLIT = "all"  # "all" = train + val + test
DEFAULT_MAX_SAMPLES = 300  # set it to -1 if you want to run on all samples
DEFAULT_OUTPUT = f"hatexplain_gpt_oss_120b_{DEFAULT_SPLIT}_updated_prompt.jsonl"
DEFAULT_SLEEP_SECONDS = 0.5  # seconds between requests to respect rate limits

# HateXplain label mapping (for gold labels)
LABEL_MAP_INT = {0: "Hateful", 1: "Normal", 2: "Offensive"}
LABEL_MAP_STR = {
    "hatespeech": "Hateful",
    "hate": "Hateful",
    "hateful": "Hateful",
    "offensive": "Offensive",
    "normal": "Normal",
    "none": "Normal",
}

# Exact prompt template from the user
PROMPT_TEMPLATE = """You are an expert content moderator responsible for ensuring compliance of posts with hate speech policies. 

---
Task: Analyze the input text and classify it as Hateful, Offensive, or Normal.

Follow these steps exactly and conservatively:

Step 1: Scan the text for potentially abusive, derogatory, or strong language.
• If none is present, classify as Normal and stop.

Step 2: Determine whether any such language is used as a direct attack
or is merely quoted, discussed, criticized, sarcastic, or contextual.
• If it is quoted, contextual, sarcastic, or condemning, classify as Normal and stop.

Step 3: If the language is a direct attack, identify the target.
• If there is no clear target, classify as Normal and stop.

Step 4: Determine whether the target is defined by a protected attribute
(race, religion, ethnicity, nationality, gender, sexual orientation, disability).

Step 5: Make the final decision:
• If the attack targets a protected class, classify as Hateful.
• If the attack targets a non-protected individual or group, classify as Offensive.
• Otherwise, classify as Normal.

---
INPUT TEXT:
{input_text}

---
RESPONSE FORMAT:
1. Classification: [Hateful / Offensive / Normal]
2. Key phrases: [short quotes or "None"]
3. Concise reasoning: [1–3 sentences following the steps]
"""



def call_arc_chat_completion(prompt: str, reasoning_effort: str = "high") -> str:
    """
    Call the ARC VT OpenAI-compatible endpoint for chat completions.

    Uses the gpt-oss-120b model as documented at:
    https://docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "reasoning_effort": reasoning_effort,
        "temperature": 0.0,  # deterministic for research / bias analysis
    }

    resp = requests.post(ARC_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _normalize_label(label_val: Any) -> str:
    if isinstance(label_val, int):
        return LABEL_MAP_INT.get(label_val, "Unknown")
    if isinstance(label_val, str):
        return LABEL_MAP_STR.get(label_val.strip().lower(), "Unknown")
    return "Unknown"


def _majority_vote(labels: list) -> str:
    labels_norm = [_normalize_label(x) for x in labels]
    labels_norm = [x for x in labels_norm if x != "Unknown"]
    if not labels_norm:
        return "Unknown"
    return max(set(labels_norm), key=labels_norm.count)


def load_hatexplain_split(split: str) -> list[Dict[str, Any]]:
    """
    Load HateXplain directly from the local GitHub repo copy using:
      - dataset.json: full dataset as a dict keyed by post_id
      - post_id_divisions.json: lists of ids for each split ("train", "val", "test")
    """
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items_by_id: dict[str, Dict[str, Any]] = {}
    for post_id, ex in raw.items():
        if isinstance(ex, dict):
            ex = dict(ex)
            ex.setdefault("id", post_id)
            items_by_id[str(post_id)] = ex

    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        divisions = json.load(f)

    if split == "all":
        split_ids = (
            divisions["train"]
            + divisions.get("val", divisions.get("validation", []))
            + divisions["test"]
        )
    else:
        split_ids = divisions[split]  # this script is only for this repo; assume key exists
    split_ids = [str(x) for x in split_ids]
    return [items_by_id[i] for i in split_ids if i in items_by_id]


def get_post_text(example: Dict[str, Any]) -> str:
    """
    Extract the input text from a HateXplain example by joining 'post_tokens'.
    """
    if "post_tokens" in example and example["post_tokens"]:
        return " ".join(example["post_tokens"])
    if "text" in example and example["text"]:
        return example["text"]
    if "post" in example and example["post"]:
        return example["post"]
    return ""


def get_gold_label(example: Dict[str, Any]) -> str:
    """
    Derive a single gold label from the example.
    - GitHub repo format typically includes `annotators` with `label`
    - HF format also includes `annotators`
    - fallback: example["label"]
    """
    if "annotators" in example and example["annotators"]:
        labels = []
        for ann in example["annotators"]:
            if isinstance(ann, dict) and "label" in ann:
                labels.append(ann["label"])
        if labels:
            return _majority_vote(labels)

    if "label" in example:
        label_val = example["label"]
        if isinstance(label_val, list):
            return _majority_vote(label_val)
        return _normalize_label(label_val)

    return "Unknown"


def get_annotator_labels_and_targets(example: Dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """
    Return per-annotator normalized labels and their target lists.
    Useful for bias analysis comparing LLM vs individual annotators.
    """
    labels_out: list[str] = []
    targets_out: list[list[str]] = []
    for ann in example.get("annotators", []):
        if not isinstance(ann, dict):
            continue
        labels_out.append(_normalize_label(ann.get("label")))
        t = ann.get("target") or []
        if isinstance(t, list):
            targets_out.append([str(x) for x in t])
        else:
            targets_out.append([str(t)])
    return labels_out, targets_out


def get_targets_union(example: Dict[str, Any]) -> list[str]:
    """
    Collect the union of all non-'None' targets across annotators.
    This is helpful for analyzing bias by target group.
    """
    _, ann_targets = get_annotator_labels_and_targets(example)
    targets: set[str] = set()
    for tgt_list in ann_targets:
        for t in tgt_list:
            if t and t.lower() != "none":
                targets.add(t)
    return sorted(targets)


def extract_model_classification(response_text: str) -> str:
    """
    Parse the model's response to get the predicted class string.
    Expected line: '1. Classification: ...' (handles markdown formatting like **Classification:**)
    """
    import re
    
    for line in response_text.splitlines():
        # Remove markdown bold formatting (**text**) for matching
        line_clean = re.sub(r'\*\*', '', line)
        line_lower = line_clean.lower().strip()
        
        # Check if line starts with "1. classification" (after removing markdown)
        if line_lower.startswith("1. classification"):
            # Extract part after colon
            if ":" in line_clean:
                part = line_clean.split(":", 1)[1].strip()
            else:
                continue
                
            # Remove markdown formatting from the part
            part = re.sub(r'\*\*', '', part)
            
            # handle bracket form
            if "[" in part and "]" in part:
                part = part.split("[", 1)[1].split("]", 1)[0].strip()
            
            part_lower = part.lower()
            if "hateful" in part_lower:
                return "Hateful"
            if "offensive" in part_lower:
                return "Offensive"
            if "normal" in part_lower:
                return "Normal"
    return "Unknown"


def classify_hatexplain(
    split: str,
    max_samples: int,
    output_path: str,
    sleep_seconds: float = 0.0,
) -> None:
    """
    Run the ARC LLM on HateXplain examples and save results as JSONL.
    Also prints accuracy + confusion matrix vs the dataset's gold labels.

    Each line in the output file has:
      {
        "index": ...,
        "id": ...,
        "text": ...,
        "gold_label": ...,
        "pred_label": ...,
        "model_response": ...
      }
    """
    examples_iter = load_hatexplain_split(split)
    if max_samples > 0:
        examples_iter = examples_iter[: min(max_samples, len(examples_iter))]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    num_examples = len(examples_iter)
    gold_labels: list[str] = []
    pred_labels: list[str] = []

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, example in enumerate(examples_iter):
            text = get_post_text(example)
            if not text:
                continue

            gold_label = get_gold_label(example)
            annotator_labels, annotator_targets = get_annotator_labels_and_targets(example)
            targets_union = get_targets_union(example)
            prompt = PROMPT_TEMPLATE.format(input_text=text)

            try:
                response_text = call_arc_chat_completion(prompt)
            except Exception as e:
                response_text = f"ERROR: {e}"

            pred_label = extract_model_classification(response_text)
            
            # Debug: print responses that couldn't be parsed (for evaluation)
            if pred_label == "Unknown":
                print(f"\nWARNING: Could not parse classification from response (example {idx+1}):")
                print(f"Full response:\n{response_text}")
                print()

            record = {
                "index": idx,
                "id": example.get("id"),
                "text": text,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "annotator_labels": annotator_labels,
                "annotator_targets": annotator_targets,
                "targets_union": targets_union,
                "model_response": response_text,
            }
            fout.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

            if gold_label in {"Hateful", "Offensive", "Normal"} and pred_label in {"Hateful", "Offensive", "Normal"}:
                gold_labels.append(gold_label)
                pred_labels.append(pred_label)

            print(f"[{idx + 1}/{num_examples}] processed", flush=True)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Saved classifications to {output_path}")

    if not gold_labels:
        print("No valid gold/predicted label pairs collected; skipping evaluation.")
        return

    total = len(gold_labels)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    acc = correct / total
    print(f"\nEvaluation on {total} examples:")
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")

    labels = ["Hateful", "Offensive", "Normal"]
    conf = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold_labels, pred_labels):
        conf[g][p] += 1

    print("\nConfusion matrix (rows=gold, cols=pred):")
    print("\t" + "\t".join(labels))
    for g in labels:
        print("\t".join([g] + [str(conf[g][p]) for p in labels]))

    # Save metrics to JSON file
    metrics_path = output_path.replace(".jsonl", "_metrics.json")
    metrics = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "confusion_matrix": conf,
        "labels": labels,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics to {metrics_path}")


def main():
    max_samples = DEFAULT_MAX_SAMPLES if DEFAULT_MAX_SAMPLES >= 0 else 10**9
    classify_hatexplain(
        split=DEFAULT_SPLIT,
        max_samples=max_samples,
        output_path=DEFAULT_OUTPUT,
        sleep_seconds=DEFAULT_SLEEP_SECONDS,
    )


if __name__ == "__main__":
    main()


