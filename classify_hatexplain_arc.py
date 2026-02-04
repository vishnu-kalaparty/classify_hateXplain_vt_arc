import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ARC_API_KEY")
if API_KEY is None:
    raise RuntimeError(
        "Please set ARC_API_KEY environment variable with your ARC LLM API key "
    )

ARC_API_URL = "http://localhost:8080/v1/completions"
MAX_TOKENS = 4096

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "HateXplain-master", "Data")
DATASET_JSON = os.path.join(DATA_DIR, "dataset.json")

DEFAULT_MAX_SAMPLES = 30  # -1 = all; >0 = cap
DEFAULT_OUTPUT = "hatexplain_gpt_oss_120b_parallel_hate_normal_only.jsonl"
DEFAULT_SLEEP_SECONDS = 0
DEFAULT_NUM_THREADS = 5  # Number of threads for parallel API calls

# START_INDEX: Set to resume from a specific index (0 = start fresh, >0 = resume/append)
START_INDEX = 0

# Few-shot prompting configuration
ENABLE_FEW_SHOT = True  # Set to True to enable few-shot prompting
FEW_SHOT_EXAMPLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "few_shot_examples.json")

# Bridge prompt that connects the base prompt with few-shot examples
FEW_SHOT_BRIDGE_PROMPT = """
---
Here are some examples to guide your classification:

{examples}

Now, analyze the following text using the same approach:
"""

# HateXplain label mapping (need offensive for correct majority vote, then filter)
LABEL_MAP_INT = {0: "Hateful", 1: "Normal", 2: "Offensive"}
LABEL_MAP_STR = {
    "hatespeech": "Hateful",
    "hate": "Hateful",
    "hateful": "Hateful",
    "offensive": "Offensive",
    "normal": "Normal",
    "none": "Normal",
}

PROMPT_TEMPLATE = """You are an expert content moderator responsible for determining whether a given text violates hate speech policies. 

Task: Analyze the input text and classify it as Hateful or Normal by strictly following the steps below.
-Step 1: Apply the following definition of hate speech - Abusive speech that constitutes a direct, serious attack on an individual or a group based on protected attributes such as race, ethnicity, religion, nationality, gender, sexual orientation, or disability. It often involves an attacker expressing hatred directly toward a target.
-Step 2: Evaluate whether the input text meets this definition.
-Step 3: If it meets the definition, classify it as 'Hateful'. Otherwise, classify it as 'Normal'.

---
INPUT TEXT:
{input_text}

---
RESPONSE FORMAT:
1. Classification: [Hateful / Normal]
2. Key phrases: [short quotes or "None"]
3. Reasoning: [1–3 sentences explaining the classification, done by following the steps, citing any relevant evidence from the text.]
"""


def load_few_shot_examples() -> list[dict]:
    """
    Load few-shot examples from the external JSON file.
    Raises an error if the file is missing or empty.
    """
    if not os.path.exists(FEW_SHOT_EXAMPLES_FILE):
        raise FileNotFoundError(
            f"Few-shot examples file not found: {FEW_SHOT_EXAMPLES_FILE}. "
            "Please create the file or disable few-shot prompting by setting ENABLE_FEW_SHOT = False."
        )
    
    with open(FEW_SHOT_EXAMPLES_FILE, "r", encoding="utf-8") as f:
        examples = json.load(f)
    
    if not examples:
        raise ValueError(
            f"Few-shot examples file is empty: {FEW_SHOT_EXAMPLES_FILE}. "
            "Please add examples or disable few-shot prompting by setting ENABLE_FEW_SHOT = False."
        )
    
    return examples


def format_few_shot_examples(examples: list[dict]) -> str:
    """
    Format few-shot examples into a string for the prompt.
    Expected format per example: {"text": "...", "classification": "Hateful" or "Normal"}
    """
    formatted_examples = []
    for i, ex in enumerate(examples, 1):
        text = ex.get("text", "")
        classification = ex.get("classification", "")
        formatted_examples.append(
            f"Example {i}:\n"
            f"Text: \"{text}\"\n"
            f"Classification: {classification}"
        )
    return "\n\n".join(formatted_examples)


def build_prompt(input_text: str, few_shot_examples: list[dict] | None = None) -> str:
    """
    Build the complete prompt for classification.
    If few_shot_examples is provided, includes the bridge prompt with examples.
    """
    base_prompt = PROMPT_TEMPLATE.format(input_text=input_text)
    
    if few_shot_examples:
        examples_str = format_few_shot_examples(few_shot_examples)
        bridge = FEW_SHOT_BRIDGE_PROMPT.format(examples=examples_str)
        # Insert bridge prompt before the INPUT TEXT section
        parts = base_prompt.split("---\nINPUT TEXT:")
        if len(parts) == 2:
            return parts[0] + bridge + "---\nINPUT TEXT:" + parts[1]
    
    return base_prompt



def call_vllm_completion(prompt: str) -> str:
    """Call vLLM /v1/completions endpoint. Returns choices[0].text."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0, # deterministic for research / bias analysis
        "seed": 42,
    }
    resp = requests.post(ARC_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


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


def load_hatexplain() -> list[Dict[str, Any]]:
    """Load all posts from dataset.json."""
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items: list[Dict[str, Any]] = []
    for post_id, ex in raw.items():
        if isinstance(ex, dict):
            ex = dict(ex)
            ex.setdefault("id", post_id)
            items.append(ex)
    return items


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
    """Extract classification from text after <|message|> token."""
    text = response_text or ""
    idx = text.rfind("<|message|>")
    if idx == -1:
        return "Unknown"
    after = text[idx + len("<|message|>"):]
    # Find "Classification:" and extract until newline
    c_idx = after.lower().find("classification:")
    if c_idx == -1:
        return "Unknown"
    start = c_idx + len("classification:")
    end = after.find("\n", start)
    label = after[start:end].strip().lower() if end != -1 else after[start:].strip().lower()
    if "hateful" in label:
        return "Hateful"
    if "normal" in label:
        return "Normal"
    return "Unknown"


def _process_single_example(
    example: Dict[str, Any],
    idx: int,
    total_examples: int,
    few_shot_examples: list[dict] | None = None,
) -> Dict[str, Any]:
    """
    Process a single example: extract data, call API, and return the record.
    This function is designed to be called from a thread pool.
    """
    text = get_post_text(example)
    gold_label = get_gold_label(example)
    annotator_labels, annotator_targets = get_annotator_labels_and_targets(example)
    targets_union = get_targets_union(example)
    prompt = build_prompt(text, few_shot_examples)

    try:
        response_text = call_vllm_completion(prompt)
    except Exception as e:
        response_text = f"ERROR: {e}"

    pred_label = extract_model_classification(response_text)

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
        "few_shot_enabled": few_shot_examples is not None,
    }
    return record


def classify_hatexplain(
    max_samples: int,
    output_path: str,
    sleep_seconds: float = 0.0,
    start_index: int = 0,
    num_threads: int = 5,
) -> None:
    """
    Run the LLM (vLLM completions) on HateXplain. Binary: Hateful vs Normal only.
    Saves JSONL and prints accuracy and 2x2 confusion matrix.

    max_samples: -1 = no limit; >0 = cap.
    start_index: Index to start from (0 = fresh start, >0 = resume/append mode).
                 When start_index > 0, files are appended to and metrics are skipped.
    num_threads: Number of threads for parallel API calls (default 5).
    """
    examples_iter = load_hatexplain()
    examples_iter = [ex for ex in examples_iter if get_gold_label(ex) in ("Hateful", "Normal")]
    examples_iter = [ex for ex in examples_iter if get_post_text(ex).strip()]
    if max_samples > 0:
        examples_iter = examples_iter[: min(max_samples, len(examples_iter))]

    total_examples = len(examples_iter)
    
    # Skip examples before start_index
    if start_index > 0:
        if start_index >= total_examples:
            print(f"START_INDEX ({start_index}) >= total examples ({total_examples}). Nothing to process.")
            return
        examples_iter = examples_iter[start_index:]
        print(f"Resuming from index {start_index}. Processing {len(examples_iter)} remaining examples.", flush=True)
    else:
        print(f"Running on Hateful and Normal only: {total_examples} examples", flush=True)

    # Load few-shot examples if enabled
    few_shot_examples = None
    if ENABLE_FEW_SHOT:
        few_shot_examples = load_few_shot_examples()
        print(f"Few-shot prompting ENABLED with {len(few_shot_examples)} examples from {FEW_SHOT_EXAMPLES_FILE}", flush=True)
    else:
        print("Few-shot prompting DISABLED", flush=True)

    print(f"Using {num_threads} threads for parallel API calls.", flush=True)

    num_examples = len(examples_iter)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    errors_path = output_path.replace(".jsonl", "_errors.jsonl")
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    error_count = 0

    # Use append mode if resuming, write mode if starting fresh
    file_mode = "a" if start_index > 0 else "w"

    # Process examples in parallel using ThreadPoolExecutor
    results: list[Dict[str, Any]] = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_idx = {}
        for i, example in enumerate(examples_iter):
            idx = start_index + i if start_index > 0 else i
            future = executor.submit(_process_single_example, example, idx, total_examples, few_shot_examples)
            future_to_idx[future] = idx

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                record = future.result()
                results.append(record)
            except Exception as e:
                # Handle unexpected errors in the thread
                results.append({
                    "index": idx,
                    "id": None,
                    "text": "",
                    "gold_label": "Unknown",
                    "pred_label": "Unknown",
                    "annotator_labels": [],
                    "annotator_targets": [],
                    "targets_union": [],
                    "model_response": f"THREAD_ERROR: {e}",
                })
            completed_count += 1
            print(f"[{completed_count}/{num_examples}] processed (index {idx})", flush=True)

    # Sort results by index to maintain order in output file
    results.sort(key=lambda r: r["index"])

    # Write results to files
    with open(output_path, file_mode, encoding="utf-8") as fout, \
         open(errors_path, file_mode, encoding="utf-8") as ferr:
        for record in results:
            fout.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

            if record["pred_label"] == "Unknown":
                ferr.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                error_count += 1

            if record["pred_label"] in {"Hateful", "Normal"}:
                gold_labels.append(record["gold_label"])
                pred_labels.append(record["pred_label"])

    print(f"Saved classifications to {output_path}")
    if error_count > 0:
        print(f"Saved {error_count} unparseable entries to {errors_path}")

    # Skip metrics calculation if resuming from a non-zero index
    # (incomplete data would give incorrect metrics)
    if start_index > 0:
        print("\nSkipping metrics calculation (resuming from non-zero index).")
        print("Run the separate metrics script after all processing is complete.")
        return

    if not gold_labels:
        print("No valid gold/predicted label pairs collected; skipping evaluation.")
        return

    total = len(gold_labels)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    acc = correct / total
    print(f"\nEvaluation on {total} examples:")
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")

    labels = ["Hateful", "Normal"]
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
    classify_hatexplain(
        max_samples=DEFAULT_MAX_SAMPLES,
        output_path=DEFAULT_OUTPUT,
        sleep_seconds=DEFAULT_SLEEP_SECONDS,
        start_index=START_INDEX,
        num_threads=DEFAULT_NUM_THREADS,
    )


if __name__ == "__main__":
    main()


