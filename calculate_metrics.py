"""
Standalone metrics calculation script for HateXplain classification results.

Reads the output JSONL and errors JSONL files and computes:
- Accuracy, Precision, Recall, F1 (macro, micro, weighted, per-class)
- Confusion matrix
- Target-distributed accuracy and F1 (per target group)
- Error analysis statistics

Usage:
    python calculate_metrics.py [--results PATH] [--errors PATH] [--output PATH]
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any

# Default file paths (same directory as script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(SCRIPT_DIR, "hatexplain_gpt_oss_120b_hate_normal_only.jsonl")
DEFAULT_ERRORS = os.path.join(SCRIPT_DIR, "hatexplain_gpt_oss_120b_hate_normal_only_errors.jsonl")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "hatexplain_gpt_oss_120b_hate_normal_only_full_metrics.json")

# Labels for binary classification
LABELS = ["Hateful", "Normal"]


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load records from a JSONL file (handles pretty-printed JSON objects)."""
    records = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return records
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Handle pretty-printed JSON objects (each object spans multiple lines)
    # Split by looking for patterns like "}\n{" at the start of a line
    depth = 0
    current_obj = []
    
    for line in content.split('\n'):
        current_obj.append(line)
        depth += line.count('{') - line.count('}')
        
        if depth == 0 and current_obj:
            obj_str = '\n'.join(current_obj).strip()
            if obj_str:
                try:
                    records.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    pass
            current_obj = []
    
    return records


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_confusion_matrix(gold_labels: list[str], pred_labels: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    """Compute confusion matrix as nested dict."""
    conf = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold_labels, pred_labels):
        if g in labels and p in labels:
            conf[g][p] += 1
    return conf


def compute_metrics_from_confusion(conf: dict[str, dict[str, int]], labels: list[str]) -> dict[str, Any]:
    """Compute all metrics from confusion matrix."""
    metrics = {}
    
    # Total counts
    total = sum(conf[g][p] for g in labels for p in labels)
    correct = sum(conf[g][g] for g in labels)
    
    metrics["total"] = total
    metrics["correct"] = correct
    metrics["accuracy"] = correct / total if total > 0 else 0.0
    
    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = conf[label][label]
        fp = sum(conf[g][label] for g in labels if g != label)
        fn = sum(conf[label][p] for p in labels if p != label)
        tn = total - tp - fp - fn
        
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
        
        per_class[label] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,  # actual count of this class
        }
    
    metrics["per_class"] = per_class
    
    # Macro averages (unweighted mean of per-class metrics)
    metrics["macro_precision"] = sum(per_class[l]["precision"] for l in labels) / len(labels)
    metrics["macro_recall"] = sum(per_class[l]["recall"] for l in labels) / len(labels)
    metrics["macro_f1"] = sum(per_class[l]["f1"] for l in labels) / len(labels)
    
    # Weighted averages (weighted by support)
    total_support = sum(per_class[l]["support"] for l in labels)
    if total_support > 0:
        metrics["weighted_precision"] = sum(per_class[l]["precision"] * per_class[l]["support"] for l in labels) / total_support
        metrics["weighted_recall"] = sum(per_class[l]["recall"] * per_class[l]["support"] for l in labels) / total_support
        metrics["weighted_f1"] = sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / total_support
    else:
        metrics["weighted_precision"] = 0.0
        metrics["weighted_recall"] = 0.0
        metrics["weighted_f1"] = 0.0
    
    # Micro averages (aggregate TP, FP, FN across all classes)
    total_tp = sum(per_class[l]["true_positives"] for l in labels)
    total_fp = sum(per_class[l]["false_positives"] for l in labels)
    total_fn = sum(per_class[l]["false_negatives"] for l in labels)
    micro_precision, micro_recall, micro_f1 = calculate_precision_recall_f1(total_tp, total_fp, total_fn)
    
    metrics["micro_precision"] = micro_precision
    metrics["micro_recall"] = micro_recall
    metrics["micro_f1"] = micro_f1
    
    return metrics


def compute_target_metrics(records: list[dict], labels: list[str]) -> dict[str, Any]:
    """
    Compute metrics broken down by target group.
    Returns accuracy and F1 for each target group (from targets_union).
    """
    # Group records by target
    target_records = defaultdict(list)
    
    for rec in records:
        gold = rec.get("gold_label")
        pred = rec.get("pred_label")
        targets = rec.get("targets_union", [])
        
        # Skip if not valid labels
        if gold not in labels or pred not in labels:
            continue
        
        # Add to each target group this record belongs to
        if not targets:
            target_records["No_Target"].append((gold, pred))
        else:
            for target in targets:
                target_records[target].append((gold, pred))
    
    # Compute metrics per target
    target_metrics = {}
    
    for target, pairs in sorted(target_records.items()):
        gold_list = [g for g, p in pairs]
        pred_list = [p for g, p in pairs]
        
        conf = compute_confusion_matrix(gold_list, pred_list, labels)
        metrics = compute_metrics_from_confusion(conf, labels)
        
        target_metrics[target] = {
            "count": len(pairs),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class": metrics["per_class"],
            "confusion_matrix": conf,
        }
    
    return target_metrics


def compute_error_analysis(records: list[dict], error_records: list[dict], labels: list[str]) -> dict[str, Any]:
    """Compute error analysis statistics."""
    analysis = {}
    
    # Count prediction types
    pred_counts = defaultdict(int)
    for rec in records:
        pred_counts[rec.get("pred_label", "Unknown")] += 1
    
    analysis["prediction_distribution"] = dict(pred_counts)
    analysis["unknown_predictions"] = pred_counts.get("Unknown", 0)
    analysis["error_records_count"] = len(error_records)
    
    # Misclassification analysis
    misclassifications = defaultdict(int)
    for rec in records:
        gold = rec.get("gold_label")
        pred = rec.get("pred_label")
        if gold in labels and pred in labels and gold != pred:
            misclassifications[f"{gold}_as_{pred}"] += 1
    
    analysis["misclassification_types"] = dict(misclassifications)
    
    # Target groups with highest error rates
    target_errors = defaultdict(lambda: {"total": 0, "errors": 0})
    for rec in records:
        gold = rec.get("gold_label")
        pred = rec.get("pred_label")
        targets = rec.get("targets_union", [])
        
        if gold not in labels or pred not in labels:
            continue
        
        target_list = targets if targets else ["No_Target"]
        for target in target_list:
            target_errors[target]["total"] += 1
            if gold != pred:
                target_errors[target]["errors"] += 1
    
    # Calculate error rates
    target_error_rates = {}
    for target, counts in target_errors.items():
        if counts["total"] > 0:
            target_error_rates[target] = {
                "total": counts["total"],
                "errors": counts["errors"],
                "error_rate": counts["errors"] / counts["total"],
            }
    
    # Sort by error rate descending
    analysis["target_error_rates"] = dict(
        sorted(target_error_rates.items(), key=lambda x: x[1]["error_rate"], reverse=True)
    )
    
    return analysis


def print_metrics(metrics: dict[str, Any], target_metrics: dict[str, Any], analysis: dict[str, Any]) -> None:
    """Print metrics in a readable format."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS REPORT")
    print("=" * 70)
    
    print(f"\n--- Overall Metrics ---")
    print(f"Total examples:    {metrics['total']}")
    print(f"Correct:           {metrics['correct']}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    
    print(f"\n--- Macro Averages ---")
    print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    
    print(f"\n--- Weighted Averages ---")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
    
    print(f"\n--- Per-Class Metrics ---")
    for label, class_metrics in metrics["per_class"].items():
        print(f"\n  {label}:")
        print(f"    Support:     {class_metrics['support']}")
        print(f"    Precision:   {class_metrics['precision']:.4f}")
        print(f"    Recall:      {class_metrics['recall']:.4f}")
        print(f"    F1:          {class_metrics['f1']:.4f}")
    
    print(f"\n--- Confusion Matrix (rows=gold, cols=pred) ---")
    labels = list(metrics["per_class"].keys())
    print("           " + "  ".join(f"{l:>10}" for l in labels))
    for g in labels:
        row = [str(metrics["confusion_matrix"][g][p]) for p in labels]
        print(f"{g:>10} " + "  ".join(f"{v:>10}" for v in row))
    
    print(f"\n--- Target-Distributed Metrics ---")
    print(f"{'Target':<25} {'Count':>8} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 70)
    for target, tmetrics in sorted(target_metrics.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"{target:<25} {tmetrics['count']:>8} {tmetrics['accuracy']:>10.4f} {tmetrics['macro_f1']:>10.4f} {tmetrics['weighted_f1']:>12.4f}")
    
    print(f"\n--- Error Analysis ---")
    print(f"Unknown predictions:     {analysis['unknown_predictions']}")
    print(f"Error records (JSONL):   {analysis['error_records_count']}")
    
    print(f"\nPrediction Distribution:")
    for pred, count in sorted(analysis["prediction_distribution"].items()):
        print(f"  {pred}: {count}")
    
    print(f"\nMisclassification Types:")
    for mistype, count in sorted(analysis["misclassification_types"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {mistype}: {count}")
    
    print(f"\nTop 10 Target Groups by Error Rate:")
    for i, (target, stats) in enumerate(list(analysis["target_error_rates"].items())[:10]):
        print(f"  {target}: {stats['errors']}/{stats['total']} ({stats['error_rate']:.4f})")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from HateXplain classification results")
    parser.add_argument("--results", "-r", default=DEFAULT_RESULTS, help="Path to results JSONL file")
    parser.add_argument("--errors", "-e", default=DEFAULT_ERRORS, help="Path to errors JSONL file")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Path to output metrics JSON file")
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results}")
    records = load_jsonl(args.results)
    print(f"Loaded {len(records)} records")
    
    print(f"Loading errors from: {args.errors}")
    error_records = load_jsonl(args.errors)
    print(f"Loaded {len(error_records)} error records")
    
    if not records:
        print("No records to process. Exiting.")
        return
    
    # Filter to valid predictions only
    valid_records = [
        rec for rec in records
        if rec.get("gold_label") in LABELS and rec.get("pred_label") in LABELS
    ]
    print(f"Valid records (Hateful/Normal only): {len(valid_records)}")
    
    # Extract labels
    gold_labels = [rec["gold_label"] for rec in valid_records]
    pred_labels = [rec["pred_label"] for rec in valid_records]
    
    # Compute confusion matrix
    conf = compute_confusion_matrix(gold_labels, pred_labels, LABELS)
    
    # Compute main metrics
    metrics = compute_metrics_from_confusion(conf, LABELS)
    metrics["confusion_matrix"] = conf
    
    # Compute target-distributed metrics
    target_metrics = compute_target_metrics(valid_records, LABELS)
    
    # Compute error analysis
    analysis = compute_error_analysis(records, error_records, LABELS)
    
    # Print to console
    print_metrics(metrics, target_metrics, analysis)
    
    # Save to JSON
    output = {
        "overall_metrics": metrics,
        "target_metrics": target_metrics,
        "error_analysis": analysis,
        "labels": LABELS,
        "total_records": len(records),
        "valid_records": len(valid_records),
        "error_records": len(error_records),
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved full metrics to: {args.output}")


if __name__ == "__main__":
    main()
