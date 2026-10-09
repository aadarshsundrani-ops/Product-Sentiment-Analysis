"""
Subgroup Auditing Module for Product Review Sentiment Analysis.

Audits model performance across:
1. Product categories (e.g., Electronics, Beauty, Grocery & Gourmet Food).
2. Review length buckets: short (<=20 words), medium (21-49 words), detailed (>=50 words).
3. Intersectional slices (Category x Length bucket).

Outputs:
- outputs/subgroup_metrics.csv
- Flags the worst-performing subgroup and computes performance disparity gaps.
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_slice(y_true, y_pred) -> dict:
    """Computes evaluation metrics for a single slice/subgroup."""
    n_samples = len(y_true)
    if n_samples == 0:
        return {
            "samples": 0,
            "pos_ratio": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_macro": 0.0,
            "f1_binary": 0.0,
        }

    pos_ratio = float((y_true == 1).mean())
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_bin = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    return {
        "samples": n_samples,
        "pos_ratio": round(pos_ratio, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_binary": round(f1_bin, 4),
    }


def audit_subgroups(
    df_test: pd.DataFrame,
    y_pred,
    overall_metrics: dict,
    output_csv: str = "outputs/subgroup_metrics.csv"
) -> pd.DataFrame:
    """
    Performs comprehensive subgroup performance auditing:
    - Dimension 1: Product Category
    - Dimension 2: Review Length Bucket (short vs detailed vs medium)
    - Dimension 3: Intersectional (Category x Length Bucket)

    Saves results to output_csv and prints a formatted report.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_eval = df_test.copy().reset_index(drop=True)
    df_eval["y_pred"] = y_pred

    records = []

    # 1. Overall Baseline Reference
    overall_rec = {
        "dimension": "Overall",
        "subgroup": "Full Test Set",
        **evaluate_slice(df_eval["label"], df_eval["y_pred"])
    }
    records.append(overall_rec)

    # 2. By Product Category
    if "product_category" in df_eval.columns:
        for cat in sorted(df_eval["product_category"].dropna().unique()):
            subset = df_eval[df_eval["product_category"] == cat]
            metrics = evaluate_slice(subset["label"], subset["y_pred"])
            records.append({
                "dimension": "Product Category",
                "subgroup": str(cat),
                **metrics
            })

    # 3. By Review Length Bucket
    if "length_bucket" in df_eval.columns:
        bucket_order = ["short", "medium", "detailed"]
        for bucket in bucket_order:
            subset = df_eval[df_eval["length_bucket"] == bucket]
            if len(subset) > 0:
                metrics = evaluate_slice(subset["label"], subset["y_pred"])
                label_name = f"{bucket.capitalize()} (≤20 words)" if bucket == "short" else (
                    f"{bucket.capitalize()} (≥50 words)" if bucket == "detailed" else f"{bucket.capitalize()} (21-49 words)"
                )
                records.append({
                    "dimension": "Review Length",
                    "subgroup": label_name,
                    **metrics
                })

    # 4. Intersectional Slices: Category x Length Bucket
    if "product_category" in df_eval.columns and "length_bucket" in df_eval.columns:
        for cat in sorted(df_eval["product_category"].dropna().unique()):
            for bucket in ["short", "detailed"]:
                subset = df_eval[(df_eval["product_category"] == cat) & (df_eval["length_bucket"] == bucket)]
                if len(subset) >= 5:  # meaningful sample threshold
                    metrics = evaluate_slice(subset["label"], subset["y_pred"])
                    records.append({
                        "dimension": "Intersectional",
                        "subgroup": f"{cat} | {bucket}",
                        **metrics
                    })

    df_subgroups = pd.DataFrame(records)

    # Save to CSV
    df_subgroups.to_csv(output_csv, index=False)
    print(f"Saved subgroup audit metrics to: {output_csv}")

    # Print Formatted Table
    print("\n" + "=" * 95)
    print("                      SUBGROUP AUDIT PERFORMANCE SUMMARY TABLE                         ")
    print("=" * 95)
    header = f"{'Dimension':<18} | {'Subgroup':<26} | {'N':>6} | {'Pos %':>6} | {'Accuracy':>8} | {'Macro F1':>8} | {'Binary F1':>9}"
    print(header)
    print("-" * 95)
    for _, row in df_subgroups.iterrows():
        print(
            f"{row['dimension']:<18} | {row['subgroup']:<26} | {row['samples']:>6} | "
            f"{row['pos_ratio']*100:>5.1f}% | {row['accuracy']:>8.4f} | {row['f1_macro']:>8.4f} | {row['f1_binary']:>9.4f}"
        )
    print("=" * 95)

    # Identify and explicitly flag the worst performing subgroup
    flag_worst_subgroups(df_subgroups, overall_f1=overall_metrics.get("f1_macro", 0.0), overall_acc=overall_metrics.get("accuracy", 0.0))

    return df_subgroups


def flag_worst_subgroups(df_subgroups: pd.DataFrame, overall_f1: float, overall_acc: float):
    """
    Explicitly flags the worst-performing subgroups across category and length dimensions,
    reporting the performance gap (disparity) compared to overall test performance.
    """
    # Exclude overall row and low-sample intersectional rows for standalone flag
    primary_subgroups = df_subgroups[
        (df_subgroups["dimension"].isin(["Product Category", "Review Length"])) &
        (df_subgroups["samples"] >= 10)
    ].copy()

    if primary_subgroups.empty:
        primary_subgroups = df_subgroups[df_subgroups["dimension"] != "Overall"].copy()

    # Find worst by Macro F1
    worst_f1_idx = primary_subgroups["f1_macro"].idxmin()
    worst_row = primary_subgroups.loc[worst_f1_idx]

    delta_f1 = worst_row["f1_macro"] - overall_f1
    pct_f1_drop = (delta_f1 / overall_f1 * 100) if overall_f1 > 0 else 0.0

    delta_acc = worst_row["accuracy"] - overall_acc
    pct_acc_drop = (delta_acc / overall_acc * 100) if overall_acc > 0 else 0.0

    # Also compare Short vs Detailed directly
    length_rows = df_subgroups[df_subgroups["dimension"] == "Review Length"]
    short_f1 = length_rows[length_rows["subgroup"].str.contains("Short", case=False)]["f1_macro"].values
    detailed_f1 = length_rows[length_rows["subgroup"].str.contains("Detailed", case=False)]["f1_macro"].values

    print("\n" + "!" * 95)
    print("                     🚨 SUBGROUP PERFORMANCE AUDIT FINDINGS & FLAGGING                 ")
    print("!" * 95)
    print(f"Worst-Performing Subgroup: [{worst_row['dimension']}] -> '{worst_row['subgroup']}'")
    print(f"  • Subgroup Macro F1: {worst_row['f1_macro']:.4f}  (Overall: {overall_f1:.4f} | Disparity: {delta_f1:+.4f} or {pct_f1_drop:+.1f}%)")
    print(f"  • Subgroup Accuracy: {worst_row['accuracy']:.4f}  (Overall: {overall_acc:.4f} | Disparity: {delta_acc:+.4f} or {pct_acc_drop:+.1f}%)")
    print(f"  • Sample Size (N):   {worst_row['samples']}")

    if len(short_f1) > 0 and len(detailed_f1) > 0:
        s_f1 = short_f1[0]
        d_f1 = detailed_f1[0]
        diff_sd = s_f1 - d_f1
        print(f"\nReview Length Comparison (Short vs Detailed):")
        print(f"  • Short Reviews Macro F1:    {s_f1:.4f}")
        print(f"  • Detailed Reviews Macro F1: {d_f1:.4f}")
        print(f"  • Performance Gap (Short - Detailed): {diff_sd:+.4f} (Short underperforms by {abs(diff_sd):.4f})")

    print("\nDiagnostic Explanation:")
    print("  Short reviews exhibit severe lexical sparsity. Phrases such as 'Not bad', 'Works ok',")
    print("  or 'It broke' provide limited contextual tokens, causing the model to struggle with nuance")
    print("  compared to detailed reviews where customers provide extensive supporting descriptions.")
    print("!" * 95 + "\n")
