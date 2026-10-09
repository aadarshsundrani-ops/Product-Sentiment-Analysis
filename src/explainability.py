"""
Explainability Module for Product Review Sentiment Analysis.

Provides:
1. Global Explainability: Top 20 positive and negative feature coefficients from
   TF-IDF + Logistic Regression (saved to CSV & PNG bar chart).
2. Local Explainability: LIME (Local Interpretable Model-agnostic Explanations)
   on representative test reviews, saved as interactive HTML and visual plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer


def extract_top_features(
    baseline_model,
    top_n: int = 20,
    output_csv: str = "outputs/top_features.csv",
    output_png: str = "outputs/top_features.png"
) -> pd.DataFrame:
    """
    Extracts top N positive and top N negative coefficients from the Logistic Regression model,
    saves them to CSV, and generates a visualization plot.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    feature_names, coefs = baseline_model.get_feature_coefficients()

    # Sort indices
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    records = []
    for rank, idx in enumerate(top_pos_idx, 1):
        records.append({
            "rank": rank,
            "feature": feature_names[idx],
            "coefficient": round(float(coefs[idx]), 4),
            "sentiment": "Positive"
        })
    for rank, idx in enumerate(top_neg_idx, 1):
        records.append({
            "rank": rank,
            "feature": feature_names[idx],
            "coefficient": round(float(coefs[idx]), 4),
            "sentiment": "Negative"
        })

    df_features = pd.DataFrame(records)
    df_features.to_csv(output_csv, index=False)
    print(f"Saved top {top_n} positive and negative features to: {output_csv}")

    # Plot top features
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Top Positive
    pos_data = df_features[df_features["sentiment"] == "Positive"].sort_values("coefficient", ascending=True)
    axes[0].barh(pos_data["feature"], pos_data["coefficient"], color="#2ca02c", edgecolor="#1b611b", height=0.7)
    axes[0].set_title(f"Top {top_n} Positive-Weighted Words", fontsize=13, weight="bold", pad=12)
    axes[0].set_xlabel("Logistic Regression Coefficient (Log-Odds)", fontsize=11)
    axes[0].grid(axis="x", linestyle="--", alpha=0.6)

    # Top Negative
    neg_data = df_features[df_features["sentiment"] == "Negative"].sort_values("coefficient", ascending=False)
    axes[1].barh(neg_data["feature"], neg_data["coefficient"], color="#d62728", edgecolor="#8c1819", height=0.7)
    axes[1].set_title(f"Top {top_n} Negative-Weighted Words", fontsize=13, weight="bold", pad=12)
    axes[1].set_xlabel("Logistic Regression Coefficient (Log-Odds)", fontsize=11)
    axes[1].grid(axis="x", linestyle="--", alpha=0.6)

    plt.suptitle("TF-IDF Feature Importance: Most Influential Sentiment Words", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved top features plot to: {output_png}")

    return df_features


def run_lime_explanations(
    baseline_model,
    df_test: pd.DataFrame,
    output_html: str = "outputs/lime_examples.html",
    output_png: str = "outputs/lime_examples_summary.png"
):
    """
    Runs LIME text explanations on 4 carefully selected test reviews:
    1. Clear True Positive
    2. Clear True Negative
    3. Review with explicit Negation
    4. Challenging/Misclassified review (Error Analysis)

    Exports an interactive HTML file and a static summary visual plot.
    """
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    explainer = LimeTextExplainer(
        class_names=["Negative", "Positive"],
        random_state=42
    )

    df_eval = df_test.copy().reset_index(drop=True)
    y_preds = baseline_model.predict(df_eval["clean_text"])
    y_probs = baseline_model.predict_proba(df_eval["clean_text"])
    df_eval["y_pred"] = y_preds
    df_eval["prob_pos"] = y_probs[:, 1]

    # Select representative samples
    selected_indices = []

    # 1. High-confidence True Positive
    tp_pool = df_eval[(df_eval["label"] == 1) & (df_eval["y_pred"] == 1) & (df_eval["prob_pos"] > 0.85)]
    idx_tp = tp_pool.index[0] if not tp_pool.empty else 0
    selected_indices.append((idx_tp, "True Positive (High Confidence)"))

    # 2. High-confidence True Negative
    tn_pool = df_eval[(df_eval["label"] == 0) & (df_eval["y_pred"] == 0) & (df_eval["prob_pos"] < 0.15)]
    idx_tn = tn_pool.index[0] if not tn_pool.empty else 1
    selected_indices.append((idx_tn, "True Negative (High Confidence)"))

    # 3. Negated Review (contains 'not' or 'no' or 'never')
    neg_word_pool = df_eval[
        df_eval["clean_text"].str.contains(r"\b(not|no|never|neither)\b", regex=True) &
        (~df_eval.index.isin([idx_tp, idx_tn]))
    ]
    idx_neg = neg_word_pool.index[0] if not neg_word_pool.empty else 2
    selected_indices.append((idx_neg, "Review with Negation Syntax"))

    # 4. Misclassified instance (Error Analysis)
    err_pool = df_eval[
        (df_eval["label"] != df_eval["y_pred"]) &
        (~df_eval.index.isin([idx_tp, idx_tn, idx_neg]))
    ]
    if not err_pool.empty:
        idx_err = err_pool.index[0]
        err_type = "False Positive" if df_eval.loc[idx_err, "y_pred"] == 1 else "False Negative"
        selected_indices.append((idx_err, f"Misclassified Review ({err_type})"))
    else:
        # Fallback to a detailed review
        det_pool = df_eval[df_eval["length_bucket"] == "detailed"]
        idx_err = det_pool.index[0] if not det_pool.empty else 3
        selected_indices.append((idx_err, "Detailed Multi-Aspect Review"))

    print(f"\nGenerating LIME explanations for {len(selected_indices)} representative test reviews...")

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>LIME Sentiment Explanations</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #212529; }",
        "h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }",
        ".example-card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 35px; padding: 24px; }",
        ".meta-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: 600; margin-right: 8px; margin-bottom: 12px; }",
        ".badge-tp { background: #e6f4ea; color: #137333; }",
        ".badge-tn { background: #fce8e6; color: #c5221f; }",
        ".badge-info { background: #e8f0fe; color: #1a73e8; }",
        ".review-box { background: #f1f3f4; padding: 14px; border-left: 4px solid #1a73e8; border-radius: 4px; font-style: italic; margin: 12px 0 20px 0; }",
        "</style></head><body>",
        "<h1>LIME Local Interpretability Analysis</h1>",
        "<p>This report presents instance-level sentiment explanations generated via <b>LIME (Local Interpretable Model-agnostic Explanations)</b> on held-out test reviews.</p>"
    ]

    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(10, 3.5 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = [axes]

    for plot_idx, (sample_idx, case_title) in enumerate(selected_indices):
        sample = df_eval.loc[sample_idx]
        text = sample["clean_text"]
        true_label = "Positive (1)" if sample["label"] == 1 else "Negative (0)"
        pred_label = "Positive (1)" if sample["y_pred"] == 1 else "Negative (0)"
        prob_pos = sample["prob_pos"]
        cat = sample.get("product_category", "N/A")
        bucket = sample.get("length_bucket", "N/A")

        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=baseline_model.predict_proba,
            num_features=8
        )

        exp_html = exp.as_html()
        badge_class = "badge-tp" if sample["y_pred"] == 1 else "badge-tn"

        card_html = f"""
        <div class='example-card'>
            <h2>Example {plot_idx + 1}: {case_title}</h2>
            <div>
                <span class='meta-badge {badge_class}'>Predicted: {pred_label} (P(Pos)={prob_pos:.3f})</span>
                <span class='meta-badge badge-info'>Ground Truth: {true_label}</span>
                <span class='meta-badge badge-info'>Category: {cat}</span>
                <span class='meta-badge badge-info'>Length: {bucket} ({sample.get('word_count', 0)} words)</span>
            </div>
            <div class='review-box'>"{text}"</div>
            {exp_html}
        </div>
        """
        html_parts.append(card_html)

        # Plot local token contributions for static PNG
        token_weights = exp.as_list()
        tokens = [t[0] for t in token_weights][::-1]
        weights = [t[1] for t in token_weights][::-1]
        colors = ["#2ca02c" if w > 0 else "#d62728" for w in weights]

        ax = axes[plot_idx]
        ax.barh(tokens, weights, color=colors, height=0.6)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"Ex {plot_idx+1}: {case_title} | True: {true_label} | Pred: {pred_label} (P(Pos)={prob_pos:.2f})",
            fontsize=11, weight="bold", pad=8
        )
        ax.set_xlabel("LIME Token Weight (Contribution to Positive Sentiment)", fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

    html_parts.append("</body></html>")

    # Save full HTML
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Saved interactive LIME HTML explanations to: {output_html}")

    # Save static figure
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved LIME summary visualization to: {output_png}")
