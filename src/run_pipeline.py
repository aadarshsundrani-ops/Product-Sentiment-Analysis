"""
End-to-End Execution Pipeline for Product Review Sentiment Analysis.

Runs the complete workflow:
1. Data loading and validation.
2. Preprocessing, rating binarization, length bucketing, and stratified split.
3. TF-IDF model training (TF-IDF 1-3 n-grams + Logistic Regression).
4. Comprehensive evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix).
5. Subgroup performance audit across categories and review lengths, flagging disparities.
6. Model explainability (global feature coefficients and local LIME explanations).
7. Outputs artifact generation in outputs/.
"""

import os
import sys
import time
import argparse
import pandas as pd

# Ensure package root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

cache_dir = os.path.join(project_root, ".matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir

from src.preprocessing import prepare_dataset, split_data
from src.models import (
    TFIDFModel,
    compute_metrics,
    plot_confusion_matrices,
    save_overall_metrics
)
from src.audit import audit_subgroups
from src.explainability import extract_top_features, run_lime_explanations


def main(
    data_path: str = "data/raw_reviews.csv",
    output_dir: str = "outputs",
    sample_size: int = None
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("      PRODUCT REVIEW SENTIMENT ANALYSIS: END-TO-END NLP PIPELINE      ")
    print("=" * 80)

    # 1. Check Data Availability
    full_data_path = os.path.join(project_root, data_path)
    if not os.path.exists(full_data_path):
        print(f"Dataset not found at {full_data_path}. Running download script...")
        from data.download_data import download_and_prepare_reviews
        download_and_prepare_reviews(output_csv=full_data_path)

    print(f"Loading raw reviews from: {full_data_path}")
    df_raw = pd.read_csv(full_data_path)
    cat_col = "product_category" if "product_category" in df_raw.columns else ("Product_Name" if "Product_Name" in df_raw.columns else None)
    cats = list(df_raw[cat_col].unique()) if cat_col else ["Standard Corpus"]
    print(f"Loaded {len(df_raw)} raw reviews across categories: {cats[:8]}")

    if sample_size and sample_size < len(df_raw):
        print(f"Subsampling to {sample_size} reviews for rapid execution...")
        df_raw = df_raw.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # 2. Dynamic Preprocessing & Labeling
    df_clean = prepare_dataset(df_raw)

    # 3. Stratified Train / Test Split
    df_train, df_test = split_data(df_clean, test_size=0.20, random_state=42)

    # 4. Train TF-IDF Model (TF-IDF 1-3 n-grams + Logistic Regression)
    print("\n" + "=" * 80)
    print("TRAINING TF-IDF MODEL: (Unigrams + Bigrams + Trigrams) + Logistic Regression")
    print("=" * 80)
    tfidf_model = TFIDFModel(max_features=35000, ngram_range=(1, 3), C=1.2, min_df=2, random_state=42)
    tfidf_model.fit(df_train["clean_text"], df_train["label"])

    y_pred = tfidf_model.predict(df_test["clean_text"])
    metrics = compute_metrics(df_test["label"], y_pred)
    print("\nTF-IDF Model Held-Out Test Set Metrics:")
    for k, v in metrics.items():
        print(f"  - {k:<12}: {v:.4f}")

    # 5. Overall Metrics & Confusion Matrix
    print("\n" + "=" * 80)
    print("SAVING EVALUATION METRICS & CONFUSION MATRIX PLOT")
    print("=" * 80)
    cm_path = os.path.join(project_root, output_dir, "confusion_matrix.png")
    plot_confusion_matrices(df_test["label"], y_pred, output_path=cm_path)

    metrics_path = os.path.join(project_root, output_dir, "overall_metrics.txt")
    save_overall_metrics(metrics, output_path=metrics_path)

    # 6. Subgroup Performance Audit
    print("\n" + "=" * 80)
    print("RUNNING SUBGROUP PERFORMANCE AUDIT (Category & Length Buckets)")
    print("=" * 80)
    subgroup_csv = os.path.join(project_root, output_dir, "subgroup_metrics.csv")
    df_subgroups = audit_subgroups(
        df_test=df_test,
        y_pred=y_pred,
        overall_metrics=metrics,
        output_csv=subgroup_csv
    )

    # 7. Explainability (Global Coefficients & Local LIME)
    print("\n" + "=" * 80)
    print("COMPUTING MODEL EXPLAINABILITY")
    print("=" * 80)
    feat_csv = os.path.join(project_root, output_dir, "top_features.csv")
    feat_png = os.path.join(project_root, output_dir, "top_features.png")
    extract_top_features(tfidf_model, top_n=20, output_csv=feat_csv, output_png=feat_png)

    lime_html = os.path.join(project_root, output_dir, "lime_examples.html")
    lime_png = os.path.join(project_root, output_dir, "lime_examples_summary.png")
    run_lime_explanations(tfidf_model, df_test, output_html=lime_html, output_png=lime_png)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE IN {elapsed:.2f}s! All artifacts generated in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End Sentiment Analysis Pipeline")
    parser.add_argument("--data-path", type=str, default="data/raw_reviews.csv", help="Path to raw reviews CSV")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for output artifacts")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional subsample size for fast run")
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size
    )
