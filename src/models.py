"""
Models and Evaluation module for Product Review Sentiment Analysis.

Includes:
1. TFIDFModel: High-accuracy TF-IDF (unigrams + bigrams + trigrams) + Logistic Regression.
2. Evaluation utilities: metrics calculation, confusion matrix plotting,
   and formatted metrics reporting.
"""

import os
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Domain-agnostic stop words: standard English functional stop words
# with strict preservation of negation tokens and sentiment modifiers.
DOMAIN_AGNOSTIC_STOP_WORDS = list(set(ENGLISH_STOP_WORDS) - {
    "not", "no", "never", "neither", "nor", "none", "very", "too", "against",
    "cannot", "cant", "wont", "dont", "hardly", "barely", "scarcely", "nothing", "nobody"
})


class TFIDFModel:
    """
    High-accuracy, dataset-friendly TF-IDF (unigrams + bigrams + trigrams) + Logistic Regression Model.
    Engineered for robust cross-domain generalization across any product review dataset.
    """
    def __init__(
        self,
        max_features: int = 35000,
        ngram_range: tuple = (1, 3),
        C: float = 1.2,
        min_df: int = 2,
        random_state: int = 42
    ):
        self.vectorizer = TfidfVectorizer(
            stop_words=DOMAIN_AGNOSTIC_STOP_WORDS,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=min_df
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, texts, labels):
        print(f"Fitting TF-IDF Vectorizer on {len(texts)} texts...")
        X_vec = self.vectorizer.fit_transform(texts)
        print("Training Logistic Regression Classifier...")
        self.classifier.fit(X_vec, labels)
        self.is_fitted = True
        return self

    def predict(self, texts, return_three_class: bool = False, low_th: float = 0.38, high_th: float = 0.58):
        X_vec = self.vectorizer.transform(texts)
        classes = list(self.classifier.classes_)

        # If already multi-class model with Positive, Negative, Neutral
        if "Positive" in classes and ("Neutral" in classes or "Negative" in classes):
            probs = self.classifier.predict_proba(X_vec)
            pos_idx = classes.index("Positive")
            neg_idx = classes.index("Negative")
            neu_idx = classes.index("Neutral") if "Neutral" in classes else None

            preds = []
            for i, p in enumerate(probs):
                pos_p = p[pos_idx]
                neg_p = p[neg_idx]
                neu_p = p[neu_idx] if neu_idx is not None else 0.0

                raw_t = str(texts[i]).lower() if i < len(texts) else ""
                has_contrast = any(w in raw_t for w in [
                    " but ", " however ", " although ", " though ", "decent but",
                    "okay for", "average", "mediocre", "neither ", "nor "
                ])

                if neu_idx is not None and (
                    (has_contrast and abs(pos_p - neg_p) < 0.35) or
                    (neu_p > 0.33 and abs(pos_p - neg_p) < 0.25)
                ):
                    preds.append("Neutral")
                elif pos_p > neg_p and pos_p > neu_p:
                    preds.append("Positive")
                elif neg_p > pos_p and neg_p > neu_p:
                    preds.append("Negative")
                else:
                    preds.append(str(classes[int(np.argmax(p))]).capitalize())

            return np.array(preds)

        # If binary model but caller requested 3-class thresholding
        if return_three_class:
            probs = self.classifier.predict_proba(X_vec)
            preds = []
            for p in probs:
                p1 = p[1] if len(p) > 1 else p[0]
                if p1 >= high_th:
                    preds.append("Positive")
                elif p1 <= low_th:
                    preds.append("Negative")
                else:
                    preds.append("Neutral")
            return np.array(preds)

        raw_preds = self.classifier.predict(X_vec)
        return raw_preds

    def predict_proba(self, texts):
        X_vec = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X_vec)

    def get_feature_coefficients(self):
        """Returns vocabulary feature names and their directional polarity coefficients."""
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        classes = list(self.classifier.classes_)

        if len(classes) > 2 and "Positive" in classes and "Negative" in classes:
            pos_idx = classes.index("Positive")
            neg_idx = classes.index("Negative")
            coefs = self.classifier.coef_[pos_idx] - self.classifier.coef_[neg_idx]
        elif len(self.classifier.coef_) > 1:
            coefs = self.classifier.coef_[0] - self.classifier.coef_[1]
        else:
            coefs = self.classifier.coef_[0]
        return feature_names, coefs

    def analyze_tokens(self, text: str) -> dict:
        """
        Extracts all vocabulary tokens from the review text with their TF, IDF,
        TF-IDF vector score, logistic regression weight, and net polarity impact.
        """
        X_vec = self.vectorizer.transform([text])
        feature_names, coefs = self.get_feature_coefficients()
        indices = X_vec.indices
        tfidf_data = X_vec.data

        token_rows = []
        words = text.lower().split()

        for idx, tfidf_val in zip(indices, tfidf_data):
            term = str(feature_names[idx])
            idf = float(self.vectorizer.idf_[idx])
            coef = float(coefs[idx])

            # Count raw term frequency in text
            if " " in term:
                raw_tf = text.lower().count(term)
            else:
                raw_tf = words.count(term)
            if raw_tf < 1:
                raw_tf = 1

            contribution = float(coef * tfidf_val)
            influence = "Positive" if contribution > 0 else ("Negative" if contribution < 0 else "Neutral")

            token_rows.append({
                "token": term,
                "tf": raw_tf,
                "idf": round(idf, 4),
                "tfidf_score": round(float(tfidf_val), 4),
                "coefficient": round(coef, 4),
                "sentiment_contribution": round(contribution, 4),
                "influence": influence,
                "is_ngram": " " in term
            })

        # Sort by absolute contribution descending
        token_rows.sort(key=lambda x: abs(x["sentiment_contribution"]), reverse=True)

        total_pos = sum(r["sentiment_contribution"] for r in token_rows if r["sentiment_contribution"] > 0)
        total_neg = sum(r["sentiment_contribution"] for r in token_rows if r["sentiment_contribution"] < 0)
        intercept = float(self.classifier.intercept_[0]) if hasattr(self.classifier, "intercept_") else 0.0

        return {
            "model_type": "TF-IDF + Logistic Regression",
            "tokens": token_rows,
            "total_tokens_found": len(token_rows),
            "intercept": round(intercept, 4),
            "total_positive_weight": round(total_pos, 4),
            "total_negative_weight": round(total_neg, 4),
            "net_logit": round(intercept + total_pos + total_neg, 4)
        }


# Aliases for backwards compatibility
BaselineModel = TFIDFModel
AdvancedModel = TFIDFModel


def compute_metrics(y_true, y_pred) -> dict:
    """Computes standard classification evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_confusion_matrices(
    y_true,
    y_pred,
    y_pred_unused=None,
    output_path: str = "outputs/confusion_matrix.png"
):
    """Plots confusion matrix heatmap for the TF-IDF model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("TF-IDF + Logistic Regression Confusion Matrix", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {output_path}")


def save_overall_metrics(
    metrics_primary: dict,
    metrics_unused=None,
    output_path: str = "outputs/overall_metrics.txt"
):
    """Saves and prints overall metrics for the TF-IDF model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header = f"{'Metric':<22} | {'TF-IDF Model Value':<20}"
    divider = "-" * len(header)

    lines = [
        "==================================================",
        "          TF-IDF MODEL PERFORMANCE SUMMARY        ",
        "==================================================",
        header,
        divider
    ]

    for key, val in metrics_primary.items():
        if isinstance(val, (int, float)):
            lines.append(f"{key.replace('_', ' ').title():<22} | {val:<20.4f}")

    lines.append(divider)
    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content + "\n")

    print("\n" + content + "\n")
    print(f"Saved overall metrics to: {output_path}")
