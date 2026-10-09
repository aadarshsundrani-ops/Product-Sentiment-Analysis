"""
End-to-End Sentiment & TF-IDF Dataset Analysis Pipeline
======================================================
This pipeline allows users to upload/load any product review dataset and:
1. Dynamically detects columns (Review_ID, Review_Text, Rating, Sentiments).
2. Cleans text, expands contractions, preserves negation n-grams and tone shifts.
3. Computes sublinear TF-IDF (unigrams & bigrams) across the dataset.
4. Performs 3-class sentiment analysis (Positive, Neutral, Negative) with calibrated probabilities.
5. Provides dataset-level feedback analytics (class balance, top praise terms, top critique terms, sliced metrics).
6. Provides per-feedback TF-IDF token attribution (Token, TF, IDF, TF-IDF score, coefficient, polarity influence).
7. Exports enriched datasets to CSV/JSON and offers an interactive CLI lookup by Review ID.
"""

import os
import sys
import re
import html
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Determine project directories safely (works in scripts, IPython, and Google Colab)
if "__file__" in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Contraction expansion map
CONTRACTION_MAP = {
    r"won\'t": "will not",
    r"can\'t": "can not",
    r"cannot": "can not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
    r"aint": "is not",
    r"arent": "are not",
    r"cant": "can not",
    r"couldnt": "could not",
    r"didnt": "did not",
    r"doesnt": "does not",
    r"dont": "do not",
    r"hadnt": "had not",
    r"hasnt": "has not",
    r"isnt": "is not",
    r"shouldnt": "should not",
    r"wasnt": "was not",
    r"werent": "were not",
    r"wont": "will not",
    r"wouldnt": "would not",
}

# Domain sentiment anchor lexicons for uncalibrated fallback
DOMAIN_POSITIVE = {
    "great": 2.2, "love": 2.1, "excellent": 2.0, "perfect": 1.9, "comfortable": 1.8,
    "best": 1.7, "five stars": 2.5, "won me over": 2.3, "soothing": 1.6, "hydrating": 1.6,
    "nice": 1.3, "good": 1.2, "happy": 1.4, "recommend": 1.5, "smooth": 1.3,
    "soft": 1.2, "durable": 1.4, "wonderful": 1.8, "amazing": 1.9, "fantastic": 1.8
}
DOMAIN_NEGATIVE = {
    "not": -1.8, "terrible": -2.4, "horrible": -2.4, "disappointed": -2.2, "waste": -2.3,
    "broken": -2.0, "cheap": -1.7, "poor": -1.8, "one star": -2.5, "awful": -2.2,
    "worst": -2.4, "defective": -2.1, "bad": -1.5, "odor": -1.7, "ugly": -1.4,
    "returned": -1.6, "returning": -1.6, "refund": -1.8, "broke": -1.9, "waste of money": -2.5
}


try:
    from src.preprocessing import clean_text, CONTRACTION_MAP, NEGATION_WORDS
except ImportError:
    try:
        from preprocessing import clean_text, CONTRACTION_MAP, NEGATION_WORDS
    except ImportError:
        NEGATION_WORDS = {"not", "never", "no", "hardly", "barely", "scarcely"}
        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"https?://\S+|www\.\S+", " ", text).lower()
            for pattern, replacement in CONTRACTION_MAP.items():
                text = re.sub(pattern, replacement, text)
            text = re.sub(r'\b(5\s*stars?|five\s*stars?|5/5|5\s*out\s*of\s*5|5\s*star)\b', ' explicit_five_star explicit_five_star great ', text)
            text = re.sub(r'\b(4\s*stars?|four\s*stars?|4/5|4\s*out\s*of\s*5|4\s*star)\b', ' explicit_four_star good ', text)
            text = re.sub(r'\b(1\s*stars?|one\s*stars?|1/5|1\s*out\s*of\s*5|zero\s*stars?|0\s*stars?|1\s*star)\b', ' explicit_one_star explicit_one_star terrible ', text)
            text = re.sub(r'\bwon\s+(?:me|us|everyone|them)\s+over\b', ' won_me_over love_it great ', text)
            text = re.sub(r'\b(?:not|was\s+not)\s+a\s+fan\s+of\b', ' not_fan_of not ', text)
            contrast_parts = re.split(r'\b(but|however|yet|although|nevertheless|though)\b', text)
            if len(contrast_parts) > 1:
                pre = contrast_parts[0]
                post = ' '.join(contrast_parts[1:])
                text = f'{pre} {post} {post}'
            text = re.sub(r'[^a-z0-9_\s]', ' ', text)
            return re.sub(r'\s+', ' ', text).strip()


class SentimentTFIDFPipeline:
    """
    Modular, End-to-End Review Sentiment and TF-IDF Pipeline.
    
    Stages:
    1. Ingestion: load_dataset()
    2. Token Preprocessing: preprocess()
    3. TF-IDF Modeling: fit_transform_tfidf()
    4. 3-Class Sentiment Prediction: analyze_sentiment()
    5. Per-review token feedback: get_review_feedback()
    6. Reporting & Export: get_dataset_summary(), export_results()
    """

    def __init__(self, model_path: Optional[str] = None, max_features: int = 10000):
        self.max_features = max_features
        self.df: pd.DataFrame = pd.DataFrame()
        self.dataset_name: str = "custom_dataset"
        
        # Column mappings
        self.id_col: str = "Review_ID"
        self.text_col: str = "Review_Text"
        self.rating_col: Optional[str] = None
        self.sentiment_col: Optional[str] = None
        self.product_col: Optional[str] = None

        # Models
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.is_fitted: bool = False
        
        # Try loading pre-trained baseline model if available
        if model_path is None:
            default_path = os.path.join(PROJECT_ROOT, "outputs", "baseline_model.joblib")
            if os.path.exists(default_path):
                model_path = default_path

        if model_path and os.path.exists(model_path):
            try:
                loaded = joblib.load(model_path)
                if hasattr(loaded, "vectorizer") and hasattr(loaded, "classifier"):
                    self.vectorizer = loaded.vectorizer
                    self.classifier = loaded.classifier
                    self.is_fitted = True
                    print(f"[Pipeline] Loaded pre-trained TF-IDF model from: {model_path}")
            except Exception as e:
                print(f"[Pipeline] Notice: Could not load {model_path} ({e}), initializing fresh TF-IDF engine.")

        if not self.is_fitted:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english"
            )

    def load_dataset(self, source: Union[str, pd.DataFrame, List[Dict[str, Any]]], name: Optional[str] = None) -> "SentimentTFIDFPipeline":
        """
        Loads dataset from a filepath (CSV, TSV, JSON, Excel), pandas DataFrame, or list of dicts.
        Dynamically detects canonical columns.
        """
        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
            self.dataset_name = name or "dataframe_input"
        elif isinstance(source, list):
            self.df = pd.DataFrame(source)
            self.dataset_name = name or "records_input"
        elif isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Dataset file not found: {source}")
            self.dataset_name = name or os.path.basename(source)
            ext = os.path.splitext(source)[1].lower()

            if ext in [".csv", ".txt"]:
                try:
                    self.df = pd.read_csv(source, encoding="utf-8-sig")
                except Exception:
                    self.df = pd.read_csv(source, sep=";", encoding="utf-8-sig")
            elif ext in [".tsv", ".tab"]:
                self.df = pd.read_csv(source, sep="\t", encoding="utf-8-sig")
            elif ext == ".json":
                self.df = pd.read_json(source)
            elif ext in [".xlsx", ".xls"]:
                self.df = pd.read_excel(source)
            else:
                self.df = pd.read_csv(source)
        else:
            raise ValueError(f"Unsupported dataset source type: {type(source)}")

        if self.df.empty:
            raise ValueError("The provided dataset is empty.")

        self._discover_columns()
        print(f"[Pipeline] Ingested {len(self.df)} records from '{self.dataset_name}'.")
        print(f"           Columns mapped: ID='{self.id_col}', Text='{self.text_col}', Rating='{self.rating_col}', Sentiments='{self.sentiment_col}'")
        return self

    def _discover_columns(self):
        """Discovers canonical column names dynamically from raw headers."""
        cols = list(self.df.columns)
        
        def match_col(candidates: List[str]) -> Optional[str]:
            for cand in candidates:
                for c in cols:
                    if str(c).strip().lower() == cand.lower():
                        return c
            for cand in candidates:
                for c in cols:
                    clean_c = re.sub(r"[^a-z0-9]", "", str(c).lower())
                    clean_cand = re.sub(r"[^a-z0-9]", "", cand.lower())
                    if clean_c == clean_cand:
                        return c
            return None

        # Text column
        text_match = match_col(["Review_Text", "review_text", "review_body", "text", "Review", "comment", "feedback", "body", "content"])
        self.text_col = text_match if text_match else (cols[1] if len(cols) > 1 else cols[0])

        # ID column
        id_match = match_col(["Review_ID", "review_id", "ReviewID", "id", "ID", "Review_No", "Index", "sr_no", "code", "unique_id"])
        if id_match:
            self.id_col = id_match
        else:
            if cols[0] != self.text_col:
                self.id_col = cols[0]
            else:
                self.id_col = "Review_ID"
                self.df["Review_ID"] = [f"REV_{i+1:05d}" for i in range(len(self.df))]

        # Rating & Sentiment columns
        self.rating_col = match_col(["Rating", "stars", "rating", "score", "star_rating", "ratings"])
        self.sentiment_col = match_col(["Sentiments", "sentiment", "Sentiment", "label", "polarity", "class"])
        self.product_col = match_col(["Product_Name", "product_name", "product_category", "product", "item_name", "category"])

        # Ensure Review_ID is string
        self.df[self.id_col] = self.df[self.id_col].astype(str)

    def preprocess(self) -> "SentimentTFIDFPipeline":
        """Preprocesses all feedback texts and assigns length buckets."""
        if self.df.empty:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("[Pipeline] Running lexical normalization & token preprocessing...")
        self.df["clean_text"] = self.df[self.text_col].apply(clean_text)
        
        self.df["word_count"] = self.df["clean_text"].apply(lambda t: len(t.split()))
        self.df["length_bucket"] = self.df["word_count"].apply(
            lambda w: "Short (<=20 words)" if w <= 20 else ("Medium (21-49 words)" if w <= 49 else "Detailed (>=50 words)")
        )
        return self

    def fit_transform_tfidf(self) -> "SentimentTFIDFPipeline":
        """Fits or transforms the TF-IDF engine across the dataset."""
        if "clean_text" not in self.df.columns:
            self.preprocess()

        texts = self.df["clean_text"].tolist()

        if not self.is_fitted:
            print("[Pipeline] Fitting sublinear TF-IDF vectorizer across dataset corpus...")
            X_tfidf = self.vectorizer.fit_transform(texts)
            
            if self.sentiment_col and self.sentiment_col in self.df.columns:
                print(f"[Pipeline] Training classifier using ground truth labels from '{self.sentiment_col}'...")
                y_raw = self.df[self.sentiment_col].astype(str).str.lower()
                y_binary = y_raw.apply(lambda s: 1 if ("pos" in s or s == "1") else 0)
                self.classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                self.classifier.fit(X_tfidf, y_binary)
                self.is_fitted = True
            elif self.rating_col and self.rating_col in self.df.columns:
                print(f"[Pipeline] Training classifier using star ratings from '{self.rating_col}'...")
                y_binary = self.df[self.rating_col].apply(lambda r: 1 if float(r) >= 4 else 0)
                self.classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                self.classifier.fit(X_tfidf, y_binary)
                self.is_fitted = True
            else:
                print("[Pipeline] Fitting TF-IDF matrix (unsupervised / domain calibrated)...")
                self.is_fitted = True
        else:
            print("[Pipeline] Transforming dataset with pre-trained TF-IDF vectorizer...")
            X_tfidf = self.vectorizer.transform(texts)

        return self

    def analyze_sentiment(self) -> "SentimentTFIDFPipeline":
        """
        Executes sentiment analysis across all reviews in the dataset.
        Assigns Predicted_Sentiment, Probability_Positive, Probability_Neutral,
        Probability_Negative, Confidence, and Top_Influential_Tokens.
        """
        if not self.is_fitted:
            self.fit_transform_tfidf()

        texts = self.df["clean_text"].tolist()
        X_tfidf = self.vectorizer.transform(texts)

        predictions = []
        confidences = []
        p_pos_list = []
        p_neu_list = []
        p_neg_list = []
        top_tokens_list = []

        has_model = (self.classifier is not None and hasattr(self.classifier, "predict_proba"))
        feature_names = self.vectorizer.get_feature_names_out()
        if has_model:
            raw_probs = self.classifier.predict_proba(X_tfidf)[:, 1]
            coefs = self.classifier.coef_[0]
            feature_map = dict(zip(feature_names, coefs))
        else:
            raw_probs = None
            feature_map = {**DOMAIN_POSITIVE, **DOMAIN_NEGATIVE}

        for i, text in enumerate(texts):
            # 1. Determine raw probability
            if raw_probs is not None:
                prob = float(raw_probs[i])
            else:
                words = text.split()
                score = sum(feature_map.get(w, 0.0) for w in words)
                prob = 1.0 / (1.0 + np.exp(-score * 0.8))

            # 2. Calibrate 3 classes (Positive, Neutral, Negative)
            if 0.44 <= prob <= 0.56:
                sentiment = "Neutral"
                p_neu = round(0.52 + abs(0.5 - prob) * 0.4, 4)
                p_pos = round((1.0 - p_neu) * (prob / 0.5) * 0.5, 4)
                p_neg = round(1.0 - p_neu - p_pos, 4)
                conf = p_neu
            elif prob > 0.56:
                sentiment = "Positive"
                p_pos = round(prob, 4)
                p_neu = round((1.0 - prob) * 0.35, 4)
                p_neg = round(1.0 - p_pos - p_neu, 4)
                conf = p_pos
            else:
                sentiment = "Negative"
                p_neg = round(1.0 - prob, 4)
                p_neu = round(prob * 0.35, 4)
                p_pos = round(1.0 - p_neg - p_neu, 4)
                conf = p_neg

            # 3. Extract top influential tokens for this feedback
            row_tfidf = X_tfidf[i]
            tokens_for_row = []
            for col_idx, tfidf_val in zip(row_tfidf.indices, row_tfidf.data):
                term = feature_names[col_idx]
                w_coef = float(feature_map.get(term, 0.0))
                impact = round(w_coef * float(tfidf_val), 4)
                display_term = (
                    term.replace("explicit_five_star", "five stars")
                    .replace("explicit_four_star", "four stars")
                    .replace("explicit_one_star", "one star")
                    .replace("won_me_over", "won me over")
                    .replace("not_", "not ")
                )
                tokens_for_row.append((display_term, impact))

            tokens_for_row.sort(key=lambda x: abs(x[1]), reverse=True)
            top_toks_str = ", ".join([f"{t[0]} ({t[1]:+.2f})" for t in tokens_for_row[:5]])

            predictions.append(sentiment)
            confidences.append(conf)
            p_pos_list.append(p_pos)
            p_neu_list.append(p_neu)
            p_neg_list.append(p_neg)
            top_tokens_list.append(top_toks_str)

        self.df["Predicted_Sentiment"] = predictions
        self.df["Confidence"] = confidences
        self.df["P_Positive"] = p_pos_list
        self.df["P_Neutral"] = p_neu_list
        self.df["P_Negative"] = p_neg_list
        self.df["Influential_Tokens"] = top_tokens_list

        return self

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Returns comprehensive aggregated analytics for the entire dataset."""
        if "Predicted_Sentiment" not in self.df.columns:
            self.analyze_sentiment()

        total = len(self.df)
        counts = self.df["Predicted_Sentiment"].value_counts().to_dict()
        pos_cnt = counts.get("Positive", 0)
        neu_cnt = counts.get("Neutral", 0)
        neg_cnt = counts.get("Negative", 0)

        length_breakdown = {}
        for bucket, grp in self.df.groupby("length_bucket"):
            b_counts = grp["Predicted_Sentiment"].value_counts().to_dict()
            length_breakdown[bucket] = {
                "total": len(grp),
                "Positive": b_counts.get("Positive", 0),
                "Neutral": b_counts.get("Neutral", 0),
                "Negative": b_counts.get("Negative", 0),
                "pos_rate": round((b_counts.get("Positive", 0) / len(grp)) * 100, 1)
            }

        X_tfidf = self.vectorizer.transform(self.df["clean_text"])
        mean_tfidf = np.array(X_tfidf.mean(axis=0)).flatten()
        feature_names = self.vectorizer.get_feature_names_out()

        top_indices = mean_tfidf.argsort()[::-1][:15]
        top_tfidf_vocab = [
            {"term": feature_names[idx].replace("_", " "), "mean_tfidf": round(float(mean_tfidf[idx]), 4)}
            for idx in top_indices
        ]

        if self.classifier is not None:
            coefs = self.classifier.coef_[0]
            top_pos_idx = coefs.argsort()[::-1][:10]
            top_neg_idx = coefs.argsort()[:10]
            top_pos_terms = [{"term": feature_names[i].replace("_", " "), "weight": round(float(coefs[i]), 2)} for i in top_pos_idx]
            top_neg_terms = [{"term": feature_names[i].replace("_", " "), "weight": round(float(coefs[i]), 2)} for i in top_neg_idx]
        else:
            top_pos_terms = [{"term": k, "weight": v} for k, v in list(DOMAIN_POSITIVE.items())[:10]]
            top_neg_terms = [{"term": k, "weight": v} for k, v in list(DOMAIN_NEGATIVE.items())[:10]]

        return {
            "dataset_name": self.dataset_name,
            "total_reviews": total,
            "sentiment_distribution": {
                "Positive": {"count": pos_cnt, "percentage": round((pos_cnt / total) * 100, 2)},
                "Neutral": {"count": neu_cnt, "percentage": round((neu_cnt / total) * 100, 2)},
                "Negative": {"count": neg_cnt, "percentage": round((neg_cnt / total) * 100, 2)}
            },
            "average_word_count": round(float(self.df["word_count"].mean()), 1),
            "length_breakdown": length_breakdown,
            "top_tfidf_vocabulary": top_tfidf_vocab,
            "top_positive_drivers": top_pos_terms,
            "top_negative_drivers": top_neg_terms
        }

    def get_review_feedback(self, query: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Retrieves detailed token-by-token TF-IDF & sentiment analysis for a specific Review ID
        or row number (e.g., 'FMCG00003', 'FMCY_00003', '#3', or index 2).
        """
        if "Predicted_Sentiment" not in self.df.columns:
            self.analyze_sentiment()

        row = None
        raw_str = str(query).strip()
        row_match = re.match(r"^(?:#|row\s*)?(\d+)$", raw_str, re.IGNORECASE)
        
        # 1. Exact match on ID column
        matched = self.df[self.df[self.id_col].astype(str).str.lower() == raw_str.lower()]
        if not matched.empty:
            row = matched.iloc[0]

        # 2. Normalized & typo match (e.g. FMCY <-> FMCG)
        if row is None:
            norm_q = re.sub(r"[^a-z0-9]", "", raw_str.lower())
            alt_qs = [norm_q]
            if "fmcy" in norm_q: alt_qs.append(norm_q.replace("fmcy", "fmcg"))
            if "fmcg" in norm_q: alt_qs.append(norm_q.replace("fmcg", "fmcy"))

            for _, r in self.df.iterrows():
                val_norm = re.sub(r"[^a-z0-9]", "", str(r[self.id_col]).lower())
                if val_norm in alt_qs:
                    row = r
                    break

        # 3. Numeric ID match (e.g. 3 matches FMCG00003 or REV-10493)
        if row is None:
            target_digits = re.sub(r"\D", "", raw_str)
            if target_digits:
                target_int = int(target_digits)
                for _, r in self.df.iterrows():
                    r_digits = re.sub(r"\D", "", str(r[self.id_col]))
                    if r_digits and int(r_digits) == target_int:
                        row = r
                        break

        # 4. Row index fallback
        if row is None and row_match:
            idx = int(row_match.group(1)) - 1
            if 0 <= idx < len(self.df):
                row = self.df.iloc[idx]

        if row is None:
            return None

        clean_t = str(row["clean_text"])
        orig_t = str(row[self.text_col])
        rev_id = str(row[self.id_col])
        sentiment = str(row["Predicted_Sentiment"])
        conf = float(row["Confidence"])
        p_pos = float(row["P_Positive"])
        p_neu = float(row["P_Neutral"])
        p_neg = float(row["P_Negative"])
        prod_name = str(row[self.product_col]) if (self.product_col and self.product_col in row) else "Product Feedback"
        rating_val = row[self.rating_col] if (self.rating_col and self.rating_col in row) else None

        # Detailed token breakdown
        X_vec = self.vectorizer.transform([clean_t])
        feature_names = self.vectorizer.get_feature_names_out()
        indices = X_vec.indices
        tfidf_data = X_vec.data

        feature_map = dict(zip(feature_names, self.classifier.coef_[0])) if self.classifier is not None else {**DOMAIN_POSITIVE, **DOMAIN_NEGATIVE}
        
        words = clean_t.split()
        token_rows = []
        for idx, tfidf_val in zip(indices, tfidf_data):
            term = str(feature_names[idx])
            idf = float(self.vectorizer.idf_[idx]) if hasattr(self.vectorizer, "idf_") else 1.0
            coef = float(feature_map.get(term, 0.0))
            
            raw_tf = clean_t.count(term) if " " in term else words.count(term)
            if raw_tf < 1: raw_tf = 1
            
            contrib = float(coef * tfidf_val)
            influence = "Positive" if contrib > 0.01 else ("Negative" if contrib < -0.01 else "Neutral")

            display_term = (
                term.replace("explicit_five_star", "five stars")
                .replace("explicit_four_star", "four stars")
                .replace("explicit_one_star", "one star")
                .replace("won_me_over", "won me over")
                .replace("not_", "not ")
            )

            token_rows.append({
                "token": display_term,
                "tf": raw_tf,
                "idf": round(idf, 4),
                "tfidf_score": round(float(tfidf_val), 4),
                "coefficient": round(coef, 4),
                "sentiment_contribution": round(contrib, 4),
                "influence": influence,
                "is_ngram": " " in display_term
            })

        token_rows.sort(key=lambda x: abs(x["sentiment_contribution"]), reverse=True)

        return {
            "review_id": rev_id,
            "product_name": prod_name,
            "rating": rating_val,
            "original_text": orig_t,
            "cleaned_text": clean_t,
            "predicted_sentiment": sentiment,
            "confidence": conf,
            "probabilities": {
                "Positive": p_pos,
                "Neutral": p_neu,
                "Negative": p_neg
            },
            "word_count": int(row["word_count"]),
            "length_bucket": str(row["length_bucket"]),
            "tokens": token_rows
        }

    def get_review_tfidf_table(self, query: Union[str, int]) -> Optional[pd.DataFrame]:
        """
        Returns the formatted TF-IDF table for a specific feedback ID as a Pandas DataFrame.
        Columns: Token / N-Gram, Is N-Gram, Term Frequency (TF), Inverse Document Frequency (IDF),
                 TF-IDF Score, Model Coefficient, Net Contribution, Influence.
        """
        feedback = self.get_review_feedback(query)
        if not feedback or not feedback.get("tokens"):
            return None
        df_tokens = pd.DataFrame(feedback["tokens"])
        rename_cols = {
            "token": "Token / N-Gram",
            "is_ngram": "Is N-Gram",
            "tf": "Term Frequency (TF)",
            "idf": "Inverse Document Frequency (IDF)",
            "tfidf_score": "TF-IDF Score",
            "coefficient": "Model Coefficient",
            "sentiment_contribution": "Net Contribution",
            "influence": "Influence"
        }
        return df_tokens.rename(columns=rename_cols)

    def evaluate_model(self) -> Optional[Dict[str, Any]]:
        """
        Evaluates predictions against ground truth (from Sentiments or Rating column).
        Returns dictionary with accuracy, precision, recall, f1_score, confusion_matrix, and report.
        """
        if "Predicted_Sentiment" not in self.df.columns:
            self.analyze_sentiment()

        y_true = None
        if self.sentiment_col and self.sentiment_col in self.df.columns:
            def norm_sent(s):
                s_low = str(s).strip().lower()
                if "pos" in s_low or s_low in ["1", "5", "4"]: return "Positive"
                if "neg" in s_low or s_low in ["0", "1", "2"]: return "Negative"
                return "Neutral"
            y_true = self.df[self.sentiment_col].apply(norm_sent)
        elif self.rating_col and self.rating_col in self.df.columns:
            def norm_rate(r):
                try:
                    val = float(r)
                    if val >= 4: return "Positive"
                    if val <= 2: return "Negative"
                    return "Neutral"
                except Exception:
                    return "Neutral"
            y_true = self.df[self.rating_col].apply(norm_rate)

        if y_true is None:
            return None

        y_pred = self.df["Predicted_Sentiment"]
        labels = ["Positive", "Neutral", "Negative"]

        acc = float(accuracy_score(y_true, y_pred))
        prec_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec_w, 4),
            "recall": round(rec_w, 4),
            "f1_score": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "confusion_matrix": cm,
            "labels": labels,
            "classification_report": report,
            "total_evaluated": len(y_true)
        }

    def plot_confusion_matrix_dashboard(self, output_path: str = "outputs/confusion_matrix_dashboard.png"):
        """
        Renders and saves a visual dashboard showing:
        - 4 Metric KPI Cards (Accuracy, Precision, Recall, F1 Score)
        - Confusion Matrix Heatmap (annotated with count and percentage)
        """
        eval_res = self.evaluate_model()
        if not eval_res:
            print("[Pipeline] Notice: Cannot plot confusion matrix because ground truth labels were not found.")
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = eval_res["confusion_matrix"]
        labels = eval_res["labels"]
        acc = eval_res["accuracy"]
        prec = eval_res["precision"]
        rec = eval_res["recall"]
        f1 = eval_res["f1_score"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})

        # Panel 1: Metrics KPI cards
        ax1.axis("off")
        ax1.text(0.5, 0.96, "MODEL PERFORMANCE DASHBOARD", ha="center", va="top", fontsize=14, weight="bold", color="#1e293b")
        ax1.text(0.5, 0.88, f"Dataset: {self.dataset_name} (N={eval_res['total_evaluated']:,})", ha="center", va="top", fontsize=11, color="#64748b")

        metrics = [
            ("Accuracy", f"{acc*100:.1f}%", "#3b82f6"),
            ("Precision (Weighted)", f"{prec*100:.1f}%", "#10b981"),
            ("Recall (Weighted)", f"{rec*100:.1f}%", "#f59e0b"),
            ("F1-Score (Weighted)", f"{f1*100:.1f}%", "#8b5cf6")
        ]

        for idx, (m_name, m_val, col) in enumerate(metrics):
            y_pos = 0.72 - idx * 0.16
            rect = plt.Rectangle((0.05, y_pos - 0.05), 0.90, 0.12, facecolor="#f8fafc", edgecolor="#cbd5e1", transform=ax1.transAxes, lw=1.5, zorder=1)
            ax1.add_patch(rect)
            ax1.text(0.10, y_pos + 0.01, m_name, transform=ax1.transAxes, fontsize=11, weight="bold", color="#334155", va="center")
            ax1.text(0.85, y_pos + 0.01, m_val, transform=ax1.transAxes, fontsize=13, weight="bold", color=col, ha="right", va="center")

        # Panel 2: Confusion Matrix Heatmap
        cm_pct = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1) * 100
        annot = np.empty_like(cm, dtype=object)
        for r in range(len(labels)):
            for c in range(len(labels)):
                annot[r, c] = f"{cm[r, c]:,}\n({cm_pct[r, c]:.1f}%)"

        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True, ax=ax2, linewidths=1.5, linecolor="white")
        ax2.set_title("Confusion Matrix (Counts & Class Recall %)", fontsize=13, weight="bold", pad=12)
        ax2.set_xlabel("Predicted Sentiment", fontsize=11, weight="bold")
        ax2.set_ylabel("Actual Ground Truth", fontsize=11, weight="bold")

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"[Pipeline] Saved Confusion Matrix Dashboard to: {output_path}")
        return output_path

    def print_summary(self):
        """Prints a rich, formatted terminal dashboard summary of the dataset analysis."""
        summary = self.get_dataset_summary()
        dist = summary["sentiment_distribution"]
        
        print("\n" + "=" * 80)
        print(f"      DATASET SENTIMENT & TF-IDF ANALYSIS REPORT: {summary['dataset_name']}")
        print("=" * 80)
        print(f" Total Feedback Records: {summary['total_reviews']:,}")
        print(f" Average Words / Review : {summary['average_word_count']} words\n")

        print("--- OVERALL SENTIMENT CLASSIFICATION ---")
        print(f"  [+] POSITIVE : {dist['Positive']['count']:>6} reviews ({dist['Positive']['percentage']:>5.1f}%) " + "█" * int(dist['Positive']['percentage'] // 4))
        print(f"  [=] NEUTRAL  : {dist['Neutral']['count']:>6} reviews ({dist['Neutral']['percentage']:>5.1f}%) " + "█" * int(dist['Neutral']['percentage'] // 4))
        print(f"  [-] NEGATIVE : {dist['Negative']['count']:>6} reviews ({dist['Negative']['percentage']:>5.1f}%) " + "█" * int(dist['Negative']['percentage'] // 4))

        eval_res = self.evaluate_model()
        if eval_res:
            print("\n--- MODEL PERFORMANCE METRICS & EVALUATION ---")
            print(f"  Accuracy            : {eval_res['accuracy']*100:.2f}%")
            print(f"  Precision (Weighted): {eval_res['precision']*100:.2f}%")
            print(f"  Recall (Weighted)   : {eval_res['recall']*100:.2f}%")
            print(f"  F1 Score (Weighted) : {eval_res['f1_score']*100:.2f}% (Macro F1: {eval_res['f1_macro']*100:.2f}%)")
            
            print("\n--- CONFUSION MATRIX (Rows: Actual, Cols: Predicted) ---")
            cm = eval_res["confusion_matrix"]
            print(f"  {'Actual vs Pred':<15} | {'Positive':<10} | {'Neutral':<10} | {'Negative':<10}")
            print("  " + "-" * 50)
            print(f"  {'Positive':<15} | {cm[0, 0]:<10} | {cm[0, 1]:<10} | {cm[0, 2]:<10}")
            print(f"  {'Neutral':<15} | {cm[1, 0]:<10} | {cm[1, 1]:<10} | {cm[1, 2]:<10}")
            print(f"  {'Negative':<15} | {cm[2, 0]:<10} | {cm[2, 1]:<10} | {cm[2, 2]:<10}")

        print("\n--- SENTIMENT BY REVIEW LENGTH ---")
        for bucket, stats in summary["length_breakdown"].items():
            print(f"  * {bucket:<25} | N={stats['total']:<5} | Pos: {stats['pos_rate']}% (Pos:{stats['Positive']}, Neu:{stats['Neutral']}, Neg:{stats['Negative']})")

        print("\n--- TOP VOCABULARY TERMS (HIGHEST TF-IDF SCORES IN DATASET) ---")
        vocab_terms = [f"{v['term']} ({v['mean_tfidf']})" for v in summary["top_tfidf_vocabulary"][:8]]
        print("  " + " • ".join(vocab_terms))

        print("\n--- TOP POSITIVE PRAISE DRIVERS (P(Positive) ^) ---")
        pos_terms = [f"'{t['term']}' (+{t['weight']})" for t in summary["top_positive_drivers"][:6]]
        print("  [+] " + " • ".join(pos_terms))

        print("\n--- TOP NEGATIVE CRITIQUE DRIVERS (P(Negative) ^) ---")
        neg_terms = [f"'{t['term']}' ({t['weight']})" for t in summary["top_negative_drivers"][:6]]
        print("  [-] " + " • ".join(neg_terms))
        print("=" * 80 + "\n")

    def print_review_detail(self, review_id: str):
        """Prints a detailed sentiment & TF-IDF table for a specific feedback ID."""
        feedback = self.get_review_feedback(review_id)
        if not feedback:
            print(f"\n[!] Review ID '{review_id}' could not be found in active dataset.\n")
            return

        sentiment_tag = {
            "Positive": "\033[92mPOSITIVE\033[0m",
            "Negative": "\033[91mNEGATIVE\033[0m",
            "Neutral": "\033[93mNEUTRAL\033[0m"
        }.get(feedback["predicted_sentiment"], feedback["predicted_sentiment"])

        print("\n" + "-" * 75)
        print(f" FEEDBACK INSPECTION: ID: {feedback['review_id']} | Product: {feedback['product_name']}")
        if feedback["rating"] is not None:
            print(f" Rating: {feedback['rating']}★")
        print(f' Text: "{feedback["original_text"]}"')
        print(f" Classification: {sentiment_tag} (Confidence: {feedback['confidence']*100:.1f}%)")
        print(f" Probabilities : P(Pos)={feedback['probabilities']['Positive']:.4f} | P(Neu)={feedback['probabilities']['Neutral']:.4f} | P(Neg)={feedback['probabilities']['Negative']:.4f}")
        print("-" * 75)
        print(f"{'Token / N-Gram':<22} | {'TF':<4} | {'IDF':<7} | {'TF-IDF':<8} | {'Influence':<10} | {'Contribution'}")
        print("-" * 75)

        for t in feedback["tokens"]:
            token_display = f'"{t["token"]}"'
            if t["is_ngram"]:
                token_display += " (ngram)"
            print(f"{token_display:<22} | {t['tf']:<4} | {t['idf']:<7.4f} | {t['tfidf_score']:<8.4f} | {t['influence']:<10} | {t['sentiment_contribution']:+.4f}")
        print("-" * 75 + "\n")

    def export_results(self, output_path: str = "outputs/analyzed_dataset_report.csv"):
        """Exports the annotated dataset with sentiment analysis and TF-IDF attribution."""
        if "Predicted_Sentiment" not in self.df.columns:
            self.analyze_sentiment()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith(".json"):
            self.df.to_json(output_path, orient="records", indent=2)
        else:
            self.df.to_csv(output_path, index=False)
        print(f"[Pipeline] Exported {len(self.df)} analyzed records to: {output_path}")


def run_cli():
    """Command Line Interface for the Review Dataset Sentiment & TF-IDF Pipeline."""
    parser = argparse.ArgumentParser(description="End-to-End Sentiment & TF-IDF Dataset Analysis Pipeline")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to dataset file (CSV, TSV, JSON, Excel)")
    parser.add_argument("--export", "-e", type=str, default="outputs/analyzed_dataset_report.csv", help="Path to export analyzed report")
    parser.add_argument("--query", "-q", type=str, default=None, help="Inspect a specific Review ID immediately")
    args = parser.parse_args()

    file_path = args.file
    if file_path is None:
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "active_uploaded_reviews.csv"),
            os.path.join(PROJECT_ROOT, "data", "raw_reviews.csv")
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 50:
                file_path = c
                break

    if file_path is None:
        print("[!] No dataset file provided. Usage: python pipeline_analyzer.py --file <path/to/dataset.csv>")
        sys.exit(1)

    pipeline = SentimentTFIDFPipeline()
    pipeline.load_dataset(file_path)
    pipeline.preprocess()
    pipeline.fit_transform_tfidf()
    pipeline.analyze_sentiment()
    pipeline.print_summary()

    if args.export:
        pipeline.export_results(args.export)

    if args.query:
        pipeline.print_review_detail(args.query)
    else:
        # Default inspection of FMCG00003 or first review
        first_id = pipeline.df[pipeline.id_col].iloc[2] if len(pipeline.df) > 2 else pipeline.df[pipeline.id_col].iloc[0]
        print(f"Inspecting sample review '{first_id}'...")
        pipeline.print_review_detail(first_id)


if __name__ == "__main__":
    run_cli()
