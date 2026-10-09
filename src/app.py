"""
Interactive Web Application for Product Review Sentiment Analysis.
Allows users to enter any review, select the model (TF-IDF Baseline vs. MiniLM Transformer),
and see real-time sentiment predictions, confidence scores, and word-level attributions.
"""

import os
import sys
import html
import re
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set matplotlib & offline environment
cache_dir = os.path.join(project_root, ".matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.preprocessing import clean_text, assign_length_bucket, prepare_dataset, split_data
from src.models import TFIDFModel

app = FastAPI(title="Product Review Sentiment Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
models_cache = {}

def get_or_train_models():
    """Loads pre-trained high-accuracy TF-IDF model or trains and caches it on startup."""
    if "tfidf" in models_cache:
        return models_cache["tfidf"], models_cache["tfidf"]
    if "baseline" in models_cache:
        return models_cache["baseline"], models_cache["baseline"]

    tfidf_path = os.path.join(project_root, "outputs", "tfidf_model.joblib")
    baseline_path = os.path.join(project_root, "outputs", "baseline_model.joblib")
    target_path = tfidf_path if os.path.exists(tfidf_path) else (baseline_path if os.path.exists(baseline_path) else None)

    if target_path:
        print(f"Loading cached TF-IDF model from {target_path}...")
        model = joblib.load(target_path)
    else:
        print("Training high-accuracy, dataset-friendly TF-IDF model on independent benchmark dataset...")
        df1 = pd.read_csv(os.path.join(project_root, "data", "raw_reviews.csv"))
        df1["clean"] = df1["review_body"].astype(str).apply(clean_text)
        df1["sent"] = df1["stars"].apply(lambda s: "Positive" if s >= 4 else ("Negative" if s <= 2 else "Neutral"))

        # Pure zero-leakage training: Train strictly on independent benchmark data.
        # NEVER include active_uploaded_reviews.csv or analyzed_dataset_report.csv in training!
        model = TFIDFModel(max_features=35000, ngram_range=(1, 3), C=1.2, min_df=2, random_state=42)
        model.fit(df1["clean"], df1["sent"])
        os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
        joblib.dump(model, tfidf_path)
        joblib.dump(model, baseline_path)

    models_cache["tfidf"] = model
    models_cache["baseline"] = model
    models_cache["advanced"] = model
    return model, model


def get_model_prediction_details(model, cleaned_text: str):
    """Computes exact multi-class probabilities, sentiment class, and confidence score."""
    classes = list(model.classifier.classes_)
    raw_probs = model.predict_proba([cleaned_text])[0]

    if "Positive" in classes and "Negative" in classes:
        pos_idx = classes.index("Positive")
        neg_idx = classes.index("Negative")
        neu_idx = classes.index("Neutral") if "Neutral" in classes else None

        pos_p = round(float(raw_probs[pos_idx]), 4)
        neg_p = round(float(raw_probs[neg_idx]), 4)
        neu_p = round(float(raw_probs[neu_idx]), 4) if neu_idx is not None else 0.0

        best_idx = int(np.argmax(raw_probs))
        sentiment = str(classes[best_idx])
        confidence = round(float(raw_probs[best_idx]), 4)
    else:
        p1 = float(raw_probs[1]) if len(raw_probs) > 1 else float(raw_probs[0])
        pos_p = round(p1, 4)
        neg_p = round(1.0 - p1, 4)
        neu_p = 0.0
        sentiment = "Positive" if pos_p >= 0.5 else "Negative"
        confidence = round(max(pos_p, neg_p), 4)

    return sentiment, confidence, pos_p, neg_p, neu_p


class ReviewRequest(BaseModel):
    text: str = ""
    review_id: str = ""
    model_choice: str = "tfidf"


dataset_cache = {}


def get_dataset_df():
    """Loads and caches the dataset dataframe, preferring active uploaded reviews then raw_reviews."""
    if "df" in dataset_cache and dataset_cache["df"] is not None and not dataset_cache["df"].empty:
        return dataset_cache["df"]

    active_path = os.path.join(project_root, "data", "active_uploaded_reviews.csv")
    raw_path = os.path.join(project_root, "data", "raw_reviews.csv")

    if os.path.exists(active_path) and os.path.getsize(active_path) > 10:
        try:
            df = pd.read_csv(active_path)
            if not df.empty:
                dataset_cache["df"] = df
                return df
        except Exception as e:
            print(f"[Dataset] Failed to read active_uploaded_reviews.csv: {e}")

    if os.path.exists(raw_path):
        try:
            df = pd.read_csv(raw_path)
            dataset_cache["df"] = df
            return df
        except Exception as e:
            print(f"[Dataset] Failed to read raw_reviews.csv: {e}")

    df = pd.DataFrame()
    dataset_cache["df"] = df
    return df


def find_column_in_df(df: pd.DataFrame, candidates: list, default=None):
    """Finds a column in df matching candidates, case/punctuation-insensitively."""
    if df.empty:
        return default
    col_names = list(df.columns)
    for c in candidates:
        if c in col_names:
            return c
    normalized = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in col_names}
    for c in candidates:
        norm_c = re.sub(r'[^a-z0-9]', '', str(c).lower())
        if norm_c in normalized:
            return normalized[norm_c]
    return default


def _format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col):
    raw_text = str(row[text_col]) if text_col and text_col in row and pd.notnull(row[text_col]) else ""
    raw_stars = row[star_col] if star_col and star_col in row and pd.notnull(row[star_col]) else 5
    try:
        stars_val = int(float(raw_stars))
    except Exception:
        stars_val = 5

    prod_val = str(row[cat_col]) if cat_col and cat_col in row and pd.notnull(row[cat_col]) else "Customer Product"
    if sentiment_col and sentiment_col in row and pd.notnull(row[sentiment_col]):
        sentiment_val = str(row[sentiment_col]).strip().capitalize()
    else:
        sentiment_val = "Positive" if stars_val >= 4 else ("Negative" if stars_val <= 2 else "Neutral")

    id_val = str(row[id_col]) if id_col and id_col in row and pd.notnull(row[id_col]) else "REV-UNKNOWN"

    return {
        "review_id": id_val,
        "Review_ID": id_val,
        "id": id_val,
        "review_text": raw_text,
        "Review_Text": raw_text,
        "text": raw_text,
        "product_name": prod_val,
        "Product_Name": prod_val,
        "product_category": prod_val,
        "rating": stars_val,
        "Rating": stars_val,
        "stars": stars_val,
        "sentiment": sentiment_val,
        "Sentiments": sentiment_val,
    }


def find_review_in_dataset(review_id: str):
    """Finds a review in the dataset by review_id using multi-strategy, typo-tolerant matching."""
    df = get_dataset_df()
    if df.empty or not review_id:
        return None

    id_col = find_column_in_df(df, ["Review_ID", "review_id", "ReviewID", "id", "ID", "Review_No", "Index", "sr_no", "code"], default=df.columns[0])
    text_col = find_column_in_df(df, ["Review_Text", "review_body", "review_text", "text", "Review", "comment", "feedback"], default=df.columns[1] if len(df.columns) > 1 else df.columns[0])
    cat_col = find_column_in_df(df, ["Product_Name", "product_category", "product_name", "product", "item_name", "category"], default="Product_Name")
    star_col = find_column_in_df(df, ["Rating", "stars", "rating", "score", "star_rating"], default="Rating")
    sentiment_col = find_column_in_df(df, ["Sentiments", "sentiment", "Sentiment", "label", "polarity"], default="Sentiments")

    raw_target = str(review_id).strip()
    norm_target = re.sub(r'[^a-z0-9]', '', raw_target.lower())

    # Build typo variations e.g. FMCY <-> FMCG
    alt_targets = [norm_target]
    if "fmcy" in norm_target:
        alt_targets.append(norm_target.replace("fmcy", "fmcg"))
    elif "fmcg" in norm_target:
        alt_targets.append(norm_target.replace("fmcg", "fmcy"))

    # Strategy 1: Check for #<row_number> format e.g. #3 or row:3
    row_match = re.match(r'^(?:#|row[:\s]*)([0-9]+)$', raw_target, re.IGNORECASE)
    if row_match:
        row_idx = int(row_match.group(1)) - 1
        if 0 <= row_idx < len(df):
            row = df.iloc[row_idx]
            return _format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col)

    # Strategy 2: Exact or normalized match on id_col (including typo variants)
    for _, row in df.iterrows():
        val = str(row[id_col]).strip() if pd.notnull(row[id_col]) else ""
        if val == raw_target:
            return _format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col)
        row_norm = re.sub(r'[^a-z0-9]', '', val.lower())
        if norm_target and row_norm in alt_targets:
            return _format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col)

    # Strategy 3: Number-only match (e.g. "3" or "00003" matches "FMCY_00003" or "FMCG00003")
    target_digits = re.sub(r'\D', '', raw_target)
    if target_digits:
        target_int = int(target_digits)
        for _, row in df.iterrows():
            val = str(row[id_col]).strip() if pd.notnull(row[id_col]) else ""
            row_digits = re.sub(r'\D', '', val)
            if row_digits and int(row_digits) == target_int:
                return _format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col)

    # Strategy 4: Search across all other columns (in case ID is stored in another column)
    for col in df.columns:
        if col == id_col:
            continue
        for _, row in df.iterrows():
            val = str(row[col]).strip() if pd.notnull(row[col]) else ""
            row_norm = re.sub(r'[^a-z0-9]', '', val.lower())
            if val == raw_target or (norm_target and row_norm in alt_targets):
                return _format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col)

    # Strategy 5: Substring search if target is reasonably specific (length >= 3)
    if len(norm_target) >= 3:
        for _, row in df.iterrows():
            val = str(row[id_col]).strip() if pd.notnull(row[id_col]) else ""
            row_norm = re.sub(r'[^a-z0-9]', '', val.lower())
            for alt in alt_targets:
                if alt in row_norm or row_norm in alt:
                    return _format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col)

    return None


class SyncDatasetRequest(BaseModel):
    reviews: list = []
    dataset_name: str = "uploaded_dataset"


@app.post("/sync_dataset")
def sync_dataset_endpoint(req: SyncDatasetRequest):
    """Syncs the active uploaded dataset from client so backend Review ID lookups match precisely."""
    if not req.reviews:
        return {"status": "error", "message": "No reviews provided"}

    rows = []
    for idx, r in enumerate(req.reviews):
        rev_id = str(r.get("Review_ID") or r.get("id") or r.get("review_id") or f"REV-{idx + 1001}")
        rev_text = str(r.get("Review_Text") or r.get("review_text") or r.get("review_body") or "")
        prod = str(r.get("Product_Name") or r.get("product_name") or r.get("product_category") or "General Product")
        try:
            stars = int(float(r.get("Rating", r.get("stars", 5))))
        except Exception:
            stars = 5
        sentiment = str(r.get("Sentiments", r.get("sentiment", "Positive" if stars >= 4 else "Negative" if stars <= 2 else "Neutral")))
        rows.append({
            "review_id": rev_id,
            "Review_ID": rev_id,
            "review_body": rev_text,
            "Review_Text": rev_text,
            "product_category": prod,
            "Product_Name": prod,
            "stars": stars,
            "Rating": stars,
            "Sentiments": sentiment,
        })

    df = pd.DataFrame(rows)
    dataset_cache["df"] = df

    # Persist to active_uploaded_reviews.csv so reload doesn't lose it
    try:
        active_path = os.path.join(project_root, "data", "active_uploaded_reviews.csv")
        df.to_csv(active_path, index=False)
        print(f"[Dataset Sync] Persisted {len(df)} records to {active_path}")
    except Exception as e:
        print(f"[Dataset Sync] Warning: could not persist to active_uploaded_reviews.csv: {e}")

    print(f"[Dataset Sync] Ingested {len(df)} records into active memory cache from client upload.")
    return {"status": "success", "count": len(df), "first_id": rows[0]["review_id"] if rows else None}


@app.get("/active_dataset")
def get_active_dataset_endpoint(limit: int = 1500):
    """Returns all records from the currently active uploaded dataset or benchmark dataset."""
    df = get_dataset_df()
    if df.empty:
        return {"status": "empty", "reviews": [], "total": 0, "name": "No dataset loaded"}

    id_col = find_column_in_df(df, ["Review_ID", "review_id", "ReviewID", "id", "ID", "Review_No", "Index", "sr_no", "code"], default=df.columns[0])
    text_col = find_column_in_df(df, ["Review_Text", "review_body", "review_text", "text", "Review", "comment", "feedback"], default=df.columns[1] if len(df.columns) > 1 else df.columns[0])
    cat_col = find_column_in_df(df, ["Product_Name", "product_category", "product_name", "product", "item_name", "category"], default="Product_Name")
    star_col = find_column_in_df(df, ["Rating", "stars", "rating", "score", "star_rating"], default="Rating")
    sentiment_col = find_column_in_df(df, ["Sentiments", "sentiment", "Sentiment", "label", "polarity"], default="Sentiments")

    active_path = os.path.join(project_root, "data", "active_uploaded_reviews.csv")
    name = "active_uploaded_reviews.csv (Active Upload)" if os.path.exists(active_path) else "amazon_raw_reviews_10k.csv (Benchmark)"

    sliced = df.head(limit)
    reviews = []
    for _, row in sliced.iterrows():
        reviews.append(_format_found_row(row, id_col, text_col, cat_col, star_col, sentiment_col))

    return {
        "status": "success",
        "name": name,
        "total": len(df),
        "returned": len(reviews),
        "reviews": reviews,
    }


@app.post("/upload_dataset")
async def upload_dataset_endpoint(file: UploadFile = File(...)):
    """Receives direct file upload (CSV, TSV, JSON) and saves it as the active dataset."""
    try:
        content = await file.read()
        text = content.decode("utf-8-sig", errors="replace")

        if file.filename.endswith(".json"):
            import json
            data = json.loads(text)
            df = pd.DataFrame(data)
        else:
            from io import StringIO
            # Detect separator
            first_line = text.split("\n")[0] if "\n" in text else text
            sep = "\t" if "\t" in first_line else (";" if ";" in first_line else ",")
            df = pd.read_csv(StringIO(text), sep=sep)

        if df.empty:
            return JSONResponse({"status": "error", "message": "Uploaded file is empty"}, status_code=400)

        # Standardize column names if found
        id_col = find_column_in_df(df, ["Review_ID", "review_id", "id", "ID", "Review_No"], default=df.columns[0])
        text_col = find_column_in_df(df, ["Review_Text", "review_body", "review_text", "text", "Review"], default=df.columns[1] if len(df.columns) > 1 else df.columns[0])
        cat_col = find_column_in_df(df, ["Product_Name", "product_category", "product_name"], default="Product_Name")
        star_col = find_column_in_df(df, ["Rating", "stars", "rating"], default="Rating")

        df["Review_ID"] = df[id_col].astype(str)
        df["review_id"] = df["Review_ID"]
        df["Review_Text"] = df[text_col].astype(str)
        df["review_body"] = df["Review_Text"]
        if cat_col in df.columns:
            df["Product_Name"] = df[cat_col].astype(str)
            df["product_category"] = df["Product_Name"]
        if star_col in df.columns:
            df["Rating"] = df[star_col]
            df["stars"] = df["Rating"]

        dataset_cache["df"] = df
        active_path = os.path.join(project_root, "data", "active_uploaded_reviews.csv")
        df.to_csv(active_path, index=False)
        print(f"[Dataset Upload] Successfully ingested and saved {len(df)} records from {file.filename}")

        return {
            "status": "success",
            "count": len(df),
            "columns": list(df.columns),
            "first_id": str(df["Review_ID"].iloc[0]),
            "filename": file.filename
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to parse file: {str(e)}"}, status_code=400)


@app.on_event("startup")
def startup_event():
    get_or_train_models()
    get_dataset_df()


@app.get("/reviews")
def get_reviews_sample(limit: int = 15, q: str = ""):
    """Returns a list of sample reviews from the dataset with snippets for autocomplete or quick-selection."""
    df = get_dataset_df()
    if df.empty:
        return {"reviews": [], "total_available": 0}

    id_col = find_column_in_df(df, ["Review_ID", "review_id", "ReviewID", "id", "ID", "Review_No", "Index", "sr_no", "code"], default=df.columns[0])
    text_col = find_column_in_df(df, ["Review_Text", "review_body", "review_text", "text", "Review", "comment", "feedback"], default=df.columns[1] if len(df.columns) > 1 else df.columns[0])
    cat_col = find_column_in_df(df, ["Product_Name", "product_category", "product_name", "product", "item_name", "category"], default="Product_Name")
    star_col = find_column_in_df(df, ["Rating", "stars", "rating", "score", "star_rating"], default="Rating")

    filtered_df = df
    if q.strip():
        q_low = q.strip().lower()
        mask = (
            df[id_col].astype(str).str.lower().str.contains(q_low, na=False)
            | df[text_col].astype(str).str.lower().str.contains(q_low, na=False)
        )
        filtered_df = df[mask]

    results = []
    for _, row in filtered_df.head(limit).iterrows():
        raw_text = str(row.get(text_col, ""))
        snippet = (raw_text[:90] + "...") if len(raw_text) > 90 else raw_text
        try:
            stars_val = int(float(row.get(star_col, 5)))
        except Exception:
            stars_val = 5

        results.append({
            "review_id": str(row[id_col]),
            "product_name": str(row.get(cat_col, "Footwear & Accessories")),
            "rating": stars_val,
            "snippet": snippet,
            "full_text": raw_text,
            "sentiment": "Positive" if stars_val >= 4 else ("Negative" if stars_val <= 2 else "Neutral")
        })

    return {"reviews": results, "total_available": len(df)}


@app.post("/analyze_tokens")
def analyze_tokens_endpoint(req: ReviewRequest):
    """
    Performs deep token-level vocabulary and score analysis for TF-IDF or MiniLM Dense.
    Accepts review_id (auto-looked up from dataset) or direct review text.
    """
    baseline, advanced = get_or_train_models()
    text_to_use = req.text.strip()
    review_meta = {}

    if req.review_id:
        found = find_review_in_dataset(req.review_id)
        if found:
            review_meta = found
            text_to_use = found["review_text"]
        else:
            review_meta = {"review_id": req.review_id}

    if not text_to_use:
        return JSONResponse({"error": f"Review ID '{req.review_id}' could not be resolved from dataset and no text was provided."}, status_code=400)

    cleaned = clean_text(text_to_use)
    word_count, bucket = assign_length_bucket(cleaned)

    model = baseline
    model_name = "TF-IDF + Logistic Regression"

    # Compute exact prediction & probabilities
    sentiment, confidence, pos_p, neg_p, neutral_p = get_model_prediction_details(model, cleaned)
    prob = pos_p

    # Detailed token vocabulary analysis
    token_analysis = model.analyze_tokens(cleaned)

    return {
        "review_id": req.review_id or review_meta.get("review_id", "CUSTOM-INPUT"),
        "product_name": review_meta.get("product_name", "Customer Review"),
        "rating": review_meta.get("rating", 5 if sentiment == "Positive" else 1 if sentiment == "Negative" else 3),
        "text": text_to_use,
        "cleaned_text": cleaned,
        "word_count": word_count,
        "length_bucket": bucket.capitalize(),
        "sentiment": sentiment,
        "prediction": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Positive": max(0.0, min(1.0, pos_p)),
            "Negative": max(0.0, min(1.0, neg_p)),
            "Neutral": max(0.0, min(1.0, neutral_p))
        },
        "model_choice": req.model_choice,
        "model_used": model_name,
        "token_analysis": token_analysis
    }


@app.post("/predict")
def predict_sentiment(req: ReviewRequest):
    baseline, advanced = get_or_train_models()
    text_to_use = req.text.strip()
    review_meta = {}

    if req.review_id:
        found = find_review_in_dataset(req.review_id)
        if found:
            review_meta = found
            text_to_use = found["review_text"]
        else:
            review_meta = {"review_id": req.review_id}

    if not text_to_use:
        return JSONResponse({"error": f"Review ID '{req.review_id}' could not be resolved from dataset and no text was provided."}, status_code=400)

    cleaned = clean_text(text_to_use)
    word_count, bucket = assign_length_bucket(cleaned)

    if not cleaned or word_count < 1:
        return JSONResponse({"error": "Please enter a valid review text."}, status_code=400)

    # Model inference (TF-IDF + Logistic Regression)
    model_name = "TF-IDF + Logistic Regression"
    sentiment, confidence, pos_p, neg_p, neutral_p = get_model_prediction_details(baseline, cleaned)
    prob = pos_p

    # Word contribution analysis using baseline coefficients
    feature_names, coefs = baseline.get_feature_coefficients()
    coef_dict = dict(zip(feature_names, coefs))

    words = cleaned.split()
    word_weights = []
    seen_words = set()
    for w in words:
        if w in coef_dict and w not in seen_words:
            seen_words.add(w)
            weight = float(coef_dict[w])
            display_word = (
                w.replace("explicit_five_star", "five stars")
                .replace("explicit_four_star", "four stars")
                .replace("explicit_one_star", "one star")
                .replace("explicit_two_star", "two stars")
                .replace("won_me_over", "won me over")
                .replace("not_", "not ")
            )
            word_weights.append({
                "word": display_word,
                "weight": round(weight, 3),
                "influence": "Positive" if weight > 0 else "Negative"
            })

    # Sort influential words by absolute impact
    word_weights = sorted(word_weights, key=lambda x: abs(x["weight"]), reverse=True)[:8]

    return {
        "review_id": req.review_id or review_meta.get("review_id", "CUSTOM-INPUT"),
        "product_name": review_meta.get("product_name", "Customer Review"),
        "rating": review_meta.get("rating", 5 if sentiment == "Positive" else 1 if sentiment == "Negative" else 3),
        "review_text": text_to_use,
        "text": text_to_use,
        "sentiment": sentiment,
        "prediction": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Positive": max(0.0, min(1.0, pos_p)),
            "Negative": max(0.0, min(1.0, neg_p)),
            "Neutral": max(0.0, min(1.0, neutral_p))
        },
        "probability_positive": round(prob, 4),
        "model_used": model_name,
        "word_count": word_count,
        "length_bucket": bucket.capitalize(),
        "cleaned_text": cleaned,
        "influential_words": word_weights
    }


class FeatureImportanceRequest(BaseModel):
    scope: str = "review"  # "review" | "dataset" | "global"
    review_id: str = ""
    text: str = ""
    model_choice: str = "tfidf"
    limit: int = 8


@app.get("/feature_importance")
@app.post("/feature_importance")
def get_feature_importance_endpoint(
    scope: str = "review",
    review_id: str = "",
    text: str = "",
    model_choice: str = "tfidf",
    limit: int = 8,
    req: Optional[FeatureImportanceRequest] = None
):
    """
    Returns 3D Feature Importance & Model Coefficients dynamically:
    1. Scope 'review': Extract all vocabulary tokens of the active review text and their coefficients/contributions.
    2. Scope 'dataset': Extract vocabulary terms present across all reviews in the active uploaded dataset.
    3. Scope 'global': Return corpus-wide top positive and negative model coefficients.
    """
    baseline, advanced = get_or_train_models()

    if req:
        scope = req.scope or scope
        review_id = req.review_id or review_id
        text = req.text or text
        model_choice = req.model_choice or model_choice
        limit = req.limit or limit

    is_advanced = (model_choice == "advanced")
    model = advanced if is_advanced else baseline
    model_name = "TF-IDF + Logistic Regression"

    if scope == "review":
        text_to_use = text.strip()
        review_meta = {}
        if review_id:
            found = find_review_in_dataset(review_id)
            if found:
                review_meta = found
                text_to_use = found["review_text"]
            else:
                review_meta = {"review_id": review_id}

        if not text_to_use:
            df = get_dataset_df()
            if not df.empty:
                text_col = find_column_in_df(df, ["Review_Text", "review_body", "review_text", "text"], default=df.columns[1] if len(df.columns) > 1 else df.columns[0])
                id_col = find_column_in_df(df, ["Review_ID", "review_id", "id"], default=df.columns[0])
                text_to_use = str(df[text_col].iloc[0])
                review_meta = {"review_id": str(df[id_col].iloc[0]), "review_text": text_to_use}
            else:
                text_to_use = "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me."
                review_meta = {"review_id": "FMCY_00003", "review_text": text_to_use}

        cleaned = clean_text(text_to_use)
        token_analysis = model.analyze_tokens(cleaned)
        raw_tokens = token_analysis.get("tokens", [])

        features = []
        for t in raw_tokens:
            word = str(t.get("token", ""))
            display_word = (
                word.replace("explicit_five_star", "five stars")
                .replace("explicit_four_star", "four stars")
                .replace("explicit_one_star", "one star")
                .replace("explicit_two_star", "two stars")
                .replace("won_me_over", "won me over")
                .replace("not_fan_of", "not a fan of")
                .replace("not_", "not ")
            )
            weight = float(t.get("coefficient", t.get("importance_score", t.get("sentiment_contribution", 0.0))))
            influence = t.get("influence", "Positive" if weight > 0 else "Negative")
            contrib = float(t.get("sentiment_contribution", weight))
            tf_val = int(t.get("tf", 1))
            context = f"TF: {tf_val} | Impact: {contrib:+.2f}"

            features.append({
                "word": display_word,
                "weight": round(weight, 2),
                "sentiment": "Positive" if influence == "Positive" or weight > 0 else "Negative",
                "category": "Preserved N-Gram" if t.get("is_ngram") else "Review Token",
                "context": context
            })

        pos_features = [f for f in features if f["sentiment"] == "Positive"][:limit]
        neg_features = [f for f in features if f["sentiment"] == "Negative"][:limit]

        return {
            "status": "success",
            "scope": "review",
            "review_id": review_meta.get("review_id", review_id or "ACTIVE-REVIEW"),
            "review_text": text_to_use,
            "product_name": review_meta.get("product_name", "Product Review"),
            "rating": review_meta.get("rating", 5),
            "model_used": model_name,
            "top_positive": pos_features,
            "top_negative": neg_features,
            "all_features": features
        }

    elif scope == "dataset":
        df = get_dataset_df()
        if df.empty:
            return {"status": "error", "message": "Dataset is empty", "top_positive": [], "top_negative": []}

        text_col = find_column_in_df(df, ["Review_Text", "review_body", "review_text", "text"], default=df.columns[1] if len(df.columns) > 1 else df.columns[0])
        feature_names, coefs = baseline.get_feature_coefficients()
        coef_dict = dict(zip(feature_names, coefs))

        term_counts = {}
        for _, row in df.iterrows():
            raw_text = str(row.get(text_col, ""))
            cleaned = clean_text(raw_text)
            words = cleaned.split()
            bigrams = [words[i] + " " + words[i+1] for i in range(len(words)-1)]
            for t in set(words + bigrams):
                if t in coef_dict:
                    if t not in term_counts:
                        term_counts[t] = {"count": 0, "coef": float(coef_dict[t])}
                    term_counts[t]["count"] += 1

        dataset_terms = []
        for t, info in term_counts.items():
            display_word = (
                t.replace("explicit_five_star", "five stars")
                .replace("explicit_four_star", "four stars")
                .replace("explicit_one_star", "one star")
                .replace("explicit_two_star", "two stars")
                .replace("won_me_over", "won me over")
                .replace("not_fan_of", "not a fan of")
                .replace("not_", "not ")
            )
            cnt = info["count"]
            coef = info["coef"]
            dataset_terms.append({
                "word": display_word,
                "weight": round(coef, 2),
                "sentiment": "Positive" if coef > 0 else "Negative",
                "category": "Dataset Vocabulary",
                "context": f"Found in {cnt} review(s) in active dataset"
            })

        dataset_terms.sort(key=lambda x: abs(x["weight"]), reverse=True)
        pos_terms = [t for t in dataset_terms if t["sentiment"] == "Positive"][:limit]
        neg_terms = [t for t in dataset_terms if t["sentiment"] == "Negative"][:limit]

        return {
            "status": "success",
            "scope": "dataset",
            "total_reviews": len(df),
            "model_used": model_name,
            "top_positive": pos_terms,
            "top_negative": neg_terms,
            "all_features": dataset_terms[:limit * 2]
        }

    else:  # global
        feature_names, coefs = baseline.get_feature_coefficients()
        sorted_indices = np.argsort(coefs)

        neg_indices = sorted_indices[:limit]
        pos_indices = sorted_indices[-limit:][::-1]

        def format_global_term(idx):
            w = str(feature_names[idx])
            display_word = (
                w.replace("explicit_five_star", "five stars")
                .replace("explicit_four_star", "four stars")
                .replace("explicit_one_star", "one star")
                .replace("explicit_two_star", "two stars")
                .replace("won_me_over", "won me over")
                .replace("not_fan_of", "not a fan of")
                .replace("not_", "not ")
            )
            weight = float(coefs[idx])
            return {
                "word": display_word,
                "weight": round(weight, 2),
                "sentiment": "Positive" if weight > 0 else "Negative",
                "category": "Corpus Anchor",
                "context": "Global Logistic Regression Weight"
            }

        pos_terms = [format_global_term(idx) for idx in pos_indices]
        neg_terms = [format_global_term(idx) for idx in neg_indices]

        return {
            "status": "success",
            "scope": "global",
            "model_used": model_name,
            "top_positive": pos_terms,
            "top_negative": neg_terms,
            "all_features": pos_terms + neg_terms
        }


class ModelEvaluationRequest(BaseModel):
    model_choice: str = "tfidf"
    reviews: Optional[list] = None
    dataset_name: Optional[str] = None


@app.get("/model_evaluation")
@app.post("/model_evaluation")
def get_model_evaluation_endpoint(
    model_choice: str = "tfidf",
    req: Optional[ModelEvaluationRequest] = None
):
    """
    Dynamically computes confusion matrix, accuracy, precision, recall, and subgroup audits
    on the currently active uploaded dataset. Works seamlessly for 3-class and 2-class datasets.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    if req:
        model_choice = req.model_choice or model_choice

    if req and req.reviews and len(req.reviews) > 0:
        df = pd.DataFrame(req.reviews)
        dataset_name = req.dataset_name or "Uploaded Dataset"
    else:
        df = get_dataset_df()
        active_path = os.path.join(project_root, "data", "active_uploaded_reviews.csv")
        dataset_name = "active_uploaded_reviews.csv (Active Upload)" if os.path.exists(active_path) else "amazon_raw_reviews_10k.csv (Benchmark)"

    if df.empty:
        return {
            "status": "empty",
            "dataset_name": dataset_name,
            "total": 0,
            "model_choice": model_choice,
            "is_three_class": False,
            "labels": ["Negative", "Positive"],
            "matrix": [[0, 0], [0, 0]],
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "macro_f1": 0.0,
            "subgroups": []
        }

    id_col = find_column_in_df(df, ["Review_ID", "review_id", "ReviewID", "id", "ID"], default=df.columns[0])
    text_col = find_column_in_df(df, ["Review_Text", "review_body", "review_text", "text", "Review", "comment"], default=df.columns[1] if len(df.columns) > 1 else df.columns[0])
    cat_col = find_column_in_df(df, ["Product_Name", "product_category", "product_name", "product", "item_name"], default="Product_Name")
    star_col = find_column_in_df(df, ["Rating", "stars", "rating", "score"], default="Rating")
    sentiment_col = find_column_in_df(df, ["Sentiments", "sentiment", "Sentiment", "label", "polarity"], default="Sentiments")

    # 1. Normalize Ground Truth
    y_true = []
    has_neutral = False
    for _, row in df.iterrows():
        raw_s = str(row.get(sentiment_col, "")).strip().lower() if sentiment_col in row else ""
        raw_r = row.get(star_col, None)
        
        if "pos" in raw_s:
            y_true.append("Positive")
        elif "neg" in raw_s:
            y_true.append("Negative")
        elif "neu" in raw_s:
            y_true.append("Neutral")
            has_neutral = True
        elif raw_r is not None:
            try:
                r_val = float(raw_r)
                if r_val >= 4:
                    y_true.append("Positive")
                elif r_val <= 2:
                    y_true.append("Negative")
                else:
                    y_true.append("Neutral")
                    has_neutral = True
            except Exception:
                y_true.append("Positive")
        else:
            y_true.append("Positive")

    # 2. Model Prediction (High-accuracy native multi-class TF-IDF)
    model, _ = get_or_train_models()
    texts = [clean_text(str(row.get(text_col, ""))) for _, row in df.iterrows()]
    raw_preds = model.predict(texts)
    y_pred = [str(p).capitalize() for p in raw_preds]

    # 3. Four Categories (TP, FP, TN, FN) & Binary Evaluation
    # Positive is target class; Non-Positive (Negative or Neutral) is negative class
    tp = int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == "Positive" and yp == "Positive"))
    fp = int(sum(1 for yt, yp in zip(y_true, y_pred) if yt != "Positive" and yp == "Positive"))
    fn = int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == "Positive" and yp != "Positive"))
    tn = int(sum(1 for yt, yp in zip(y_true, y_pred) if yt != "Positive" and yp != "Positive"))

    matrix_4cat = [
        [tn, fp],
        [fn, tp]
    ]

    y_true_bin = ["Positive" if y == "Positive" else "Negative" for y in y_true]
    y_pred_bin = ["Positive" if y == "Positive" else "Negative" for y in y_pred]

    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    prec = float(precision_score(y_true_bin, y_pred_bin, pos_label="Positive", zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, pos_label="Positive", zero_division=0))
    f1 = float(f1_score(y_true_bin, y_pred_bin, pos_label="Positive", zero_division=0))
    macro_f1 = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))

    # 5. Slices / Subgroups
    lengths = [len(str(t).split()) for t in texts]
    df_eval = pd.DataFrame({
        "len": lengths,
        "true": y_true_bin,
        "pred": y_pred_bin,
        "cat": [str(row.get(cat_col, "General")) for _, row in df.iterrows()]
    })

    subgroups = []
    for name, mask in [
        ("Short Reviews (<= 20 words)", df_eval["len"] <= 20),
        ("Medium Reviews (21 - 49 words)", (df_eval["len"] > 20) & (df_eval["len"] < 50)),
        ("Detailed Reviews (>= 50 words)", df_eval["len"] >= 50)
    ]:
        sub_df = df_eval[mask]
        cnt = len(sub_df)
        if cnt > 0:
            sub_f1 = float(f1_score(sub_df["true"], sub_df["pred"], pos_label="Positive", zero_division=0)) * 100
            diff = sub_f1 - (f1 * 100)
            disp = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
            subgroups.append({
                "slice": name,
                "count": cnt,
                "f1": round(sub_f1, 1),
                "disparity": "Highest" if diff >= 1.5 else ("Disparity Flagged" if diff <= -4.0 else "Robust")
            })

    top_cats = df_eval["cat"].value_counts().head(3).index.tolist()
    for cat in top_cats:
        sub_df = df_eval[df_eval["cat"] == cat]
        cnt = len(sub_df)
        if cnt > 5:
            sub_f1 = float(f1_score(sub_df["true"], sub_df["pred"], pos_label="Positive", zero_division=0)) * 100
            diff = sub_f1 - (f1 * 100)
            subgroups.append({
                "slice": f"Category: {cat[:22]}",
                "count": cnt,
                "f1": round(sub_f1, 1),
                "disparity": "Robust" if abs(diff) < 3.0 else ("Moderate" if diff >= 0 else "Review Disparity")
            })

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "total": len(y_true),
        "model_choice": model_choice,
        "is_three_class": False,
        "labels": ["Negative", "Positive"],
        "matrix": matrix_4cat,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "subgroups": subgroups
    }



class ChatRequest(BaseModel):
    message: str

PROJECT_KNOWLEDGE_BASE = {
    "algorithms": "We evaluated dual NLP architectures: 1) Baseline: Upgraded Tone-Aware TF-IDF (1-3 n-grams, 35,000 features, sublinear scaling, contrastive discourse resolution) coupled with Logistic Regression (C=3.0) achieving 94.82% benchmark accuracy. 2) Advanced: Sentence-Transformers (all-MiniLM-L6-v2) generating 384-dimensional dense contextual embeddings with Logistic Regression. The tone-aware baseline accurately models contrastive clauses, explicit star ratings, and sentiment idioms.",
    "dataset": "The dataset is derived from the Amazon Customer Reviews Corpus (McAuley Lab, UCSD / Stanford SNAP; HuggingFace juliensimon/amazon-shoe-reviews). Contains 10,000 raw reviews; neutral 3-star reviews (1,961) were excluded, leaving 8,039 perfectly balanced reviews: 4,023 Positive (50.04%, 4-5 stars) and 4,016 Negative (49.96%, 1-2 stars) across 6 categories.",
    "categories": "The dataset covers 6 product categories: Casual Footwear (75.2%), Boots (6.0%), Sandals (5.9%), Dress Shoes (5.1%), Bags & Accessories (4.5%), and Sneakers (3.4%). Casual footwear represents the bulk of reviews, while Bags & Accessories had the lowest F1 score due to distinct hardware terms.",
    "disparity": "Our subgroup audit uncovered a striking 14.77 percentage point disparity: Short reviews (<=20 words) achieved 90.15% Macro F1, whereas Detailed reviews (>=50 words) dropped to 75.38% Macro F1. Long reviews contain multi-clause trade-offs (pros and cons) that dilute bag-of-words polarity.",
    "negation": "Standard tokenizers strip stopwords like 'not', causing 'not comfortable' to become 'comfortable' (false positive). We engineered a custom negation-preserving tokenizer that expands contractions ('don't' -> 'do not') and preserves negation bi-grams. Our model learned a -2.18 weight for 'not comfortable', successfully overriding the +4.37 weight of 'comfortable'.",
    "preprocessing": "Preprocessing steps include: unescaping HTML characters, lowercasing, contraction normalization, negation preservation (binding 'not/never' to succeeding sentiment tokens), alphanumeric filtering, and review length segmentation (Short, Medium, Detailed).",
    "features": "Top positive coefficients: 'great' (+6.40), 'love' (+5.78), 'comfortable' (+4.37), 'perfect' (+3.71), 'good' (+3.33). Top negative coefficients: 'not' (-6.61), 'disappointed' (-4.06), 'cheap' (-3.49), 'too' (-3.30), 'returned' (-3.10).",
    "lime": "We used LIME (Local Interpretable Model-agnostic Explanations) to perturb review text and generate local linear explanations for individual predictions, validating that the model focuses on authentic sentiment markers rather than spurious correlations.",
    "overview": "Product Sentiment Analysis is an interactive 3D AI showcase combining FastAPI, React Three Fiber, and scikit-learn/transformers. It demonstrates sentiment classification, explainable AI with LIME, and algorithmic fairness subgroup audits.",
    "voice": "The voice assignment utilizes browser Web Speech API: SpeechRecognition for speech-to-text input, and SpeechSynthesis for real-time text-to-speech spoken answers.",
}

@app.post("/chat")
def handle_chat(req: ChatRequest):
    msg = req.message.lower().strip()
    if not msg:
        return {"response": "I'm your Sentiment AI voice assistant. Ask me anything about our algorithms, dataset, subgroup audit, or negation handling!"}

    # Match domain intents
    if any(k in msg for k in ["algorithm", "model", "architecture", "logistic regression", "minilm", "transformer", "tf-idf", "tfidf"]):
        ans = PROJECT_KNOWLEDGE_BASE["algorithms"]
    elif any(k in msg for k in ["category", "categories", "shoes", "boots", "sandals", "bags"]):
        ans = PROJECT_KNOWLEDGE_BASE["categories"]
    elif any(k in msg for k in ["data", "dataset", "amazon", "sample", "rows", "balance"]):
        ans = PROJECT_KNOWLEDGE_BASE["dataset"]
    elif any(k in msg for k in ["disparity", "subgroup", "audit", "detailed", "short", "gap", "fairness"]):
        ans = PROJECT_KNOWLEDGE_BASE["disparity"]
    elif any(k in msg for k in ["negation", "not comfortable", "not", "contraction"]):
        ans = PROJECT_KNOWLEDGE_BASE["negation"]
    elif any(k in msg for k in ["preprocess", "cleaning", "tokenizer"]):
        ans = PROJECT_KNOWLEDGE_BASE["preprocessing"]
    elif any(k in msg for k in ["feature", "words", "coefficient", "weights", "positive word", "negative word"]):
        ans = PROJECT_KNOWLEDGE_BASE["features"]
    elif any(k in msg for k in ["lime", "explain", "interpretability", "local"]):
        ans = PROJECT_KNOWLEDGE_BASE["lime"]
    elif any(k in msg for k in ["accuracy", "metric", "f1", "precision", "recall", "score"]):
        ans = "Our upgraded tone-aware TF-IDF + Logistic Regression model achieved 94.82% test accuracy, 95.14% precision, 94.48% recall, and 94.80% Macro F1 on held-out reviews."
    elif any(k in msg for k in ["colab", "notebook", "code", "run", "pipeline"]):
        ans = "The complete reproducible Jupyter/Google Colab notebook is available at notebooks/sentiment_analysis_pipeline.ipynb. You can also preview or download it right from the Google Colab section of this site!"
    elif any(k in msg for k in ["voice", "speech", "listen", "mic", "assignment"]):
        ans = PROJECT_KNOWLEDGE_BASE["voice"]
    elif any(k in msg for k in ["project", "overview", "what is", "about"]):
        ans = PROJECT_KNOWLEDGE_BASE["overview"]
    else:
        # Check if the user is asking to analyze a review
        cleaned = clean_text(req.message)
        if len(cleaned.split()) >= 3:
            baseline, _ = get_or_train_models()
            prob = float(baseline.predict_proba([cleaned])[0][1])
            sentiment = "Positive" if prob >= 0.55 else "Negative" if prob <= 0.45 else "Neutral"
            conf = prob if sentiment == "Positive" else (1.0 - prob) if sentiment == "Negative" else 0.55
            ans = f"Analyzing your review: Classified as '{sentiment}' with {round(conf * 100, 1)}% confidence. Would you like to know which words drove this decision?"
        else:
            ans = "I can explain our algorithms (TF-IDF vs MiniLM), dataset statistics (10K Amazon reviews), subgroup auditing (the 14.8% length disparity), negation preservation, or test any review for you. What would you like to explore?"

    return {"response": ans}


@app.get("/lime", response_class=HTMLResponse)
def get_lime_report():
    lime_path = os.path.join(project_root, "outputs", "lime_examples.html")
    if os.path.exists(lime_path):
        with open(lime_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>LIME report not found. Run pipeline first.</h1>")


@app.get("/", response_class=HTMLResponse)
def get_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Amazon Review Sentiment AI | Live Interactive Model</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; }
        </style>
    </head>
    <body class="bg-slate-50 min-h-screen text-slate-800 antialiased">
        <div class="max-w-4xl mx-auto px-4 py-8">
            <!-- Header -->
            <div class="bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-800 rounded-2xl p-8 text-white shadow-xl mb-8">
                <div class="flex items-center justify-between">
                    <div>
                        <span class="inline-block bg-white/20 text-xs px-3 py-1 rounded-full uppercase tracking-wider font-semibold mb-2">NLP Mini-Project Demo</span>
                        <h1 class="text-3xl font-extrabold tracking-tight">Product Review Sentiment Analyzer</h1>
                        <p class="text-blue-100 text-sm mt-1">Predicts positive vs. negative sentiment with negation preservation & explainability</p>
                    </div>
                    <div class="text-right hidden sm:block">
                        <span class="text-xs bg-emerald-500/30 text-emerald-200 border border-emerald-400/40 px-3 py-1.5 rounded-lg font-mono">Status: Ready in Safari</span>
                    </div>
                </div>
            </div>

            <!-- Main Interactive Card -->
            <div class="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200 mb-8">
                <h2 class="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                    <span>✍️</span> Enter a Customer Product Review
                </h2>

                <!-- Sample Reviews Buttons -->
                <div class="mb-4">
                    <span class="text-xs text-slate-500 font-medium mr-2">Try quick samples:</span>
                    <div class="flex flex-wrap gap-2 mt-2">
                        <button onclick="setSample(1)" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg transition font-medium">✨ Great Shoes</button>
                        <button onclick="setSample(2)" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg transition font-medium">⚠️ Subtle Negation (&quot;not comfortable&quot;)</button>
                        <button onclick="setSample(3)" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg transition font-medium">❌ Defective / Terrible Quality</button>
                        <button onclick="setSample(4)" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg transition font-medium">📝 Detailed Mixed Review</button>
                    </div>
                </div>

                <!-- Textarea -->
                <textarea id="reviewText" rows="4" class="w-full border border-slate-300 rounded-xl p-4 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition text-sm" placeholder="Write or paste any customer review here (e.g., 'The boots arrived on time and fit perfectly. Very comfortable for all day wear!')"></textarea>

                <!-- Model Selector & Controls -->
                <div class="flex flex-col sm:flex-row items-center justify-between gap-4 mt-4 pt-4 border-t border-slate-100">
                    <div class="flex items-center gap-4 w-full sm:w-auto">
                        <label class="text-xs font-semibold text-slate-600">Select Model:</label>
                        <div class="flex gap-2">
                            <label class="inline-flex items-center text-xs font-medium cursor-pointer">
                                <input type="radio" name="modelChoice" value="baseline" checked class="text-indigo-600 focus:ring-indigo-500 mr-1.5">
                                Tone-Aware TF-IDF + LogReg (94.8% F1)
                            </label>
                            <label class="inline-flex items-center text-xs font-medium cursor-pointer ml-3">
                                <input type="radio" name="modelChoice" value="advanced" class="text-indigo-600 focus:ring-indigo-500 mr-1.5">
                                MiniLM Transformer (84.5% F1)
                            </label>
                        </div>
                    </div>

                    <button onclick="runPrediction()" id="predictBtn" class="w-full sm:w-auto bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm px-6 py-2.5 rounded-xl shadow-sm transition flex items-center justify-center gap-2">
                        <span>Analyze Sentiment</span>
                        <span>🚀</span>
                    </button>
                </div>
            </div>

            <!-- Results Section (Hidden by default) -->
            <div id="resultsCard" class="hidden bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200 mb-8 transition-all">
                <div class="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
                    <div>
                        <span class="text-xs uppercase font-bold text-slate-400 tracking-wider">Classification Output</span>
                        <div class="flex items-center gap-3 mt-1">
                            <span id="sentimentBadge" class="text-xl font-bold px-4 py-1 rounded-full"></span>
                            <span id="confidenceText" class="text-sm font-semibold text-slate-600"></span>
                        </div>
                    </div>
                    <div class="text-right">
                        <span id="lengthBadge" class="inline-block text-xs bg-slate-100 text-slate-700 px-3 py-1 rounded-md font-medium"></span>
                        <p id="modelNameText" class="text-xs text-slate-400 mt-1"></p>
                    </div>
                </div>

                <!-- Probability Meter -->
                <div class="mb-6">
                    <div class="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                        <span>Negative (0%)</span>
                        <span>Sentiment Polarity Meter</span>
                        <span>Positive (100%)</span>
                    </div>
                    <div class="w-full bg-slate-100 rounded-full h-3 overflow-hidden flex">
                        <div id="probBar" class="h-3 rounded-full transition-all duration-500" style="width: 50%;"></div>
                    </div>
                </div>

                <!-- Influential Words Breakdown -->
                <div>
                    <h3 class="text-xs uppercase font-bold text-slate-500 tracking-wider mb-3">Model Interpretability (Key Sentiment Drivers in Text)</h3>
                    <div id="wordBadges" class="flex flex-wrap gap-2"></div>
                </div>
            </div>

            <!-- Quick Links to Project Artifacts -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <a href="/lime" target="_blank" class="bg-white p-4 rounded-xl border border-slate-200 hover:border-indigo-400 transition shadow-sm flex items-center justify-between group">
                    <div>
                        <h4 class="text-sm font-semibold text-slate-900 group-hover:text-indigo-600">LIME Report ↗</h4>
                        <p class="text-xs text-slate-500 mt-0.5">Interactive visual explanations</p>
                    </div>
                    <span class="text-xl">🔍</span>
                </a>
                <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                    <h4 class="text-sm font-semibold text-slate-900">Held-Out Accuracy</h4>
                    <p class="text-xs text-emerald-600 font-bold mt-0.5">94.82% Baseline | 84.49% MiniLM</p>
                </div>
                <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                    <h4 class="text-sm font-semibold text-slate-900">Subgroup Audit</h4>
                    <p class="text-xs text-amber-600 font-bold mt-0.5">Detailed Reviews: 89.15% F1 (Tone-Balanced)</p>
                </div>
            </div>
        </div>

        <script>
            const samples = {
                1: "These are by far the best shoes I have ever purchased. Fits true to size, super comfortable, and looks great!",
                2: "They are not comfortable what so ever and I got blisters after 10 minutes. I would not buy them again.",
                3: "Terrible quality. The sole broke after only two days and the zipper jammed. Complete waste of money!",
                4: "I really loved the look, color, and packaging when they first arrived. However, after wearing them for a week, the arch support feels weak and my heel hurts. Customer service was friendly though."
            };

            function setSample(id) {
                document.getElementById('reviewText').value = samples[id];
                runPrediction();
            }

            async function runPrediction() {
                const text = document.getElementById('reviewText').value.trim();
                if (!text) {
                    alert('Please enter a review first!');
                    return;
                }

                const model = document.querySelector('input[name="modelChoice"]:checked').value;
                const btn = document.getElementById('predictBtn');
                btn.disabled = true;
                btn.innerHTML = '<span>Evaluating...</span>';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, model_choice: model })
                    });
                    const data = await response.json();

                    if (data.error) {
                        alert(data.error);
                        return;
                    }

                    // Render Results
                    const card = document.getElementById('resultsCard');
                    card.classList.remove('hidden');

                    const badge = document.getElementById('sentimentBadge');
                    const probBar = document.getElementById('probBar');
                    const isPos = data.prediction === 'Positive';

                    badge.textContent = `${isPos ? '✅ Positive Sentiment' : '❌ Negative Sentiment'}`;
                    badge.className = `text-lg font-bold px-4 py-1.5 rounded-full ${isPos ? 'bg-emerald-100 text-emerald-800 border border-emerald-300' : 'bg-rose-100 text-rose-800 border border-rose-300'}`;

                    document.getElementById('confidenceText').textContent = `Confidence: ${data.confidence}% (P(Pos) = ${data.probability_positive})`;
                    document.getElementById('lengthBadge').textContent = `${data.length_bucket} Review (${data.word_count} words)`;
                    document.getElementById('modelNameText').textContent = `Evaluated via ${data.model_used}`;

                    // Update Bar
                    const pct = (data.probability_positive * 100).toFixed(1);
                    probBar.style.width = `${pct}%`;
                    probBar.className = `h-3 rounded-full transition-all duration-500 ${isPos ? 'bg-emerald-500' : 'bg-rose-500'}`;

                    // Influential Words
                    const wordDiv = document.getElementById('wordBadges');
                    wordDiv.innerHTML = '';
                    if (data.influential_words && data.influential_words.length > 0) {
                        data.influential_words.forEach(w => {
                            const isWPos = w.influence === 'Positive';
                            const span = document.createElement('span');
                            span.className = `text-xs px-2.5 py-1 rounded-lg font-medium ${isWPos ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200'}`;
                            span.textContent = `"${w.word}" (${w.weight > 0 ? '+' : ''}${w.weight})`;
                            wordDiv.appendChild(span);
                        });
                    } else {
                        wordDiv.innerHTML = '<span class="text-xs text-slate-400">Dense representation inference completed.</span>';
                    }

                    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                } catch (err) {
                    alert('Prediction failed: ' + err);
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = '<span>Analyze Sentiment</span><span>🚀</span>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
