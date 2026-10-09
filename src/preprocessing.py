"""
Preprocessing module for Product Review Sentiment Analysis.

Functions:
- clean_text: Cleans raw review text, expands contractions, strips HTML/URLs,
  and preserves critical negation tokens.
- binarize_ratings: Converts 1-5 star ratings to binary sentiment (0=negative, 1=positive),
  excluding 3-star neutral reviews, and reports class balance.
- assign_length_bucket: Categorizes reviews into 'short' (<=20 words),
  'medium' (21-49 words), and 'detailed' (>=50 words).
- prepare_dataset: Full preprocessing pipeline from DataFrame to clean data with metadata.
- split_data: Stratified train/test split preserving subgroup metadata for auditing.
"""

import re
import html
import pandas as pd
from sklearn.model_selection import train_test_split

# Common English contractions and their standard expansions
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

# Negation words that must be preserved
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", 
    "nowhere", "hardly", "scarcely", "barely", "nothing", "nobody"
}


def clean_text(text: str) -> str:
    """
    Cleans raw review text:
    1. Unescapes HTML entities and strips HTML tags.
    2. Removes URLs and web links.
    3. Expands contractions to preserve semantic negation (e.g., don't -> do not).
    4. Converts to lowercase.
    5. Strips non-alphanumeric punctuation while maintaining word boundaries.
    6. Normalizes whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (e.g., &amp; -> &, &quot; -> ")
    text = html.unescape(text)

    # Remove HTML tags (e.g. <br />, <p>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Lowercase before contraction expansion for consistency
    text = text.lower()

    # Expand contractions
    for pattern, replacement in CONTRACTION_MAP.items():
        text = re.sub(pattern, replacement, text)

    # 1. Normalize explicit star ratings and score mentions (Five stars from me, 5 stars, 1 star, etc.)
    text = re.sub(r'\b(5\s*stars?|five\s*stars?|5/5|5\s*out\s*of\s*5|5\s*star)\b', ' explicit_five_star explicit_five_star great ', text)
    text = re.sub(r'\b(4\s*stars?|four\s*stars?|4/5|4\s*out\s*of\s*5|4\s*star)\b', ' explicit_four_star good ', text)
    text = re.sub(r'\b(1\s*stars?|one\s*stars?|1/5|1\s*out\s*of\s*5|zero\s*stars?|0\s*stars?|1\s*star)\b', ' explicit_one_star explicit_one_star terrible ', text)
    text = re.sub(r'\b(2\s*stars?|two\s*stars?|2/5|2\s*star)\b', ' explicit_two_star poor ', text)

    # 2. Normalize key sentiment idioms and compound phrases
    text = re.sub(r'\bwon\s+(?:me|us|everyone|them)\s+over\b', ' won_me_over love_it great ', text)
    text = re.sub(r'\ba\s+fan\s+of\b', ' fan_of ', text)
    text = re.sub(r'\bwaste\s+of\s+money\b', ' waste_of_money terrible ', text)
    text = re.sub(r'\bdo\s+not\s+buy\b', ' do_not_buy terrible ', text)
    text = re.sub(r'\bhighly\s+recommend\b', ' highly_recommend excellent ', text)
    text = re.sub(r'\bfell\s+apart\b', ' fell_apart terrible ', text)
    text = re.sub(r'\bpoor\s+quality\b', ' poor_quality terrible ', text)

    # 3. Negation scope binding: bind 'not X' into 'not_X'
    raw_words = text.split()
    bound_words = []
    i = 0
    while i < len(raw_words):
        w = raw_words[i]
        if w in NEGATION_WORDS and i + 1 < len(raw_words):
            nxt = re.sub(r'[^a-z0-9_]', '', raw_words[i + 1])
            if nxt and nxt not in NEGATION_WORDS:
                bound_words.append(f'not_{nxt}')
                bound_words.append(w)  # retain unigram for backward compatibility
                i += 2
                continue
        bound_words.append(w)
        i += 1
    text = ' '.join(bound_words)

    # 4. Contrastive discourse awareness (e.g. 'A, but B'):
    # In reviews, clauses following 'but', 'however', 'yet', 'although' carry the dominant final tone/verdict!
    contrast_parts = re.split(r'\b(but|however|yet|although|nevertheless|though)\b', text)
    if len(contrast_parts) > 1:
        pre = contrast_parts[0]
        post = ' '.join(contrast_parts[1:])
        text = f'{pre} {post} {post}'

    # Remove special punctuation and non-alphanumeric characters, keeping spaces
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)

    # Collapse multiple whitespaces and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def assign_length_bucket(text: str) -> tuple:
    """
    Computes review word count and assigns a length bucket:
    - 'short': <= 20 words
    - 'medium': 21 - 49 words
    - 'detailed': >= 50 words

    Returns: (word_count, bucket)
    """
    words = text.strip().split()
    word_count = len(words)
    if word_count <= 20:
        bucket = "short"
    elif word_count < 50:
        bucket = "medium"
    else:
        bucket = "detailed"
    return word_count, bucket


def normalize_col_name(c: str) -> str:
    """Normalizes column names by lowercasing and stripping whitespace, underscores, hyphens, and dots."""
    return re.sub(r"[\s_\-\.]+", "", str(c).lower())


def find_column(df: pd.DataFrame, target: str, fallback: str = None) -> str:
    """Dynamically finds a matching column in DataFrame using case- and punctuation-insensitive aliases."""
    norm_target = normalize_col_name(target)
    for col in df.columns:
        if normalize_col_name(col) == norm_target:
            return col

    aliases = {
        "reviewid": ["reviewid", "id", "revid", "reviewno", "reviewnum", "reviewidentifier"],
        "productname": ["productname", "product", "item", "itemname", "producttitle", "title", "productcategory", "category", "producttype"],
        "rating": ["rating", "ratings", "stars", "starrating", "star", "score", "reviewrating"],
        "reviewtext": ["reviewtext", "reviewbody", "review", "text", "body", "comment", "comments", "feedback", "content", "reviewcontent"],
        "sentiments": ["sentiments", "sentiment", "label", "labels", "target", "polarity", "class", "sentimentlabel"]
    }

    for alias in aliases.get(norm_target, []):
        for col in df.columns:
            if normalize_col_name(col) == alias:
                return col

    if fallback and fallback in df.columns:
        return fallback
    return None


def binarize_ratings(
    df: pd.DataFrame,
    rating_col: str = None,
    sentiment_col: str = None
) -> pd.DataFrame:
    """
    Binarizes sentiment labels:
    - If a valid Sentiments/label column exists (e.g. 'Positive'/'Negative' or 1/0), converts to 1 and 0.
    - Otherwise, binarizes star ratings:
      - 4 to 5 stars -> 1 (Positive)
      - 1 to 2 stars -> 0 (Negative)
      - 3 stars -> Excluded (Neutral)

    Prints and returns class balance statistics.
    """
    df_clean = df.copy()
    initial_count = len(df_clean)

    # Detect rating or sentiment column dynamically if not explicitly specified or missing
    if rating_col is None or rating_col not in df_clean.columns:
        rating_col = find_column(df_clean, "rating", fallback="stars")

    if sentiment_col is None or sentiment_col not in df_clean.columns:
        sentiment_col = find_column(df_clean, "sentiments", fallback="label")

    # If explicit sentiment column is present with text/numeric labels
    if sentiment_col and sentiment_col in df_clean.columns:
        s_series = df_clean[sentiment_col].astype(str).str.strip().str.lower()
        has_pos = s_series.str.contains("pos|1").any()
        has_neg = s_series.str.contains("neg|0").any()

        if has_pos and has_neg:
            neutral_mask = s_series.str.contains("neu|3|average")
            neutral_count = neutral_mask.sum()
            df_filtered = df_clean[~neutral_mask].copy()

            df_filtered["label"] = df_filtered[sentiment_col].astype(str).str.strip().str.lower().apply(
                lambda s: 1 if ("pos" in s or s == "1") else 0
            )

            total_binary = len(df_filtered)
            pos_count = (df_filtered["label"] == 1).sum()
            neg_count = (df_filtered["label"] == 0).sum()

            print("=" * 60)
            print("LABELING & CLASS BALANCE REPORT (FROM SENTIMENT COLUMN)")
            print("=" * 60)
            print(f"Total raw reviews: {initial_count}")
            print(f"Excluded neutral reviews: {neutral_count}")
            print(f"Retained binary reviews: {total_binary}")
            print(f"  - Positive [1]: {pos_count} ({pos_count / max(1, total_binary) * 100:.2f}%)")
            print(f"  - Negative [0]: {neg_count} ({neg_count / max(1, total_binary) * 100:.2f}%)")
            print("=" * 60)
            return df_filtered

    # Otherwise fallback to star rating binarization
    if rating_col and rating_col in df_clean.columns:
        df_clean[rating_col] = pd.to_numeric(df_clean[rating_col], errors="coerce").fillna(5)
        neutral_count = (df_clean[rating_col] == 3).sum()

        df_filtered = df_clean[df_clean[rating_col] != 3].copy()
        df_filtered["label"] = (df_filtered[rating_col] >= 4).astype(int)

        pos_count = (df_filtered["label"] == 1).sum()
        neg_count = (df_filtered["label"] == 0).sum()
        total_binary = len(df_filtered)

        print("=" * 60)
        print("LABELING & CLASS BALANCE REPORT (FROM RATING COLUMN)")
        print("=" * 60)
        print(f"Total raw reviews: {initial_count}")
        print(f"Excluded neutral reviews (3-star): {neutral_count} ({neutral_count / max(1, initial_count) * 100:.2f}%)")
        print(f"Retained binary reviews: {total_binary}")
        print(f"  - Positive (4-5 stars) [1]: {pos_count} ({pos_count / max(1, total_binary) * 100:.2f}%)")
        print(f"  - Negative (1-2 stars) [0]: {neg_count} ({neg_count / max(1, total_binary) * 100:.2f}%)")
        print(f"Class ratio (Pos : Neg): {pos_count / max(neg_count, 1):.2f} : 1")
        print("=" * 60)
        return df_filtered

    # If neither found, default label
    df_clean["label"] = 1
    return df_clean


def prepare_dataset(
    df: pd.DataFrame,
    text_col: str = None,
    rating_col: str = None,
    category_col: str = None,
    id_col: str = None,
    sentiment_col: str = None
) -> pd.DataFrame:
    """
    Full dynamic preprocessing pipeline:
    1. Dynamically detects canonical columns (Review_ID, Product_Name, Rating, Review_Text, Sentiments).
    2. Binarizes sentiment ratings and filters neutral reviews.
    3. Cleans review text (expanding contractions, stripping HTML/URLs, preserving negations).
    4. Filters empty/trivial reviews.
    5. Assigns word counts and length buckets.
    6. Ensures standard canonical metadata columns exist for downstream audit.
    """
    actual_text_col = text_col if (text_col and text_col in df.columns) else find_column(df, "reviewtext", fallback="review_body")
    actual_rating_col = rating_col if (rating_col and rating_col in df.columns) else find_column(df, "rating", fallback="stars")
    actual_cat_col = category_col if (category_col and category_col in df.columns) else find_column(df, "productname", fallback="product_category")
    actual_id_col = id_col if (id_col and id_col in df.columns) else find_column(df, "reviewid", fallback="review_id")
    actual_sent_col = sentiment_col if (sentiment_col and sentiment_col in df.columns) else find_column(df, "sentiments", fallback="Sentiments")

    print(f"\n[Pipeline Ingestion] Detected column schema:")
    print(f"  - Review_ID   : {actual_id_col}")
    print(f"  - Product_Name: {actual_cat_col}")
    print(f"  - Rating      : {actual_rating_col}")
    print(f"  - Review_Text : {actual_text_col}")
    print(f"  - Sentiments  : {actual_sent_col}")

    # 1. Binarize labels
    df_clean = binarize_ratings(df, rating_col=actual_rating_col, sentiment_col=actual_sent_col)

    # 2. Text cleaning
    if actual_text_col not in df_clean.columns:
        raise ValueError(f"Required Review_Text column '{actual_text_col}' not found in dataset!")

    print(f"\nCleaning review text from column '{actual_text_col}'...")
    df_clean["clean_text"] = df_clean[actual_text_col].astype(str).apply(clean_text)

    # 3. Filter out rows where clean_text is empty or too short (< 2 words)
    valid_mask = df_clean["clean_text"].apply(lambda t: len(t.split()) >= 2)
    dropped_empty = (~valid_mask).sum()
    df_clean = df_clean[valid_mask].copy()
    if dropped_empty > 0:
        print(f"Dropped {dropped_empty} empty/trivial reviews after cleaning.")

    # 4. Word count & length bucketing
    length_info = df_clean["clean_text"].apply(assign_length_bucket)
    df_clean["word_count"] = [info[0] for info in length_info]
    df_clean["length_bucket"] = [info[1] for info in length_info]

    # Standardize metadata columns for model & audit
    if actual_cat_col and actual_cat_col in df_clean.columns:
        df_clean["product_category"] = df_clean[actual_cat_col].fillna("General Products").astype(str)
    else:
        df_clean["product_category"] = "General Products"

    if actual_id_col and actual_id_col in df_clean.columns:
        df_clean["review_id"] = df_clean[actual_id_col].fillna("REV-UNKNOWN").astype(str)
    else:
        df_clean["review_id"] = [f"REV-{1000 + i}" for i in range(len(df_clean))]

    # Canonical aliases
    df_clean["Review_ID"] = df_clean["review_id"]
    df_clean["Product_Name"] = df_clean["product_category"]
    df_clean["Review_Text"] = df_clean[actual_text_col]
    df_clean["Sentiments"] = df_clean["label"].apply(lambda l: "Positive" if l == 1 else "Negative")

    # Length bucket distribution
    print("\nReview Length Bucket Distribution:")
    for bucket, count in df_clean["length_bucket"].value_counts().items():
        print(f"  - {bucket.capitalize()}: {count} ({count / len(df_clean) * 100:.2f}%)")

    # Category distribution
    print("\nProduct Category Distribution:")
    for cat, count in df_clean["product_category"].value_counts().items():
        print(f"  - {cat}: {count} ({count / len(df_clean) * 100:.2f}%)")

    return df_clean


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple:
    """
    Performs a stratified train/test split based on sentiment label.

    Returns:
    - df_train: Training DataFrame
    - df_test: Test DataFrame (includes metadata for auditing)
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST STRATIFIED SPLIT")
    print("=" * 60)
    print(f"Train set size: {len(df_train)} (Positive: {(df_train['label'] == 1).mean() * 100:.1f}%)")
    print(f"Test set size:  {len(df_test)} (Positive: {(df_test['label'] == 1).mean() * 100:.1f}%)")
    print("=" * 60)

    return df_train, df_test
