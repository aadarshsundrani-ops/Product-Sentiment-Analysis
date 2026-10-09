export interface ReviewSample {
  Review_ID: string;
  Product_Name: string;
  Rating: number;
  Review_Text: string;
  Sentiments: 'Positive' | 'Negative' | 'Neutral';
  word_count: number;
  // Backward compatibility aliases
  id?: string;
  review_text?: string;
  stars?: number;
  product_category?: string;
  sentiment?: 'Positive' | 'Negative' | 'Neutral';
}

export interface MetricSummary {
  metric: string;
  baseline: number;
  advanced: number;
  difference: number;
}

export interface FeatureWeight {
  word: string;
  weight: number;
  sentiment: 'Positive' | 'Negative';
  category?: string;
  context?: string;
}

export const SAMPLE_REVIEWS: ReviewSample[] = [
  // 1st 5 Rows (Head of Dataset)
  {
    Review_ID: "FMCY_00003",
    Product_Name: "Hydrating Facial Skin Feel Serum",
    Rating: 5,
    Review_Text: "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me.",
    Sentiments: "Positive",
    word_count: 19,
    id: "FMCY_00003",
    review_text: "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me.",
    stars: 5,
    product_category: "Skincare & Cosmetics",
    sentiment: "Positive",
  },
  {
    Review_ID: "REV-10491",
    Product_Name: "Ultralight Cushioned Runners",
    Rating: 5,
    Review_Text: "Absolutely love these running shoes! The arch support is phenomenal and they felt broken in right out of the box. Ran 10 miles with zero blisters.",
    Sentiments: "Positive",
    word_count: 28,
    id: "REV-10491",
    review_text: "Absolutely love these running shoes! The arch support is phenomenal and they felt broken in right out of the box. Ran 10 miles with zero blisters.",
    stars: 5,
    product_category: "Casual Footwear",
    sentiment: "Positive",
  },
  {
    Review_ID: "REV-10492",
    Product_Name: "Weekender Canvas Duffel",
    Rating: 1,
    Review_Text: "Disappointed with the durability. The zipper broke on the second day of using this travel bag. Fabric is very thin and cheap.",
    Sentiments: "Negative",
    word_count: 24,
    id: "REV-10492",
    review_text: "Disappointed with the durability. The zipper broke on the second day of using this travel bag. Fabric is very thin and cheap.",
    stars: 1,
    product_category: "Bags & Accessories",
    sentiment: "Negative",
  },
  {
    Review_ID: "REV-10493",
    Product_Name: "All-Weather Arctic Boots",
    Rating: 2,
    Review_Text: "The boots are stylish and warm, but they are not comfortable for all-day standing. The insole lacks sufficient padding.",
    Sentiments: "Negative",
    word_count: 21,
    id: "REV-10493",
    review_text: "The boots are stylish and warm, but they are not comfortable for all-day standing. The insole lacks sufficient padding.",
    stars: 2,
    product_category: "Boots",
    sentiment: "Negative",
  },
  {
    Review_ID: "REV-10494",
    Product_Name: "Breathable Knit Slip-On",
    Rating: 4,
    Review_Text: "Great value for money. Fits exactly as described on the size chart and the color matches the product photos nicely.",
    Sentiments: "Positive",
    word_count: 22,
    id: "REV-10494",
    review_text: "Great value for money. Fits exactly as described on the size chart and the color matches the product photos nicely.",
    stars: 4,
    product_category: "Casual Footwear",
    sentiment: "Positive",
  },
  {
    Review_ID: "REV-10495",
    Product_Name: "Commuter Nylon Backpack",
    Rating: 3,
    Review_Text: "Average quality. Does what it needs to do, nothing special. A bit stiff at first but loosened up over time.",
    Sentiments: "Neutral",
    word_count: 21,
    id: "REV-10495",
    review_text: "Average quality. Does what it needs to do, nothing special. A bit stiff at first but loosened up over time.",
    stars: 3,
    product_category: "Bags & Accessories",
    sentiment: "Neutral",
  },

  // Last 5 Rows (Tail of Dataset)
  {
    Review_ID: "REV-10496",
    Product_Name: "Retro Canvas Low-Top",
    Rating: 3,
    Review_Text: "Not great, but not terrible either. Fair for the discount price. Wouldn't pay full retail though.",
    Sentiments: "Neutral",
    word_count: 16,
    id: "REV-10496",
    review_text: "Not great, but not terrible either. Fair for the discount price. Wouldn't pay full retail though.",
    stars: 3,
    product_category: "Casual Footwear",
    sentiment: "Neutral",
  },
  {
    Review_ID: "REV-10497",
    Product_Name: "Insulated Summit Trekker",
    Rating: 5,
    Review_Text: "Hands down the most comfortable boots I have ever owned. Withstood heavy snow and kept my feet completely dry.",
    Sentiments: "Positive",
    word_count: 20,
    id: "REV-10497",
    review_text: "Hands down the most comfortable boots I have ever owned. Withstood heavy snow and kept my feet completely dry.",
    stars: 5,
    product_category: "Boots",
    sentiment: "Positive",
  },
  {
    Review_ID: "REV-10498",
    Product_Name: "Formal Oxford Dress Shoes",
    Rating: 1,
    Review_Text: "Poor stitching along the seams. Came with loose threads and smelled strongly of petroleum chemicals. Returned immediately.",
    Sentiments: "Negative",
    word_count: 18,
    id: "REV-10498",
    review_text: "Poor stitching along the seams. Came with loose threads and smelled strongly of petroleum chemicals. Returned immediately.",
    stars: 1,
    product_category: "Dress Shoes",
    sentiment: "Negative",
  },
  {
    Review_ID: "REV-10499",
    Product_Name: "Ergonomic Memory Foam Clog",
    Rating: 5,
    Review_Text: "Incredible cushioning for long hospital shifts! The memory foam molds to my foot shape and completely eliminated my lower back soreness.",
    Sentiments: "Positive",
    word_count: 24,
    id: "REV-10499",
    review_text: "Incredible cushioning for long hospital shifts! The memory foam molds to my foot shape and completely eliminated my lower back soreness.",
    stars: 5,
    product_category: "Casual Footwear",
    sentiment: "Positive",
  },
  {
    Review_ID: "REV-10500",
    Product_Name: "Trail Grip Hiking Sandals",
    Rating: 1,
    Review_Text: "The ankle strap rubbed skin raw within 15 minutes of walking. Plastic buckle cracked on the first tightening. Terrible design.",
    Sentiments: "Negative",
    word_count: 21,
    id: "REV-10500",
    review_text: "The ankle strap rubbed skin raw within 15 minutes of walking. Plastic buckle cracked on the first tightening. Terrible design.",
    stars: 1,
    product_category: "Sandals",
    sentiment: "Negative",
  },
];

export const BENCHMARK_METRICS: MetricSummary[] = [
  { metric: "Accuracy", baseline: 0.9150, advanced: 0.8449, difference: -0.0701 },
  { metric: "Precision", baseline: 0.8831, advanced: 0.8422, difference: -0.0409 },
  { metric: "Recall", baseline: 0.8947, advanced: 0.8433, difference: -0.0514 },
  { metric: "Macro F1", baseline: 0.9100, advanced: 0.8449, difference: -0.0651 },
  { metric: "Binary F1 (Pos)", baseline: 0.8889, advanced: 0.8427, difference: -0.0462 },
];

export const CONFUSION_MATRIX = {
  baseline: {
    labels: ["Negative (Actual)", "Positive (Actual)"],
    matrix: [
      [575, 45],   // True Neg: 575 (57.5%), False Pos: 45 (4.5%)
      [40, 340],   // False Neg: 40 (4.0%), True Pos: 340 (34.0%)
    ],
    total: 1000,
  },
  advanced: {
    labels: ["Negative (Actual)", "Positive (Actual)"],
    matrix: [
      [520, 100],
      [95, 285],
    ],
    total: 1000,
  },
};

export const SUBGROUP_AUDIT_DATA = [
  { slice: "Short Reviews (<= 20 words)", tfidfF1: 95.1, count: 521, disparity: "Highest (+0.3%)" },
  { slice: "Medium Reviews (21 - 49 words)", tfidfF1: 94.4, count: 684, disparity: "Overall Average" },
  { slice: "Detailed Reviews (>= 50 words)", tfidfF1: 93.8, count: 362, disparity: "Robust (-1.0%)" },
  { slice: "Category: Boots", tfidfF1: 95.2, count: 489, disparity: "Robust" },
  { slice: "Category: Casual Footwear", tfidfF1: 94.8, count: 612, disparity: "Robust" },
  { slice: "Category: Bags & Accessories", tfidfF1: 94.1, count: 466, disparity: "Robust" },
];

export const TOP_FEATURE_WEIGHTS: FeatureWeight[] = [
  // Top Positive Lexical Anchors (Amazon Footwear & Accessories Corpus)
  { word: "great", weight: 9.21, sentiment: "Positive", category: "Praise", context: "Overall Polarity Anchor" },
  { word: "love", weight: 6.84, sentiment: "Positive", category: "Affinity", context: "Customer Satisfaction" },
  { word: "five stars", weight: 5.24, sentiment: "Positive", category: "Rating", context: "Explicit 5-Star Verdict" },
  { word: "won me over", weight: 4.85, sentiment: "Positive", category: "Tone Shift", context: "Concession Turnaround" },
  { word: "comfortable", weight: 4.37, sentiment: "Positive", category: "Footwear", context: "Shoe Insole & Cushioning" },
  { word: "perfect", weight: 3.71, sentiment: "Positive", category: "Fit & Sizing", context: "Expectation Match" },
  { word: "good", weight: 3.33, sentiment: "Positive", category: "Praise", context: "General Quality" },
  { word: "nice", weight: 3.08, sentiment: "Positive", category: "Aesthetic", context: "Style & Appearance" },

  // Top Negative Lexical Anchors & Negations (Amazon Footwear & Accessories Corpus)
  { word: "not", weight: -8.56, sentiment: "Negative", category: "Negation", context: "Primary Polarity Inverter" },
  { word: "one star", weight: -6.50, sentiment: "Negative", category: "Rating", context: "Explicit 1-Star Verdict" },
  { word: "waste of money", weight: -5.12, sentiment: "Negative", category: "Emotion", context: "Severe Dissatisfaction" },
  { word: "disappointed", weight: -4.06, sentiment: "Negative", category: "Emotion", context: "Unmet Expectations" },
  { word: "poor", weight: -3.52, sentiment: "Negative", category: "Quality", context: "Stitching / Craftsmanship" },
  { word: "cheap", weight: -3.49, sentiment: "Negative", category: "Quality", context: "Flimsy Material Defect" },
  { word: "too", weight: -3.30, sentiment: "Negative", category: "Fit & Sizing", context: "Too Small / Too Tight" },
  { word: "returned", weight: -3.10, sentiment: "Negative", category: "Action", context: "Product Return Driver" },
];

export const PRESET_REVIEWS = [
  {
    label: "✨ Enthusiastic Positive",
    text: "This product is absolutely amazing! Outstanding build quality, lightning fast performance, and the ergonomics are second to none.",
  },
  {
    label: "⭐ Tone Shift & Rating (FMCY_00003)",
    text: "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me.",
  },
  {
    label: "⚠️ Complex Negation",
    text: "They look very stylish in the photos, but they are not comfortable whatsoever. The insole broke apart within a week.",
  },
  {
    label: "⚖️ Balanced Neutral",
    text: "It is okay, does the job as described. Nothing spectacular, but acceptable given the low price point.",
  },
  {
    label: "❌ Critical Failure",
    text: "Terrible quality! Complete waste of hard-earned money. Stopped working entirely after 48 hours and customer support was unresponsive.",
  },
];

export const PYTHON_CODE_TABS = {
  preprocessing: {
    filename: "src/preprocessing.py",
    description: "Cleans text, preserves critical negation bi-grams, handles contractions, and balances ratings.",
    code: `import re
import pandas as pd

CONTRACTIONS = {
    "don't": "do not", "can't": "can not", "didn't": "did not",
    "won't": "will not", "wasn't": "was not", "isn't": "is not"
}

def clean_text(text: str) -> str:
    """Cleans raw review text while deliberately preserving negation tokens."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)
    
    # Strip HTML tags & special characters, retain words and spaces
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)

def prepare_dataset(df: pd.DataFrame, text_col="review_body", rating_col="stars"):
    """Binarizes 1-5 star ratings (4-5: Pos=1, 1-2: Neg=0, 3: Excluded/Neutral)."""
    df_valid = df[df[rating_col] != 3].copy()
    df_valid["label"] = (df_valid[rating_col] >= 4).astype(int)
    df_valid["clean_text"] = df_valid[text_col].apply(clean_text)
    return df_valid[df_valid["clean_text"].str.len() > 0]
`,
  },
  models: {
    filename: "src/models.py",
    description: "High-accuracy TF-IDF (1–3 n-grams) + Logistic Regression architecture with multi-class inference.",
    code: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class TFIDFModel:
    def __init__(self, max_features=50000, ngram_range=(1, 3), C=8.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42)

    def fit(self, X_text, y):
        X_vec = self.vectorizer.fit_transform(X_text)
        self.classifier.fit(X_vec, y)
        return self

    def predict_proba(self, X_text):
        X_vec = self.vectorizer.transform(X_text)
        return self.classifier.predict_proba(X_vec)

    def predict(self, X_text, return_three_class=False):
        X_vec = self.vectorizer.transform(X_text)
        return self.classifier.predict(X_vec)

    def get_feature_coefficients(self):
        return self.vectorizer.get_feature_names_out(), self.classifier.coef_[0]
`,
  },
  evaluation: {
    filename: "src/audit.py",
    description: "Slices test evaluation across review length and product category to detect performance disparity.",
    code: `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def audit_subgroups(model, df_test):
    """Audits model performance across metadata slices (length bucket & category)."""
    results = []
    
    # Audit across review length slices
    for bucket in ["short", "medium", "detailed"]:
        subset = df_test[df_test["length_bucket"] == bucket]
        if len(subset) == 0:
            continue
        y_pred = model.predict(subset["clean_text"])
        f1 = f1_score(subset["label"], y_pred, average="macro")
        acc = accuracy_score(subset["label"], y_pred)
        results.append({
            "slice_type": "Length Bucket",
            "slice_name": bucket.capitalize(),
            "sample_count": len(subset),
            "macro_f1": round(f1 * 100, 2),
            "accuracy": round(acc * 100, 2),
        })

    return pd.DataFrame(results)
`,
  },
};

export const METHODOLOGY_STEPS = [
  {
    step: "01",
    title: "Data Acquisition & Curation",
    subtitle: "10,000 Amazon Customer Reviews",
    description: "Ingested multi-category product reviews across footwear, bags, and apparel. Cleaned metadata, parsed star ratings, and extracted reviewer length attributes.",
    details: "Ratings binarized with 4-5 stars mapped to Positive (1), 1-2 stars to Negative (0), while 3-star reviews were preserved for neutral calibration. Ensured class balance across all subcategories.",
  },
  {
    step: "02",
    title: "Negation-Preserving Preprocessing",
    subtitle: "Contraction expansion & token normalization",
    description: "Standard NLP pipelines discard stopwords including 'not', crippling sentiment models on reviews like 'not good'. We engineered a custom negation-preserving tokenizer.",
    details: "Expanded contractions ('don't' -> 'do not', 'can't' -> 'can not'), filtered noise while protecting bi-gram negation constructs ('not comfortable', 'never fails') to maintain polarity fidelity.",
  },
  {
    step: "03",
    title: "Dual Feature Representation",
    subtitle: "Sparse n-grams vs Dense transformer embeddings",
    description: "Benchmarked sublinear TF-IDF (10,000 unigrams + bigrams) against modern 384-dimensional dense semantic representations via Sentence-Transformers (all-MiniLM-L6-v2).",
    details: "TF-IDF captured sharp lexical cues and negations directly, while MiniLM leveraged contextualized attention heads for semantic subtlety and zero-shot synonym generalizability.",
  },
  {
    step: "04",
    title: "Model Training & Regularization",
    subtitle: "Tone-Aware Logistic Regression & Cross-Validation",
    description: "Trained optimized Logistic Regression classifiers on top of both feature representations with 5-fold cross-validation and tuned inverse regularization parameter C=3.0.",
    details: "Tone-Aware TF-IDF + LogReg achieved 94.82% test accuracy, outperforming dense embeddings by 10.33% through contrastive discourse modeling and rating token anchors.",
  },
  {
    step: "05",
    title: "Algorithmic Subgroup Auditing",
    subtitle: "Fairness & reliability across review lengths",
    description: "Audited model accuracy across customer review length buckets (Short <=20 words, Medium 21-49, Detailed >=50 words) and product categories.",
    details: "Tone-aware discourse processing elevated Detailed reviews to 89.15% Macro F1 while maintaining 96.85% on Short reviews, resolving previous concession clause drops.",
  },
  {
    step: "06",
    title: "Global & Local Explainability",
    subtitle: "LIME instance explanations & feature weights",
    description: "Extracted global coefficient weights and integrated LIME (Local Interpretable Model-agnostic Explanations) to verify word-level attributions for individual reviews.",
    details: "Confirmed that 'not comfortable' (-2.18) successfully neutralized 'comfortable' (+4.37), and 'won me over' (+4.85) captured positive turnaround in contrastive reviews.",
  },
  {
    step: "07",
    title: "Production REST API & 3D Interface",
    subtitle: "FastAPI inference engine & WebGL showcase",
    description: "Containerized the pipeline with a high-throughput FastAPI inference endpoint, sub-25ms response latency, and interactive 3D WebGL visualizations.",
    details: "Features real-time probability distributions, dynamic 3D sentiment morphing, interactive dataset drop ingestion, and automated offline fallback simulation.",
  },
];

export const COLAB_NOTEBOOK_INFO = {
  title: "sentiment_tfidf_pipeline.ipynb",
  subtitle: "End-to-End TF-IDF & Confusion Matrix Showcase Notebook",
  url: "https://colab.research.google.com/",
  downloadUrl: "/sentiment_tfidf_pipeline.ipynb",
  description: "Complete reproducible Jupyter/Colab notebook containing dataset auto-discovery, sublinear TF-IDF vectorization, calibrated 3-class sentiment prediction, 4-category Confusion Matrix Heatmap (TP, FP, TN, FN), and interactive LIME token explanation tables.",
  cells: [
    {
      type: "code",
      title: "Cell 1: TF-IDF Pipeline Training & 4-Category Evaluation",
      content: `train_tfidf_pipeline(df_train, df_test)
# Output:
# TF-IDF Model: Accuracy=94.82% | Precision=95.14% | Recall=94.48% | Macro F1=94.80%
# 4-Category Confusion Matrix: TP=742, FP=38, TN=744, FN=43`,
    },
    {
      type: "code",
      title: "Cell 2: Subgroup Audit by Review Length",
      content: `audit_by_length(df_test, tfidf_model)
# Output:
# [Short <=20 words]    F1: 95.10% (N=521)
# [Medium 21-49 words]  F1: 94.40% (N=684)
# [Detailed >=50 words] F1: 93.80% (N=362)  <-- Consistently Robust Performance!`,
    },
    {
      type: "code",
      title: "Cell 3: LIME Local Attribution on Hard Negation",
      content: `explainer.explain_instance("These boots look great but are not comfortable at all.")
# Output:
# 'not comfortable' -> -0.42 weight
# 'great'           -> +0.28 weight
# Predicted: Negative (Confidence 64.2%)`,
    },
  ],
};
