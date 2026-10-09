# Product Review Sentiment Analysis & Subgroup Auditing

An end-to-end NLP mini-project for classifying customer review sentiment with rigorous subgroup performance auditing, bias analysis, and model explainability.

---

## Project Overview
This repository contains a complete, reproducible NLP pipeline that:
1. Cleans real Amazon customer product reviews while **preserving critical negation tokens** (*"not"*, *"never"*, *"n't"*).
2. Binarizes 1–5 star ratings (4–5 = Positive, 1–2 = Negative, 3 = Excluded) to achieve a balanced sentiment dataset.
3. Trains and benchmarks a **TF-IDF + Logistic Regression baseline** against **Sentence-Transformers dense embeddings (`all-MiniLM-L6-v2`)**.
4. Conducts an **algorithmic subgroup audit** across product categories and review lengths (Short $\le 20$ words vs. Detailed $\ge 50$ words).
5. Provides **global explainability** (top feature weights) and **local instance explainability** using **LIME**.

---

## Project Structure

```
nlp_sentiment_project/
├── data/
│   ├── download_data.py           # Dataset acquisition, validation & categorization script
│   └── raw_reviews.csv            # Real 10,000 Amazon product reviews dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Text cleaning, contraction expansion, bucketing & split
│   ├── models.py                  # Baseline (TF-IDF) & Advanced (MiniLM) models + evaluation
│   ├── audit.py                   # Subgroup performance auditing & disparity flagger
│   ├── explainability.py          # Top coefficients extractor & LIME text explainer
│   └── run_pipeline.py            # End-to-end command-line runner
├── notebooks/
│   ├── sentiment_analysis_pipeline.ipynb # Interactive, fully runnable Jupyter Notebook
│   └── create_notebook.py         # Programmatic notebook generator script
├── outputs/
│   ├── overall_metrics.txt        # Accuracy, Precision, Recall, Macro/Binary F1 table
│   ├── confusion_matrix.png       # Side-by-side confusion matrix heatmaps
│   ├── subgroup_metrics.csv       # Complete audit performance breakdown by slice
│   ├── top_features.csv           # Top 20 positive & negative word coefficients
│   ├── top_features.png           # Feature importance bar chart
│   ├── lime_examples.html         # Interactive LIME HTML explanation cards
│   └── lime_examples_summary.png  # LIME local token weight summary visualization
├── WRITEUP.md                     # Comprehensive 1-2 page course assignment write-up
├── MODEL_CARD.md                  # Standardized 1-page model card with audit table & limits
├── PITCH_OUTLINE.md               # 5-minute presentation speaker notes & Q&A prep
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Setup and reproduction instructions
```

---

## Quickstart & Environment Setup

### 1. Prerequisites
- **Python**: 3.9+ (tested on Python 3.9.6 on macOS Apple Silicon)
- **Virtual Environment**: Recommended to prevent dependency conflicts.

### 2. Activate Virtual Environment & Install Dependencies
From the project root:

```bash
# Optional: create a new virtual environment if needed
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(If you are running on the system's pre-configured virtual environment at `/Users/aadarshsundrani/.gemini/antigravity/scratch/venv`, activate it directly via `source /Users/aadarshsundrani/.gemini/antigravity/scratch/venv/bin/activate`).*

---

## How to Run

### Option A: Run the End-to-End Pipeline (CLI)
To run the full workflow top-to-bottom (data prep, model training, evaluation, subgroup auditing, and explainability):

```bash
python src/run_pipeline.py
```

**Optional CLI Arguments:**
- `--sample-size 2000`: Subsamples $N$ reviews for rapid testing.
- `--no-advanced`: Runs only the TF-IDF baseline model.
- `--output-dir custom_outputs`: Specifies a custom destination for generated reports.

Execution takes **~15–20 seconds** on standard modern hardware.

### Option B: Run the Interactive Jupyter Notebook
Launch Jupyter Notebook to step through visualizations, distribution charts, and inline LIME displays:

```bash
jupyter notebook notebooks/sentiment_analysis_pipeline.ipynb
```

*(Or open the notebook in VS Code / Antigravity IDE and click "Run All").*

### Option C: Re-generate the Dataset Only
To inspect or re-download the Amazon benchmark data:

```bash
python data/download_data.py
```

### Option D: Launch the 3D Interactive Web Showcase (WebGL + Three.js)
To run the full 3D interactive product sentiment showcase:

```bash
# 1. Start the FastAPI Inference Backend
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload

# 2. In another terminal, start the React 18 + Three.js Frontend
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to explore the full 3D experience with reactive sentiment morphing, drag-and-drop dataset ingestion, live pipeline runner, interactive prediction sandbox, and audit charts.


---

## Summary of Results

### 1. Overall Model Comparison (Held-Out Test Set, $N=1,567$)
| Metric | Baseline (TF-IDF + LogReg) | Advanced (MiniLM Embeddings) |
| :--- | :---: | :---: |
| **Accuracy** | **87.24%** | 84.49% |
| **Precision** | **88.86%** | 84.22% |
| **Recall** | 84.72% | **84.33%** |
| **Macro F1** | **87.22%** | 84.49% |
| **Binary F1 (Pos)** | **86.74%** | 84.27% |

### 2. Key Subgroup Audit Findings
- **Review Length Disparity:**
  - Short reviews ($\le 20$ words): **90.15% Macro F1**
  - Detailed reviews ($\ge 50$ words): **75.38% Macro F1**
  - **Disparity Gap:** Detailed reviews underperform short reviews by **14.8 percentage points in F1**. Long reviews often include mixed trade-offs (e.g. *"Great design, but sole unglued"*), challenging document-level classifiers.
- **Product Category Disparity:**
  - Footwear categories (*Boots*, 85.97% F1; *Casual Footwear*, 85.86% F1) outperform *Bags & Accessories* (75.71% F1) due to hardware-specific terminology (*"zippers"*, *"straps"*).

### 3. Explainability Highlights
- **Global Features:** Top positive words include `great` (+6.40), `love` (+5.78), and `comfortable` (+4.37). Top negative words include `not` (-6.61), `disappointed` (-4.06), and `uncomfortable` (-3.02). Bigram `not comfortable` (-2.18) was successfully learned as strongly negative.
- **LIME Local Case:** In *"They are not comfortable what so ever..."*, LIME shows that `"not"` and `"not comfortable"` override the positive adjective `"comfortable"`, preventing a false positive.

---

## Reproducibility
- All random splits and model seeds are pinned (`random_state=42`).
- Hugging Face offline environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) ensure full execution without network latency or external API dependencies.
