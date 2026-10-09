# Model Card: Product Review Sentiment Classifier

## 1. Model Details
- **Model Name:** Amazon Product Review Sentiment Analyzer (Baseline TF-IDF + LogReg & MiniLM Embeddings)
- **Model Version:** 1.0.0
- **Model Type:** Binary text sentiment classifier (Linear Bag-of-Words & Dense Sentence Embedding Classifiers)
- **Primary Architecture:** Sublinear TF-IDF (1-2 n-grams, 10,000 features) + L2-regularized Logistic Regression (`C=1.0`)
- **Secondary Architecture:** Sentence-Transformers `all-MiniLM-L6-v2` (384-dimensional dense contextual embeddings) + Logistic Regression
- **Release Date:** September 2026
- **License:** Open educational & academic research use

---

## 2. Intended Use & Scope
- **Intended Purpose:** Automated classification of e-commerce customer product reviews into **Positive** or **Negative** sentiment to assist in satisfaction tracking, feedback triage, and quality control.
- **Primary Users:** E-commerce product managers, data science teams, and customer experience operations.
- **Out-of-Scope Uses:** Legal/compliance determinations, automated punitive action against sellers, or applications outside consumer product feedback without domain retraining.

---

## 3. Dataset Summary
- **Data Source:** Amazon Customer Reviews Benchmark (McAuley Lab, UCSD / Stanford SNAP; Hugging Face `juliensimon/amazon-shoe-reviews`).
- **Data License:** Permissive Non-Commercial Academic/Research Terms.
- **Total Dataset Size:** 10,000 raw reviews; 8,039 retained post-binarization (Neutral 3-star reviews excluded).
- **Label Distribution:** 4,023 Positive (50.04%, 4-5 stars), 4,016 Negative (49.96%, 1-2 stars).
- **Data Splits:** 80% Train ($N=6,265$), 20% Test ($N=1,567$), stratified by label.

---

## 4. Overall Model Performance
Evaluated on the held-out test set ($N=1,567$):

| Model | Accuracy | Precision | Recall | Macro F1 | Binary F1 (Pos) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LogReg)** | **87.24%** | **88.86%** | **84.72%** | **87.22%** | **86.74%** |
| **Advanced (MiniLM Embeddings)** | 84.49% | 84.22% | 84.33% | 84.49% | 84.27% |

*The TF-IDF baseline achieves higher discriminative accuracy due to high-weight polarity unigrams and bigrams.*

---

## 5. Subgroup Performance Audit
Evaluated on the test set across product categories and review length buckets:

| Dimension | Subgroup | Sample Size ($N$) | Pos % | Accuracy | Macro F1 | Disparity vs. Overall F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall** | **Full Test Set** | **1,567** | **49.3%** | **84.49%** | **84.49%** | **Baseline** |
| **Product Category** | Casual Footwear | 1,197 | 47.9% | 85.88% | 85.86% | +1.37% |
| Product Category | Boots | 77 | 63.6% | 87.01% | 85.97% | +1.48% |
| Product Category | Sneakers & Athletic | 44 | 50.0% | 84.09% | 84.08% | -0.41% |
| Product Category | Sandals & Slippers | 103 | 53.4% | 78.64% | 78.62% | -5.87% |
| Product Category | Dress Shoes & Flats | 76 | 50.0% | 76.32% | 76.30% | -8.19% |
| Product Category | **Bags & Accessories** | **70** | **48.6%** | **75.71%** | **75.71%** | **-8.78% (Worst Category)** |
| **Review Length** | Short ($\le 20$ words) | 767 | 53.8% | 90.22% | 90.15% | +5.66% |
| Review Length | Medium ($21-49$ words) | 458 | 44.8% | 81.44% | 81.26% | -3.23% |
| Review Length | **Detailed ($\ge 50$ words)** | **342** | **45.0%** | **75.73%** | **75.38%** | **-9.11% (Worst Length)** |

> **Audit Flag:** Detailed reviews ($\ge 50$ words) suffer a **14.77 percentage point drop** in Macro F1 compared to Short reviews (0.7538 vs. 0.9015). Bags & Accessories lags overall accuracy by 8.78%.

---

## 6. Explainability Findings

### Global Feature Weights (TF-IDF Logistic Regression)
- **Top 5 Positive Words:** `great` (+6.40), `love` (+5.78), `comfortable` (+4.37), `perfect` (+3.71), `good` (+3.33).
- **Top 5 Negative Words:** `not` (-6.61), `disappointed` (-4.06), `cheap` (-3.49), `too` (-3.30), `returned` (-3.10).
- **Key Syntax Finding:** Contraction expansion and negation preservation allowed the model to rank `"not"` as its #1 negative feature and accurately weight bigrams like `"not comfortable"` (-2.18).

### Local Explanation Case Study (LIME)
- **Test Review:** *"They are not comfortable what so ever and I got blisters ... so I no longer wear them :("*
- **Ground Truth:** Negative (0) | **Model Prediction:** Negative ($P(\text{Negative}) = 0.982$)
- **Verbal Explanation:** In standard bag-of-words models, the presence of the word *"comfortable"* often triggers false positive classifications. However, LIME local attribution shows that the model assigned dominant negative weights to `"not"`, `"not comfortable"`, and `"blisters"`, proving that the classifier correctly captures local negation context rather than being misled by isolated positive adjectives.

---

## 7. Model Limitations & Risks
1. **Severe Degradation on Long, Detailed Reviews (Limitation):**  
   The model's Macro F1 drops sharply from **90.15%** on short reviews to **75.38%** on reviews $\ge 50$ words. In extended reviews, customers frequently discuss trade-offs (e.g., *"Beautiful color and soft leather, but the sole unglued on day two and customer support was terrible"*). Because the model evaluates sentiment at document level rather than aspect level, contradictory sentiment clauses dilute polarity scores.
2. **Vocabulary Shift Across Categories:**  
   Categories with hardware-centric terminology (*Bags & Accessories*, 75.71% F1) underperform footwear categories (*Boots*, 85.97% F1) due to terms like *"strap"*, *"zipper"*, and *"lining"* that are underrepresented in footwear training data.
3. **Sarcasm and Indirect Criticism:**  
   Reviews containing sarcastic praise (e.g., *"Great if you enjoy walking on nails"*) lack overt negative adjectives and are susceptible to false-positive classification.
