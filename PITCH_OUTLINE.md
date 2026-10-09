# 5-Minute In-Class Pitch: Speaker Notes & Presentation Outline

**Project:** Product Review Sentiment Analysis with Subgroup Auditing & Explainability  
**Format:** Concise bullet-point speaker notes (estimated 5-minute delivery, ~1 minute per section)

---

## Minute 1: Problem & Motivation (The Hook)
- **The Hook:** E-commerce platforms receive millions of product reviews daily—monitoring customer satisfaction manually is impossible.
- **The Business Need:** Automatically classify reviews into Positive vs. Negative to trigger returns, route tickets, and flag defective products.
- **The Common Pitfall:** Most NLP projects stop at overall accuracy (e.g., *"We got 87%, we're done"*).
- **Our Angle:** A model that looks great on aggregate can fail catastrophically on specific customer segments. We built a high-performing classifier, audited it across product categories and review lengths, and inspected its inner workings with LIME.

---

## Minute 2: Dataset & Preprocessing (The Engineering Choice)
- **Dataset:** 10,000 real customer reviews from the public Amazon Benchmark Dataset (Julian McAuley Lab, UCSD).
- **Labeling:** Binarized from star ratings—4-5 stars = Positive, 1-2 stars = Negative. Dropped neutral 3-star reviews for clean polarity.
- **Class Balance:** Resulted in 8,039 reviews with an exact 50/50 balance (50.04% Positive vs. 49.96% Negative).
- **The Critical Preprocessing Decision:** Negation Preservation.
  - Standard text cleaners strip punctuation and remove stop words like `"not"`, converting *"not comfortable"* into *"comfortable"*—a complete polarity flip!
  - We engineered a custom preprocessor: expanded contractions (`"don't"` $\to$ `"do not"`), preserved negation tokens (`"not"`, `"no"`, `"never"`), and extracted bigrams.

---

## Minute 3: Modeling & Results (The Comparison)
- **Two Architectures Compared:**
  - **Baseline:** Sublinear TF-IDF (unigrams + bigrams, top 10k features) + Logistic Regression.
  - **Advanced:** Pretrained Contextual Sentence Embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) + Linear Probe.
- **Held-Out Test Results (1,567 reviews):**
  - Baseline (TF-IDF): **87.24% Accuracy | 87.22% Macro F1**
  - Advanced (MiniLM): **84.49% Accuracy | 84.49% Macro F1**
- **Why Did TF-IDF Win?**
  - Product sentiment relies heavily on explicit polarity words (*"great"*, *"comfortable"*, *"disappointed"*, *"returned"*).
  - TF-IDF assigns direct, unbounded weights to these tokens.
  - Dense STS embeddings compress sentences into a fixed vector, occasionally diluting isolated negative signals in longer text.

---

## Minute 4: Subgroup Audit & Explainability (The Star Graded Component)
- **Subgroup Audit Findings (Our Core Value-Add):**
  - **Length Disparity:** 
    - Short reviews ($\le 20$ words): **90.15% Macro F1**
    - Detailed reviews ($\ge 50$ words): **75.38% Macro F1**
    - **The Gap:** A massive **14.8 percentage point drop (-16.4%)** on detailed reviews!
  - **Why?** Long reviews contain mixed multi-aspect sentiment (e.g., *"Love the color and style, but the strap tore after two days"*). Document-level classifiers get confused by conflicting clauses.
  - **Category Disparity:** Boots scored 85.97% F1, while Bags & Accessories lagged at 75.71% F1 due to hardware-specific vocabulary (*"zippers"*, *"clasps"*, *"straps"*).
- **Explainability Validation:**
  - **Global Weights:** `"not"` was the #1 negative feature (-6.61). Bigram `"not comfortable"` (-2.18) was learned as strongly negative.
  - **LIME Local Case:** In *"They are not comfortable what so ever..."*, LIME proved the model prioritized `"not"` and `"not comfortable"`, proving it wasn't fooled by the positive word `"comfortable"`.

---

## Minute 5: Limitations & Future Improvements (Self-Aware Next Steps)
- **Honest Limitations:**
  - Single-vector document classification fails on mixed-aspect long reviews.
  - Sarcastic reviews (*"Great if you like bleeding heels"*) still confuse the bag-of-words model.
- **What I'd Build With More Time:**
  - **Aspect-Based Sentiment Analysis (ABSA):** Break reviews into sentence clauses and predict sentiment per aspect (e.g., Size: Positive, Durability: Negative).
  - **Domain Fine-Tuning:** End-to-end fine-tuning of `DeBERTa-v3` on long-form reviews to resolve multi-clause discourse.
- **Final Takeaway:** Always audit beyond overall accuracy—subgroup analysis and explainability turn an ML toy into a trustworthy production model.

---

## Quick Q&A Preparation Cheat Sheet
- **Q: Why didn't you use DistilBERT directly?**  
  *A:* MiniLM embeddings with a linear probe run in seconds, provide 384-dimensional contextual vectors without GPU bottlenecks, and allow instant reproducible evaluation across all audit slices.
- **Q: Why did you exclude 3-star reviews?**  
  *A:* 3-star reviews are inherently neutral or mixed; in binary sentiment tasks, forcing them into 0 or 1 introduces subjective noise that degrades decision boundaries.
- **Q: Why did detailed reviews score lower than short reviews?**  
  *A:* Detailed reviews contain concessions and mixed feedback across multiple aspects, whereas short reviews are concise and unambiguous.
