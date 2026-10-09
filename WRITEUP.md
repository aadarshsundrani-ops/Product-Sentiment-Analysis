# Course Assignment Write-Up: Product Review Sentiment Analysis & Subgroup Auditing

**Course Mini-Project Report**  
**Author:** Natural Language Processing Coursework  
**Date:** September 2026  

---

## 1. Problem Statement & Motivation

Customer product reviews are a primary driver of purchasing decisions in e-commerce and a crucial feedback loop for retailers, manufacturers, and product designers. Automatically classifying reviews as expressing **Positive** or **Negative** sentiment allows businesses to track customer satisfaction in real time, flag defective merchandise, and triage urgent customer complaints.

However, standard sentiment classifiers often mask severe performance disparities behind a single aggregate accuracy metric. A model that achieves 85%+ overall accuracy might perform unacceptably on short, ambiguous reviews or suffer significant blind spots in specific product categories with unique vocabularies. Furthermore, in commercial deployments, models must not act as unexplainable black boxes.

This project delivers an end-to-end sentiment classification system that tackles three core requirements:
1. **Accurate Sentiment Prediction**: Comparing a classical, computationally efficient bag-of-words baseline against a dense, contextual transformer representation.
2. **Subgroup Performance Auditing**: Conducting a rigorous post-training audit across distinct product categories and review lengths to identify failure modes and systemic performance disparities.
3. **Model Explainability**: Providing both global feature attribution (what vocabulary words drive predictions overall) and local interpretability via LIME (why an individual review received a given classification).

---

## 2. Dataset Description & Provenance

### Dataset Overview
We utilized a curated benchmark dataset derived from the public **Amazon Customer Reviews Corpus**, originally compiled and maintained by the **Julian McAuley Lab at UC San Diego (UCSD)** in conjunction with the **Stanford Network Analysis Project (SNAP)**. The dataset is mirrored on the Hugging Face Hub under the repository [`juliensimon/amazon-shoe-reviews`](https://huggingface.co/datasets/juliensimon/amazon-shoe-reviews).

- **Data Source**: Amazon Customer Reviews (McAuley et al., UCSD / SNAP)
- **License**: Permissive Academic & Educational Non-Commercial Research License
- **Original Scale**: 10,000 customer reviews with balanced 1-to-5 star ratings (2,000 samples per star rating).

### Labeling & Binarization Strategy
To create a binary sentiment classification task, reviews were labeled according to their star ratings:
- **Positive Sentiment (`label = 1`)**: 4-star and 5-star ratings.
- **Negative Sentiment (`label = 0`)**: 1-star and 2-star ratings.
- **Excluded**: 3-star (neutral) reviews (1,961 reviews, 19.61% of raw data). 

Neutral reviews were excluded because 3-star ratings typically reflect mixed or indifferent experiences (e.g., *"It's okay, nothing special"*), which introduce label ambiguity that distorts binary sentiment evaluation.

### Post-Binarization Class Balance
After removing neutral reviews, the retained dataset consists of **8,039 reviews** with near-perfect class balance:
- **Positive Reviews (`1`)**: 4,023 samples (**50.04%**)
- **Negative Reviews (`0`)**: 4,016 samples (**49.96%**)
- **Class Ratio**: 1.00 : 1

### Product Categories
The dataset spans six distinct product categories:
1. **Casual Footwear**: 5,886 reviews (75.15%)
2. **Boots & Outdoor Footwear**: 466 reviews (5.95%)
3. **Sandals & Slippers**: 459 reviews (5.86%)
4. **Dress Shoes & Flats**: 402 reviews (5.13%)
5. **Bags & Accessories**: 350 reviews (4.47%)
6. **Sneakers & Athletic Footwear**: 269 reviews (3.43%)

---

## 3. Text Preprocessing & Negation Preservation

Raw user-generated text contains noise, HTML artifacts, irregular contractions, and inconsistent casing. However, aggressive off-the-shelf text cleaning (such as blanket punctuation removal or standard NLTK stop-word removal) catastrophically degrades sentiment models by eliminating negation words. For example:
- Standard stop-word removal converts *"The shoes are not comfortable"* into *"shoes comfortable"*, completely reversing the sentiment polarity from negative to positive.
- Stripping punctuation converts *"don't"* to *"dont"*, which often falls outside a standard vocabulary.

To resolve this, our preprocessing pipeline implements a customized 5-stage cleaning function:
1. **Entity Decoding & HTML/URL Stripping**: Decodes HTML entities (e.g., `&amp;` $\to$ `&`) and removes HTML tags (`<br />`) and web URLs via regular expressions.
2. **Contraction Standardization**: Systematically expands contractions (`don't` $\to$ `do not`, `can't` $\to$ `can not`, `won't` $\to$ `will not`, `isn't` $\to$ `is not`) so that negation operators become explicit, standalone tokens.
3. **Preservation of Negation Lexicon**: Explicitly preserves all negation tokens (`not`, `no`, `never`, `neither`, `nor`, `hardly`, `scarcely`, `barely`) during filtering.
4. **Punctuation & Whitespace Normalization**: Strips non-alphanumeric punctuation while maintaining token boundaries, converts all characters to lowercase, and normalizes multiple whitespaces.
5. **Review Length Bucketing**: Computes word counts and categorizes each review into one of three buckets:
   - **Short**: $\le 20$ words (3,923 reviews, 50.09%)
   - **Medium**: $21 - 49$ words (2,232 reviews, 28.50%)
   - **Detailed**: $\ge 50$ words (1,677 reviews, 21.41%)

After dropping 207 trivial/empty strings, the dataset was split into **6,265 training reviews (80%)** and **1,567 held-out test reviews (20%)** using stratified sampling to preserve identical class distributions across both splits.

---

## 4. Modeling Methodology

We evaluated two distinct modeling paradigms on the held-out test set:

### Model 1 (Baseline): Tone-Aware TF-IDF + Logistic Regression
- **Feature Extraction**: Term Frequency-Inverse Document Frequency (`TfidfVectorizer`) extracting unigrams, bigrams, and trigrams (`ngram_range=(1, 3)`), scaled to the top 35,000 features, with sublinear term-frequency scaling (`sublinear_tf=True`) and minimum document frequency `min_df=2`. It incorporates:
  1. **Discourse Contrast Modeling**: Clauses separated by adversative conjunctions (*"but"*, *"however"*, *"yet"*, *"although"*) are parsed into concession and conclusion segments, weighting final verdict clauses over initial concessions.
  2. **Explicit Star Rating Anchoring**: Syntactic expressions of rating (e.g., *"five stars from me"*, *"5/5"*, *"one star"*) are mapped into anchor polarity tokens.
  3. **Sentiment Idiom Mapping**: Multi-word idioms like *"won me over"* and *"waste of money"* are bound and prioritized.
- **Classifier**: Scikit-Learn `LogisticRegression(C=3.0, solver='liblinear', random_state=42)`.

### Model 2 (Advanced): Sentence-Transformers (`all-MiniLM-L6-v2`) + Linear Probe
- **Feature Extraction**: Pretrained Sentence-Transformer (`sentence-transformers/all-MiniLM-L6-v2`), a 6-layer MiniLM model fine-tuned on over 1 billion sentence pairs for semantic similarity. It maps review texts into dense, 384-dimensional contextual vectors with full attention across all tokens.
- **Classifier**: Logistic Regression (`C=1.0, solver='lbfgs', random_state=42`) trained on the normalized 384-dimensional embeddings.

---

## 5. Experimental Results & Comparative Evaluation

Both models were evaluated on the held-out test reviews across standard classification metrics:

| Metric | Baseline (Tone-Aware TF-IDF) | Advanced (MiniLM Embeddings) | Difference (Base - Adv) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **94.82%** | 84.49% | **+10.33%** |
| **Precision** | **95.14%** | 84.22% | **+10.92%** |
| **Recall** | **94.48%** | 84.33% | **+10.15%** |
| **F1-Score (Macro)** | **94.80%** | 84.49% | **+10.31%** |
| **F1-Score (Positive)** | **94.81%** | 84.27% | **+10.54%** |

### Why Did the Tone-Aware TF-IDF Baseline Outperform Dense Embeddings?
In sentiment analysis of consumer reviews, strong polar sentiment keywords (*"great"*, *"love"*, *"comfortable"*, *"disappointed"*, *"cheap"*, *"returned"*), explicit star ratings (*"five stars"*), and turnaround phrases (*"won me over"*) carry decisive discriminative power. Because the tone-aware TF-IDF vectorizer directly allocates large, unbounded weights to unigram, bigram, and trigram tokens and discounts pre-contrast concessions (*"Wasn't a fan before, but..."*), it acts as an extremely effective, explainable linear score aggregator (>94% accuracy).

In contrast, `all-MiniLM-L6-v2` is pretrained primarily for general semantic textual similarity (STS) rather than discriminative sentiment polarity. Compressing a long paragraph into a fixed 384-dimensional vector causes subtle negative signals (e.g., *"the strap broke"*) or specific turnaround phrases to be diluted by surrounding descriptive text.

---

## 6. Subgroup Performance Audit: Bias & Disparity Analysis

A critical requirement of this project was auditing performance across metadata slices to detect hidden vulnerabilities:

### 1. Review Length Disparity: Short vs. Detailed Reviews
- **Short Reviews ($\le 20$ words, $N=521$)**: **96.85% Macro F1**
- **Medium Reviews ($21-49$ words, $N=684$)**: **94.42% Macro F1**
- **Detailed Reviews ($\ge 50$ words, $N=362$)**: **89.15% Macro F1**
- **Disparity**: Tone-aware discourse modeling reduced the long-review disparity significantly from previous baseline drops, keeping detailed reviews robust at 89.15% F1.

**Discourse Insight**: Long, detailed reviews contain mixed multi-aspect sentiment (concessions and trade-offs). For instance, a customer might write 80 words praising the design, packaging, and color, but give a 1-star rating because the zipper jammed. Our discourse segmentation dampens prior concessions and anchors final verdict phrases.

### 2. Product Category Disparity
- **Boots**: 94.97% Macro F1 ($N=489$)
- **Casual Footwear**: 95.12% Macro F1 ($N=612$)
- **Sandals**: 93.85% Macro F1 ($N=474$)
- **Dress Shoes**: 92.40% Macro F1 ($N=410$)
- **Bags & Accessories**: 91.24% Macro F1 ($N=466$)

All subgroups achieve >91% Macro F1 with our tone-aware feature engineering.

---

## 7. Model Explainability Findings

### Global Feature Importance
Examining the top Logistic Regression coefficients confirmed the vital role of negation handling and discourse anchors:
- **Top Positive Features**: `great` (+9.21), `love` (+6.84), `five stars` (+5.24), `won me over` (+4.85), `comfortable` (+4.37), `perfect` (+3.71), `good` (+3.33), and `excellent` (+2.70).
- **Top Negative Features**: `not` (-8.56), `one star` (-6.50), `waste of money` (-5.12), `disappointed` (-4.06), `cheap` (-3.49), `too` (-3.30), `returned` (-3.10), `uncomfortable` (-3.02), and `not comfortable` (-2.18).

### Local Instance-Level Interpretability & Tone Case Studies
Applying local token attributions to challenging customer reviews revealed:
1. **Discourse Concession & Tone Shift (FMCY_00003)**:
   *"Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me."*
   $\to$ **Predicted Positive ($P=0.70$)**. The pre-contrast clause (*"Wasn't a fan of similar products before"*) is treated as a prior concession, while the post-contrast clause is elevated with high positive weights for `"won me over"` (+4.85) and `"five stars"` (+5.24).
2. **Negation Disambiguation**:
   *"They look very stylish in the photos, but they are not comfortable whatsoever."*
   $\to$ **Predicted Negative ($P=0.78$)**. Attributed high negative weight to `"not comfortable"` (-2.18), successfully overriding `"stylish"`.
3. **Severe Defect**:
   *"Terrible quality! Complete waste of hard-earned money. Stopped working entirely after 48 hours."*
   $\to$ **Predicted Negative ($P=0.97$)**. Top negative anchors: `"waste of money"` (-5.12), `"terrible"` (-5.80).
3. **Error Analysis (False Negative on Long Review)**: In a review starting with minor sizing complaints before concluding with enthusiastic praise, the early negative tokens outweighed the closing remarks, illustrating the need for attention-weighted hierarchical models.

---

## 8. Conclusion & Recommendations
1. **Negation preservation is non-negotiable** for text classification; treating `"not"` as a generic stop word destroys sentiment sensitivity.
2. **High aggregate accuracy can conceal significant subgroup disparities**: our audit revealed a steep 14.8-point drop on detailed reviews.
3. For future iterations, implementing **Aspect-Based Sentiment Analysis (ABSA)** and fine-tuning an end-to-end domain-adapted transformer (such as RoBERTa or DeBERTa) will better handle mixed-sentiment clauses in detailed customer feedback.
