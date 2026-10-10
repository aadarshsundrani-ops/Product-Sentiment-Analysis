# Product Sentiment Analysis: Brief Write-up

**Authors:** Aadarsh Sundrani (25030422001) and Khush Jain (25030422063), BBA (AI), Symbiosis Artificial Intelligence Institute
**Live demo:** https://product-sentiment-analysis-ashen.vercel.app
**Dataset:** [FMCG Customer Sentiments (Kaggle)](https://www.kaggle.com/datasets/viratsharma14/fmcg-customer-sentiments)

## 1. Problem

The goal is to decide whether an FMCG product review is **positive** or **not positive**. Automated sentiment analysis often struggles with two things: **negation** ("not good") and **mixed reviews** that praise one thing and complain about another ("good effectiveness, but not impressed with the smell"). A single accuracy score hides these failures, so the project asks: *does the model treat every kind of review fairly, and can we see why it predicts what it does?* The project has three parts:

1. **Build** a working sentiment classifier with TF-IDF and Logistic Regression.
2. **Audit** it by review length and product category, not just with one overall score.
3. **Explain** which words drive its predictions, and show everything in a live web app.

## 2. Dataset

**FMCG Customer Sentiments**, a public dataset on Kaggle by Virat Sharma.

| Item | Value |
|---|---|
| Reviews | 3,500 |
| Product categories | 6 |
| Labels | positive 1,415 / negative 1,508 / neutral 577 |
| Distinct review texts | 2,618 |
| Median review length | 13 words |
| Evaluation set on the website | 1,500 records (603 positive, 897 not positive) |

**Labels.** The dataset has three sentiment labels. The website reports a binary result, **positive vs. not positive**, so negative and neutral reviews are grouped together.

## 3. Preprocessing

- Review text is converted to lowercase and tokenised into words and word pairs.
- Reviews are grouped by length for the audit: **short** (20 words or fewer), **medium** (21–49 words) and **detailed** (50 words or more).
- Reviews are also grouped by product category.
- *To confirm:* stop-word handling, the exact n-gram range, the number of TF-IDF features kept, and the train/test split method and random seed.

## 4. Approach and model

The model is a two-step pipeline:

- **Features: TF-IDF.** Each word is scored by how often it appears in a review (term frequency, TF) multiplied by how rare it is across all reviews (inverse document frequency, IDF, roughly `log(N / (1 + df))`). Common words get low scores and distinctive words get high scores. Each review becomes a sparse vector of these scores.
- **Classifier: Logistic Regression.** It learns one weight per TF-IDF feature. A review's score is the weighted sum of its features, passed through a sigmoid to give a probability of "positive". Training minimises cross-entropy loss with L2 regularisation.

A linear model was chosen on purpose. Every prediction is a weighted sum over words, so each word's contribution can be read directly. This is what makes the token-level explanation possible.

**The website** lets a user:
- see the confusion matrix of actual vs. predicted sentiment,
- read accuracy, precision, recall and F1,
- calculate TF-IDF scores for any review text,
- try their own product review,
- view the length and category audit.

## 5. Results

**Overall (evaluation set, n = 1,500)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority-class baseline (always "not positive") | 59.8% | — | 0.0% | 0.0% |
| **TF-IDF + Logistic Regression** | **91.1%** | **89.6%** | **88.2%** | **88.9%** |

The model beats the baseline by 31.3 points.

**Confusion matrix**

| | Predicted positive | Predicted not positive |
|---|---|---|
| **Actual positive** | 532 true positives (35.5%) | 71 false negatives (4.7%) |
| **Actual not positive** | 62 false positives (4.1%) | 835 true negatives (55.7%) |

The two error types are close to balanced: 71 positive reviews are missed and 62 others are wrongly flagged as positive.

**Length and category audit (F1)**

| Subgroup | Definition | n | F1 | Website flag |
|---|---|---|---|---|
| Short reviews | 20 words or fewer | 888 | 89.6% | Robust |
| Medium reviews | 21–49 words | 331 | 84.4% | Disparity flagged |
| Detailed reviews | 50 words or more | 281 | 92.1% | Highest |
| Baby Care | Product category | 267 | 85.8% | Review disparity |

Medium-length reviews score lowest of the three length groups, 7.7 points below detailed reviews. Baby Care also trails the overall F1 of 88.9%. *Still to add:* Health & Hygiene (n = 258), the other categories and a negation subgroup.

**Explainability (provisional).** These weights come from a re-run of TF-IDF (unigrams + bigrams) with Logistic Regression (C = 1.0) on the same dataset. Replace them with the website's own values if they differ.

| Top positive features | Top negative features |
|---|---|
| highly (+2.6) | not (−3.5) |
| loved (+1.9) | quite (−1.6) |
| five stars (+1.9) | terrible (−1.5) |
| would buy (+1.8) | would not (−1.5) |
| highly recommended (+1.7) | waste of (−1.3) |

The negation cue "not" carries the strongest weight in either direction. The bigrams "not worth" and "not good" are learned as negative, while "good" alone is positive, so word pairs help with simple negation.

**Mixed-sentiment example.** For *"Good effectiveness overall, but not impressed with the smell. Would buy again."* (true label: positive), the model predicts **not positive** (p(positive) = 0.41). "would buy" and "impressed" push towards positive, while "not" and "but not" push away. The praise and the complaint nearly cancel, so the probability lands just below 0.5.

## 6. Limitations

1. **Word-counting model.** TF-IDF scores weighted words and cannot read full context. Negation beyond common word pairs, sarcasm and mixed opinions remain the main risk.
2. **Uneven performance across groups.** Medium-length reviews (84.4% F1) and Baby Care (85.8% F1) are weaker than the rest. The overall 91.1% accuracy hides this.
3. **Binary framing.** The dataset has three labels, but the result is binary, so the 577 neutral reviews are not scored as their own class.
4. **Repeated reviews.** Only 2,618 of the 3,500 review texts are distinct, so identical reviews can fall in both training and test data and inflate scores.
5. **Narrow data.** English reviews from six FMCG categories with a median length of 13 words. Results may not carry over to other products or longer reviews.
6. **Small audit samples.** Each subgroup holds 258 to 888 records, so a gap of a few points may be noise.

**Recommended use:** a first-pass filter for triaging customer feedback, with human review of low-confidence predictions (probability between 0.4 and 0.6). It should not drive automated decisions.

**Future work:**
- Remove duplicate review texts before splitting, then re-run the evaluation.
- Add negation and mixed-sentiment subgroups to the audit; 1,678 reviews contain a negation word such as "not" or "never".
- Benchmark against a Transformer model (for example a fine-tuned DistilBERT) and repeat the same audit.
- Extend the classifier to three classes so neutral reviews are scored.
