import { FeatureWeight, TOP_FEATURE_WEIGHTS, ReviewSample } from '../data/sampleData';

export interface PredictRequest {
  text?: string;
  review_id?: string;
  model_choice?: 'tfidf' | 'baseline' | 'advanced';
}

export interface InfluentialWord {
  word: string;
  weight: number;
  influence: 'Positive' | 'Negative';
}

export interface PredictResponse {
  sentiment: 'Positive' | 'Negative' | 'Neutral';
  confidence: number;
  probabilities: {
    Positive: number;
    Negative: number;
    Neutral: number;
  };
  probability_positive?: number;
  model_used?: string;
  word_count?: number;
  length_bucket?: string;
  cleaned_text?: string;
  influential_words?: InfluentialWord[];
  is_mock?: boolean;
}

const CANDIDATE_URLS = [
  import.meta.env.VITE_API_BASE_URL,
  'http://127.0.0.1:8001',
  'http://127.0.0.1:8000',
].filter(Boolean) as string[];

let activeApiBaseUrl: string | null = 'http://127.0.0.1:8001';

const POSITIVE_LEXICON: Record<string, number> = {
  great: 6.4,
  love: 5.8,
  comfortable: 4.4,
  perfect: 3.9,
  good: 3.1,
  nice: 2.8,
  fit: 2.6,
  cute: 2.4,
  excellent: 5.2,
  amazing: 6.1,
  phenomenal: 5.9,
  fantastic: 5.5,
  best: 4.8,
  awesome: 4.5,
  recommend: 3.6,
  warm: 2.2,
  stylish: 2.5,
  durable: 3.8,
};

const NEGATIVE_LEXICON: Record<string, number> = {
  not: -6.6,
  disappointed: -4.1,
  poor: -3.5,
  uncomfortable: -3.0,
  cheap: -2.8,
  returned: -2.6,
  bad: -2.5,
  terrible: -5.8,
  horrible: -5.5,
  broke: -4.3,
  broken: -4.2,
  waste: -4.6,
  useless: -3.9,
  awful: -5.0,
  worst: -5.4,
  smell: -2.2,
  thin: -1.8,
  stiff: -1.5,
};

export interface TokenItem {
  token: string;
  tf: number;
  idf?: number;
  tfidf_score?: number;
  coefficient?: number;
  sentiment_contribution?: number;
  influence: 'Positive' | 'Negative' | 'Neutral';
  is_ngram?: boolean;
  semantic_similarity?: number;
  importance_score?: number;
  sim_to_positive?: number;
  sim_to_negative?: number;
}

export interface DenseDimensionItem {
  dimension: number;
  embedding_value: number;
  dimension_weight: number;
  activation_contribution: number;
  influence: 'Positive' | 'Negative';
}

export interface TokenAnalysisResult {
  model_type: string;
  tokens: TokenItem[];
  total_tokens_found: number;
  intercept?: number;
  total_positive_weight?: number;
  total_negative_weight?: number;
  net_logit?: number;
  base_probability?: number;
  dense_dimensions?: DenseDimensionItem[];
}

export interface TokenAnalysisResponse {
  review_id: string;
  product_name: string;
  rating: number;
  text: string;
  cleaned_text: string;
  word_count: number;
  length_bucket: string;
  sentiment: 'Positive' | 'Negative' | 'Neutral';
  prediction: 'Positive' | 'Negative' | 'Neutral';
  confidence: number;
  probabilities: {
    Positive: number;
    Negative: number;
    Neutral: number;
  };
  model_choice: 'tfidf' | 'baseline' | 'advanced';
  model_used: string;
  token_analysis: TokenAnalysisResult;
  is_mock?: boolean;
}

export interface AnalyzeTokensRequest {
  text?: string;
  review_id?: string;
  product_name?: string;
  rating?: number;
  model_choice?: 'tfidf' | 'baseline' | 'advanced';
}

export async function checkBackendHealth(): Promise<boolean> {
  for (const url of CANDIDATE_URLS) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200);
      const res = await fetch(`${url}/docs`, {
        method: 'GET',
        signal: controller.signal,
        mode: 'cors',
      });
      clearTimeout(timeoutId);
      if (res.ok || res.status === 404) {
        activeApiBaseUrl = url;
        return true;
      }
    } catch {
      // Continue to next candidate
    }
  }
  return false;
}

export async function fetchDatasetReviewsSample(q: string = '', limit: number = 15): Promise<any[]> {
  const urlsToTry = activeApiBaseUrl ? [activeApiBaseUrl, ...CANDIDATE_URLS] : CANDIDATE_URLS;
  for (const url of urlsToTry) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${url}/reviews?limit=${limit}&q=${encodeURIComponent(q)}`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        return data.reviews || [];
      }
    } catch {
      // Fallback
    }
  }
  return [];
}

export async function analyzeReviewTokens(
  req: AnalyzeTokensRequest,
  forceMock = false
): Promise<TokenAnalysisResponse> {
  if (!forceMock) {
    const urlsToTry = activeApiBaseUrl ? [activeApiBaseUrl, ...CANDIDATE_URLS] : CANDIDATE_URLS;
    for (const url of urlsToTry) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4500);

        const response = await fetch(`${url}/analyze_tokens`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: req.text || '',
            review_id: req.review_id || '',
            product_name: req.product_name || '',
            rating: req.rating,
            model_choice: req.model_choice || 'tfidf',
          }),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (response.ok) {
          activeApiBaseUrl = url;
          const data = await response.json();
          return {
            ...data,
            is_mock: false,
          };
        }
      } catch {
        console.warn('Backend unavailable, switching token analysis to offline simulation...');
      }
    }
  }

  return simulateLocalTokenAnalysis(
    req.text || '',
    req.model_choice || 'tfidf',
    req.review_id,
    req.product_name,
    req.rating
  );
}

export async function predictSentiment(
  req: PredictRequest,
  forceMock = false
): Promise<PredictResponse> {
  if (!forceMock) {
    const urlsToTry = activeApiBaseUrl ? [activeApiBaseUrl, ...CANDIDATE_URLS] : CANDIDATE_URLS;
    for (const url of urlsToTry) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3500);

        const response = await fetch(`${url}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: req.text || '',
            review_id: req.review_id || '',
            model_choice: req.model_choice || 'tfidf',
          }),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (response.ok) {
          activeApiBaseUrl = url;
          const data = await response.json();
          return {
            sentiment: data.sentiment || data.prediction,
            confidence: data.confidence > 1 ? data.confidence / 100 : data.confidence,
            probabilities: data.probabilities || {
              Positive: data.probability_positive || 0.5,
              Negative: 1 - (data.probability_positive || 0.5),
              Neutral: 0.1,
            },
            probability_positive: data.probability_positive,
            model_used: data.model_used || 'TF-IDF + Logistic Regression (Live API)',
            word_count: data.word_count || (req.text ? req.text.split(/\s+/).length : 12),
            length_bucket: data.length_bucket,
            cleaned_text: data.cleaned_text || req.text,
            influential_words: data.influential_words || [],
            is_mock: false,
          };
        }
      } catch {
        console.warn('Backend unavailable, engaging intelligent offline NLP engine...');
      }
    }
  }

  // Graceful offline fallback simulation
  return simulateLocalInference(req.text || '', req.model_choice);
}

function simulateLocalInference(text: string, modelChoice: 'tfidf' | 'baseline' | 'advanced' = 'tfidf'): PredictResponse {
  const lower = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ');
  const words = lower.split(/\s+/).filter(Boolean);
  const wordCount = words.length;

  let score = 0;
  const influentialWords: InfluentialWord[] = [];

  // 1. Explicit rating detection (Five stars, 1 star, etc.)
  if (/\b(5\s*stars?|five\s*stars?|5\/5|5\s*out\s*of\s*5|5\s*star)\b/i.test(text)) {
    score += 6.5;
    influentialWords.push({ word: 'five stars', weight: 6.5, influence: 'Positive' });
  } else if (/\b(4\s*stars?|four\s*stars?|4\/5|4\s*out\s*of\s*5|4\s*star)\b/i.test(text)) {
    score += 3.5;
    influentialWords.push({ word: 'four stars', weight: 3.5, influence: 'Positive' });
  } else if (/\b(1\s*stars?|one\s*stars?|1\/5|1\s*out\s*of\s*5|zero\s*stars?|0\s*stars?|1\s*star)\b/i.test(text)) {
    score -= 6.5;
    influentialWords.push({ word: 'one star', weight: -6.5, influence: 'Negative' });
  } else if (/\b(2\s*stars?|two\s*stars?|2\/5|2\s*star)\b/i.test(text)) {
    score -= 3.5;
    influentialWords.push({ word: 'two stars', weight: -3.5, influence: 'Negative' });
  }

  // 2. High-impact sentiment idioms
  if (/\bwon\s+(?:me|us|everyone)\s+over\b/i.test(text)) {
    score += 5.2;
    influentialWords.push({ word: 'won me over', weight: 5.2, influence: 'Positive' });
  }
  if (/\bwaste\s+of\s+money\b/i.test(text)) {
    score -= 5.5;
    influentialWords.push({ word: 'waste of money', weight: -5.5, influence: 'Negative' });
  }
  if (/\bdo\s+not\s+buy\b/i.test(text)) {
    score -= 5.0;
    influentialWords.push({ word: 'do not buy', weight: -5.0, influence: 'Negative' });
  }
  if (/\bhighly\s+recommend\b/i.test(text)) {
    score += 4.5;
    influentialWords.push({ word: 'highly recommend', weight: 4.5, influence: 'Positive' });
  }

  // 3. Contrastive discourse awareness ('A, but B')
  // The clause after 'but' represents the decisive verdict
  const contrastParts = text.split(/\b(but|however|yet|although|nevertheless|though)\b/i);
  let preScore = 0;
  let postScore = 0;

  if (contrastParts.length > 2) {
    const preText = contrastParts[0].toLowerCase();
    const postText = contrastParts.slice(1).join(' ').toLowerCase();

    for (const w of preText.split(/\s+/).filter(Boolean)) {
      if (POSITIVE_LEXICON[w]) preScore += POSITIVE_LEXICON[w];
      if (NEGATIVE_LEXICON[w]) preScore += NEGATIVE_LEXICON[w];
    }
    for (const w of postText.split(/\s+/).filter(Boolean)) {
      if (POSITIVE_LEXICON[w]) postScore += POSITIVE_LEXICON[w];
      if (NEGATIVE_LEXICON[w]) postScore += NEGATIVE_LEXICON[w];
    }
    // Concession is attenuated (0.35x), final verdict is amplified (1.8x)
    score += preScore * 0.35 + postScore * 1.8;
  } else {
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      const prev = i > 0 ? words[i - 1] : '';

      if (POSITIVE_LEXICON[w]) {
        const isNegated = prev === 'not' || prev === 'never' || prev === 'hardly';
        const weight = isNegated ? -POSITIVE_LEXICON[w] * 0.8 : POSITIVE_LEXICON[w];
        score += weight;
        influentialWords.push({
          word: isNegated ? `${prev} ${w}` : w,
          weight: Number(weight.toFixed(2)),
          influence: weight > 0 ? 'Positive' : 'Negative',
        });
      } else if (NEGATIVE_LEXICON[w]) {
        const weight = NEGATIVE_LEXICON[w];
        score += weight;
        influentialWords.push({
          word: w,
          weight: Number(weight.toFixed(2)),
          influence: 'Negative',
        });
      }
    }
  }

  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z * 0.5));
  const rawProb = sigmoid(score);

  let sentiment: 'Positive' | 'Negative' | 'Neutral';
  let conf: number;
  let pPos: number;
  let pNeg: number;
  let pNeu: number;

  if (rawProb >= 0.58) {
    sentiment = 'Positive';
    pPos = Number(rawProb.toFixed(3));
    pNeu = Number(((1 - rawProb) * 0.4).toFixed(3));
    pNeg = Number(Math.max(0, 1 - pPos - pNeu).toFixed(3));
    conf = pPos;
  } else if (rawProb <= 0.42) {
    sentiment = 'Negative';
    pNeg = Number((1 - rawProb).toFixed(3));
    pNeu = Number((rawProb * 0.4).toFixed(3));
    pPos = Number(Math.max(0, 1 - pNeg - pNeu).toFixed(3));
    conf = pNeg;
  } else {
    sentiment = 'Neutral';
    pNeu = Number((0.55 + Math.abs(0.5 - rawProb) * 0.3).toFixed(3));
    pPos = Number(((1 - pNeu) * 0.52).toFixed(3));
    pNeg = Number(Math.max(0, 1 - pNeu - pPos).toFixed(3));
    conf = pNeu;
  }

  const lengthBucket = wordCount <= 20 ? 'Short' : wordCount <= 49 ? 'Medium' : 'Detailed';

  return {
    sentiment,
    confidence: conf,
    probabilities: {
      Positive: pPos,
      Negative: pNeg,
      Neutral: pNeu,
    },
    probability_positive: pPos,
    model_used: 'TF-IDF + Logistic Regression [Offline Engine]',
    word_count: wordCount,
    length_bucket: lengthBucket,
    cleaned_text: words.join(' '),
    influential_words: influentialWords.slice(0, 6),
    is_mock: true,
  };
}

export function simulateLocalTokenAnalysis(
  text: string,
  modelChoice: 'tfidf' | 'baseline' | 'advanced' = 'tfidf',
  reviewId?: string,
  productName?: string,
  rating?: number
): TokenAnalysisResponse {
  const inference = simulateLocalInference(text, modelChoice);
  const lower = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ');
  const words = lower.split(/\s+/).filter(Boolean);

  const tokenItems: TokenItem[] = [];

  // Check n-grams & key tone phrases
  if (lower.includes('not comfortable')) {
    tokenItems.push({
      token: 'not comfortable',
      tf: 1,
      idf: 5.7356,
      tfidf_score: 0.2283,
      coefficient: -2.1781,
      sentiment_contribution: -0.4973,
      influence: 'Negative',
      is_ngram: true,
      semantic_similarity: -0.324,
      importance_score: -0.185,
    });
  }
  if (/\b(5\s*stars?|five\s*stars?)\b/i.test(text)) {
    tokenItems.push({
      token: 'five stars',
      tf: 1,
      idf: 6.8421,
      tfidf_score: 0.3125,
      coefficient: 5.241,
      sentiment_contribution: 1.6378,
      influence: 'Positive',
      is_ngram: true,
      semantic_similarity: 0.582,
      importance_score: 0.45,
    });
  }
  if (/\bwon\s+(?:me|us|everyone)\s+over\b/i.test(text)) {
    tokenItems.push({
      token: 'won me over',
      tf: 1,
      idf: 7.1245,
      tfidf_score: 0.2845,
      coefficient: 4.852,
      sentiment_contribution: 1.3804,
      influence: 'Positive',
      is_ngram: true,
      semantic_similarity: 0.512,
      importance_score: 0.38,
    });
  }
  if (/\bwaste\s+of\s+money\b/i.test(text)) {
    tokenItems.push({
      token: 'waste of money',
      tf: 1,
      idf: 7.214,
      tfidf_score: 0.298,
      coefficient: -5.124,
      sentiment_contribution: -1.5269,
      influence: 'Negative',
      is_ngram: true,
      semantic_similarity: -0.54,
      importance_score: -0.42,
    });
  }

  // Token items for vocabulary
  const seen = new Set<string>();
  for (const w of words) {
    if (seen.has(w) || w.length < 2) continue;
    seen.add(w);

    const tf = words.filter((x) => x === w).length;
    const isPos = Boolean(POSITIVE_LEXICON[w]);
    const isNeg = Boolean(NEGATIVE_LEXICON[w]);

    const coef = isPos ? POSITIVE_LEXICON[w] : isNeg ? NEGATIVE_LEXICON[w] : (Math.sin(w.length) * 0.8);
    const idf = Number((2.0 + (w.length % 5) * 0.75).toFixed(4));
    const tfidf = Number((Math.min(0.4, (1 + Math.log(tf)) * (idf / 10))).toFixed(4));
    const contrib = Number((coef * tfidf).toFixed(4));

    const simPos = Number((isPos ? 0.35 : isNeg ? -0.25 : 0.05).toFixed(3));
    const simNeg = Number((isNeg ? 0.40 : isPos ? -0.20 : 0.04).toFixed(3));
    const semanticSim = Number((simPos - simNeg).toFixed(4));
    const importance = Number((contrib * 0.25).toFixed(4));

    tokenItems.push({
      token: w,
      tf,
      idf,
      tfidf_score: tfidf,
      coefficient: Number(coef.toFixed(3)),
      sentiment_contribution: contrib,
      influence: contrib > 0 ? 'Positive' : contrib < 0 ? 'Negative' : 'Neutral',
      is_ngram: false,
      semantic_similarity: semanticSim,
      importance_score: importance,
      sim_to_positive: simPos,
      sim_to_negative: simNeg,
    });
  }

  tokenItems.sort((a, b) => Math.abs(b.sentiment_contribution || 0) - Math.abs(a.sentiment_contribution || 0));

  const totalPos = Number(tokenItems.filter((t) => (t.sentiment_contribution || 0) > 0).reduce((s, t) => s + (t.sentiment_contribution || 0), 0).toFixed(4));
  const totalNeg = Number(tokenItems.filter((t) => (t.sentiment_contribution || 0) < 0).reduce((s, t) => s + (t.sentiment_contribution || 0), 0).toFixed(4));

  const denseDims: DenseDimensionItem[] = [
    { dimension: 224, embedding_value: 0.0932, dimension_weight: 2.6532, activation_contribution: 0.2473, influence: 'Positive' },
    { dimension: 61, embedding_value: 0.1160, dimension_weight: 1.7469, activation_contribution: 0.2026, influence: 'Positive' },
    { dimension: 188, embedding_value: -0.0841, dimension_weight: -2.1045, activation_contribution: 0.1770, influence: 'Positive' },
    { dimension: 312, embedding_value: 0.0715, dimension_weight: -2.3140, activation_contribution: -0.1655, influence: 'Negative' },
    { dimension: 45, embedding_value: -0.0924, dimension_weight: 1.5540, activation_contribution: -0.1436, influence: 'Negative' },
  ];

  return {
    review_id: reviewId || 'REV-SIMULATED',
    product_name: productName || 'Evaluated Review Item',
    rating: rating !== undefined && !isNaN(rating) ? rating : (inference.sentiment === 'Positive' ? 5 : inference.sentiment === 'Negative' ? 1 : 3),
    text,
    cleaned_text: words.join(' '),
    word_count: words.length,
    length_bucket: inference.length_bucket || 'Short',
    sentiment: inference.sentiment,
    prediction: inference.sentiment,
    confidence: inference.confidence,
    probabilities: inference.probabilities,
    model_choice: modelChoice,
    model_used: 'TF-IDF + Logistic Regression [Vocabulary Token Decomposition]',
    token_analysis: {
      model_type: 'TF-IDF + Logistic Regression',
      tokens: tokenItems,
      total_tokens_found: tokenItems.length,
      intercept: 0.0124,
      total_positive_weight: totalPos,
      total_negative_weight: totalNeg,
      net_logit: Number((totalPos + totalNeg).toFixed(4)),
      base_probability: inference.confidence,
      dense_dimensions: denseDims,
    },
    is_mock: true,
  };
}

export async function syncDatasetWithBackend(
  reviews: any[],
  datasetName: string = 'uploaded_dataset'
): Promise<boolean> {
  const urlsToTry = activeApiBaseUrl ? [activeApiBaseUrl, ...CANDIDATE_URLS] : CANDIDATE_URLS;
  for (const url of urlsToTry) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);
      const res = await fetch(`${url}/sync_dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviews: reviews.slice(0, 1000),
          dataset_name: datasetName,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        return true;
      }
    } catch {
      // Ignore fallback
    }
  }
  return false;
}

export async function fetchActiveDatasetFromBackend(limit: number = 1000): Promise<{ name: string; reviews: ReviewSample[] } | null> {
  const urlsToTry = activeApiBaseUrl ? [activeApiBaseUrl, ...CANDIDATE_URLS] : CANDIDATE_URLS;
  for (const url of urlsToTry) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);
      const res = await fetch(`${url}/active_dataset?limit=${limit}`, {
        method: 'GET',
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        activeApiBaseUrl = url;
        const data = await res.json();
        if (data.status === 'success' && Array.isArray(data.reviews) && data.reviews.length > 0) {
          const mapped: ReviewSample[] = data.reviews.map((r: any, idx: number) => {
            const idVal = String(r.Review_ID || r.review_id || r.id || `REV-${idx + 1001}`).trim();
            const textVal = String(r.Review_Text || r.review_text || r.review_body || r.text || '').trim();
            const prodVal = String(r.Product_Name || r.product_name || r.product_category || 'Customer Product').trim();
            const starsVal = Number(r.Rating ?? r.rating ?? r.stars ?? 5);
            const sentVal = String(r.Sentiments || r.sentiment || (starsVal >= 4 ? 'Positive' : starsVal <= 2 ? 'Negative' : 'Neutral'));
            return {
              Review_ID: idVal,
              id: idVal,
              Product_Name: prodVal,
              product_category: prodVal,
              Rating: starsVal,
              stars: starsVal,
              Review_Text: textVal,
              review_text: textVal,
              Sentiments: sentVal as any,
              sentiment: sentVal as any,
              word_count: textVal.split(/\s+/).filter(Boolean).length || 10,
            };
          });
          return { name: data.name || 'active_uploaded_reviews.csv', reviews: mapped };
        }
      }
    } catch {
      // Continue
    }
  }
  return null;
}

export interface FeatureImportanceResponse {
  status: string;
  scope: 'review' | 'dataset' | 'global';
  review_id?: string;
  review_text?: string;
  product_name?: string;
  rating?: number;
  sentiment?: string;
  model_used?: string;
  total_reviews?: number;
  top_positive: FeatureWeight[];
  top_negative: FeatureWeight[];
  all_features?: FeatureWeight[];
}

export async function fetchFeatureImportance(params: {
  scope: 'review' | 'dataset' | 'global';
  review_id?: string;
  text?: string;
  model_choice?: 'tfidf' | 'baseline' | 'advanced';
  limit?: number;
}): Promise<FeatureImportanceResponse> {
  const urlsToTry = activeApiBaseUrl ? [activeApiBaseUrl, ...CANDIDATE_URLS] : CANDIDATE_URLS;
  for (const url of urlsToTry) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);
      const res = await fetch(`${url}/feature_importance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scope: params.scope,
          review_id: params.review_id || '',
          text: params.text || '',
          model_choice: params.model_choice || 'tfidf',
          limit: params.limit || 8,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        activeApiBaseUrl = url;
        const data = await res.json();
        return data;
      }
    } catch {
      // Continue to next URL
    }
  }

  // Fallback to local simulation if offline
  return simulateLocalFeatureImportance(params);
}

function simulateLocalFeatureImportance(params: {
  scope: 'review' | 'dataset' | 'global';
  review_id?: string;
  text?: string;
  model_choice?: 'tfidf' | 'baseline' | 'advanced';
  limit?: number;
}): FeatureImportanceResponse {
  const limit = params.limit || 8;

  if (params.scope === 'global') {
    const pos = TOP_FEATURE_WEIGHTS.filter((f) => f.sentiment === 'Positive').slice(0, limit);
    const neg = TOP_FEATURE_WEIGHTS.filter((f) => f.sentiment === 'Negative').slice(0, limit);
    return {
      status: 'success',
      scope: 'global',
      model_used: 'TF-IDF + Logistic Regression',
      top_positive: pos,
      top_negative: neg,
      all_features: [...pos, ...neg],
    };
  }

  // Review scope simulation
  const reviewText = params.text || "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me.";
  const sim = simulateLocalTokenAnalysis(reviewText, params.model_choice || 'tfidf');
  const tokens = sim.token_analysis?.tokens || [];

  const features: FeatureWeight[] = tokens.map((t) => {
    const rawWord = t.token;
    const displayWord = rawWord
      .replace(/explicit_five_star/g, 'five stars')
      .replace(/explicit_four_star/g, 'four stars')
      .replace(/explicit_one_star/g, 'one star')
      .replace(/explicit_two_star/g, 'two stars')
      .replace(/won_me_over/g, 'won me over')
      .replace(/not_fan_of/g, 'not a fan of')
      .replace(/not_/g, 'not ');

    const weightVal = Number(
      t.coefficient !== undefined
        ? t.coefficient
        : (t.importance_score !== undefined
          ? t.importance_score
          : (t.sentiment_contribution !== undefined ? t.sentiment_contribution : 1.0))
    );

    return {
      word: displayWord,
      weight: Number(weightVal.toFixed(2)),
      sentiment: (t.influence === 'Positive' || weightVal > 0) ? 'Positive' : 'Negative',
      category: t.is_ngram ? 'Preserved N-Gram' : 'Review Token',
      context: `TF: ${t.tf || 1} | Score: ${t.tfidf_score !== undefined ? t.tfidf_score : weightVal.toFixed(2)}`,
    };
  });

  const pos = features.filter((f) => f.sentiment === 'Positive').slice(0, limit);
  const neg = features.filter((f) => f.sentiment === 'Negative').slice(0, limit);

  return {
    status: 'success',
    scope: 'review',
    review_id: params.review_id || 'FMCY_00003',
    review_text: reviewText,
    product_name: 'Product Review',
    model_used: 'TF-IDF + Logistic Regression',
    top_positive: pos,
    top_negative: neg,
    all_features: features,
  };
}

export interface DynamicEvaluationResponse {
  status: string;
  dataset_name: string;
  total: number;
  model_choice: 'tfidf' | 'baseline' | 'advanced';
  is_three_class: boolean;
  labels: string[];
  matrix: number[][];
  tp?: number;
  fp?: number;
  tn?: number;
  fn?: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  macro_f1: number;
  subgroups: {
    slice: string;
    count: number;
    f1: number;
    disparity: string;
  }[];
}

export async function fetchModelEvaluation(
  modelChoice: 'tfidf' | 'baseline' | 'advanced' = 'tfidf',
  reviews?: any[],
  datasetName?: string
): Promise<DynamicEvaluationResponse | null> {
  const urlsToTry = activeApiBaseUrl ? [activeApiBaseUrl, ...CANDIDATE_URLS] : CANDIDATE_URLS;
  for (const url of urlsToTry) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4500);
      let res: Response;
      if (reviews && reviews.length > 0) {
        res = await fetch(`${url}/model_evaluation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_choice: modelChoice,
            reviews: reviews.slice(0, 1500),
            dataset_name: datasetName,
          }),
          signal: controller.signal,
        });
      } else {
        res = await fetch(`${url}/model_evaluation?model_choice=${modelChoice}`, {
          signal: controller.signal,
        });
      }
      clearTimeout(timeoutId);
      if (res.ok) {
        activeApiBaseUrl = url;
        const data = await res.json();
        if (data && data.status === 'success' && data.total > 0) {
          return data;
        }
      }
    } catch {
      // Continue candidate scan
    }
  }
  return null;
}


