import React, { useState, useEffect } from 'react';
import {
  Send,
  Sparkles,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Hash,
  Type,
  Search,
  Layers,
  TrendingUp,
  TrendingDown,
  Cpu,
  Zap,
  CheckCircle2,
} from 'lucide-react';
import { fireSafeConfetti } from '../../utils/confetti';
import { ThinkingOrbCanvas } from '../3d/ThinkingOrbCanvas';
import { PRESET_REVIEWS, ReviewSample } from '../../data/sampleData';
import {
  getActiveDataset,
  getActiveDatasetName,
  subscribeToDataset,
  findReviewInActiveDataset,
  setActiveReviewTokens,
} from '../../store/datasetStore';
import {
  analyzeReviewTokens,
  TokenAnalysisResponse,
} from '../../services/api';
import { SentimentState } from '../3d/HeroSentimentCenterpiece';

interface PredictionSectionProps {
  onSentimentResult: (sentiment: SentimentState) => void;
}

export const PredictionSection: React.FC<PredictionSectionProps> = ({ onSentimentResult }) => {
  const initialDataset = getActiveDataset();
  const initialReview = initialDataset[0] || null;

  // Active dataset state
  const [activeDataset, setActiveDatasetState] = useState<ReviewSample[]>(initialDataset);
  const [activeDatasetName, setActiveDatasetName] = useState<string>(getActiveDatasetName());

  // Input modes: 'review_id' | 'custom_text'
  const [inputMode, setInputMode] = useState<'review_id' | 'custom_text'>('review_id');
  const [selectedReviewId, setSelectedReviewId] = useState<string>(
    initialReview ? (initialReview.Review_ID || initialReview.id || 'REV-10491') : 'REV-10491'
  );
  const [inputText, setInputText] = useState<string>(
    initialReview ? (initialReview.Review_Text || initialReview.review_text || '') : ''
  );
  const [matchedReviewMeta, setMatchedReviewMeta] = useState<ReviewSample | null>(initialReview);

  // Model choice: 'baseline' (TF-IDF + LogReg) | 'advanced' (Sentence-Transformers MiniLM)
  const [modelChoice, setModelChoice] = useState<'baseline' | 'advanced'>('baseline');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Analysis result state
  const [tokenResult, setTokenResult] = useState<TokenAnalysisResponse | null>(null);
  const [tokenFilter, setTokenFilter] = useState<'all' | 'positive' | 'negative' | 'ngrams'>('all');
  const [debugOpen, setDebugOpen] = useState(false);

  const executeAnalysis = async (text: string, reviewId: string, model: 'baseline' | 'advanced') => {
    if (!text.trim() && !reviewId.trim()) {
      setError('Please provide a valid Review ID or enter review text.');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const res = await analyzeReviewTokens({
        text: text.trim(),
        review_id: reviewId.trim(),
        model_choice: model,
      });

      setTokenResult(res);
      onSentimentResult(res.sentiment);

      // Publish active review tokens to shared store for 3D Feature Importance & Model Coefficients
      if (res.token_analysis?.tokens) {
        setActiveReviewTokens(res.token_analysis.tokens, {
          review_id: res.review_id || reviewId,
          review_text: res.text || text,
          product_name: res.product_name,
          rating: res.rating,
          sentiment: res.sentiment,
        });
      }

      // If backend resolved review text or metadata from active dataset, populate UI
      if (res.text && (!text.trim() || inputMode === 'review_id')) {
        setInputText(res.text);
      }
      if (res.review_id && inputMode === 'review_id') {
        setSelectedReviewId(res.review_id);
      }
      if (res.product_name || res.review_id) {
        setMatchedReviewMeta((prev) => ({
          ...(prev || {}),
          Review_ID: res.review_id || reviewId,
          Product_Name: res.product_name || prev?.Product_Name || 'Product Review',
          Rating: res.rating ?? prev?.Rating ?? 5,
          Review_Text: res.text || text || '',
          Sentiments: res.sentiment as any,
          word_count: res.word_count || (res.text ? res.text.split(/\s+/).filter(Boolean).length : 10),
        }));
      }

      if (res.sentiment === 'Positive') {
        fireSafeConfetti({
          particleCount: 25,
          spread: 45,
          origin: { y: 0.6 },
        });
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to analyze review tokens.');
    } finally {
      setIsLoading(false);
    }
  };

  // Initial load: analyze default review
  useEffect(() => {
    executeAnalysis(inputText, selectedReviewId, modelChoice);
  }, []);

  // Listen for custom trigger from Dataset Ingestion section
  useEffect(() => {
    const handleDatasetTrigger = (event: Event) => {
      const customEvent = event as CustomEvent<{
        reviewId: string;
        reviewText: string;
        productName?: string;
        rating?: number;
      }>;
      if (customEvent.detail) {
        const { reviewId, reviewText, productName, rating } = customEvent.detail;
        setInputMode('review_id');
        setSelectedReviewId(reviewId);
        setInputText(reviewText);
        setMatchedReviewMeta({
          Review_ID: reviewId,
          Product_Name: productName || 'Selected Review Product',
          Rating: rating || 5,
          Review_Text: reviewText,
          Sentiments: (rating && rating >= 4 ? 'Positive' : rating && rating <= 2 ? 'Negative' : 'Neutral') as any,
          word_count: reviewText.split(/\s+/).filter(Boolean).length,
        });

        executeAnalysis(reviewText, reviewId, modelChoice);

        const el = document.getElementById('prediction-interface');
        if (el) {
          el.scrollIntoView({ behavior: 'smooth' });
        }
      }
    };

    window.addEventListener('analyze-dataset-review', handleDatasetTrigger);
    return () => {
      window.removeEventListener('analyze-dataset-review', handleDatasetTrigger);
    };
  }, [modelChoice]);

  // Subscribe to uploaded dataset changes
  useEffect(() => {
    const unsubscribe = subscribeToDataset((dataset, name) => {
      setActiveDatasetState(dataset);
      setActiveDatasetName(name);

      if (dataset.length > 0) {
        const found = findReviewInActiveDataset(selectedReviewId);
        const target = found || dataset[0];
        if (target) {
          const revId = target.Review_ID || target.id || '';
          const revText = target.Review_Text || target.review_text || '';
          setSelectedReviewId(revId);
          setInputText(revText);
          setMatchedReviewMeta(target);
          setError(null);
          executeAnalysis(revText, revId, modelChoice);
        }
      }
    });

    return unsubscribe;
  }, [modelChoice, selectedReviewId]);

  // Handle Review ID Selection or Typing
  const handleReviewIdChange = (id: string, autoExecute: boolean = false) => {
    setSelectedReviewId(id);
    const found = findReviewInActiveDataset(id);

    if (found) {
      setMatchedReviewMeta(found);
      const text = found.Review_Text || found.review_text || '';
      setInputText(text);
      setError(null);
      if (autoExecute) {
        executeAnalysis(text, id, modelChoice);
      }
    } else {
      // If not yet in local memory, pass empty string to backend so backend resolves directly from its dataset!
      if (autoExecute && id.trim()) {
        executeAnalysis('', id, modelChoice);
      }
    }
  };

  const handleRunAnalysis = () => {
    if (inputMode === 'review_id') {
      const found = findReviewInActiveDataset(selectedReviewId);
      const text = found ? (found.Review_Text || found.review_text || '') : '';
      executeAnalysis(text, selectedReviewId, modelChoice);
    } else {
      executeAnalysis(inputText, '', modelChoice);
    }
  };

  const handleModelChange = (model: 'baseline' | 'advanced') => {
    setModelChoice(model);
    if (inputMode === 'review_id') {
      const found = findReviewInActiveDataset(selectedReviewId);
      const text = found ? (found.Review_Text || found.review_text || '') : inputText;
      executeAnalysis(text, selectedReviewId, model);
    } else {
      executeAnalysis(inputText, '', model);
    }
  };

  const allTokens = tokenResult?.token_analysis?.tokens || [];
  const filteredTokens = allTokens.filter((t) => {
    if (tokenFilter === 'positive') return t.influence === 'Positive';
    if (tokenFilter === 'negative') return t.influence === 'Negative';
    if (tokenFilter === 'ngrams') return Boolean(t.is_ngram);
    return true;
  });

  const sentimentColorClass =
    tokenResult?.sentiment === 'Positive'
      ? 'dark:text-emerald-400 text-emerald-800 dark:bg-emerald-500/10 bg-emerald-100 dark:border-emerald-500/30 border-emerald-300'
      : tokenResult?.sentiment === 'Negative'
      ? 'dark:text-rose-400 text-rose-800 dark:bg-rose-500/10 bg-rose-100 dark:border-rose-500/30 border-rose-300'
      : 'dark:text-amber-400 text-amber-800 dark:bg-amber-500/10 bg-amber-100 dark:border-amber-500/30 border-amber-300';

  const confidencePercent = tokenResult ? Math.round(tokenResult.confidence * 100) : 0;

  return (
    <section id="prediction-interface" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Section Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 mb-3 shadow-sm">
          <Sparkles className="w-3.5 h-3.5" />
          Interactive Token Vocabulary &amp; Model Analyzer
        </div>
        <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight text-slate-900 dark:text-white">
          TF-IDF Review Token Analyzer
        </h2>
        <p className="text-slate-600 dark:text-slate-300 text-sm sm:text-base max-w-2xl mx-auto mt-2 leading-relaxed">
          Extract vocabulary tokens directly from dataset reviews by Review ID or enter custom text.
          Inspect <strong>Sublinear TF-IDF weights</strong>, n-gram preserving tokenization, and logistic regression polarity coefficients.
        </p>
      </div>

      {/* Top Grid: Input Form (Left) & Sentiment Prediction / Probabilities (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mb-8">
        {/* Left Column: Dual-Mode Input Form */}
        <div className="lg:col-span-7 rounded-2xl glass-panel p-6 sm:p-7 border border-surface-border space-y-5 shadow-sm">
          {/* Dual Input Mode Tabs (Requirement 2) */}
          <div className="flex items-center justify-between border-b border-surface-border pb-3">
            <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-100 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 text-xs">
              <button
                onClick={() => setInputMode('review_id')}
                className={`px-3 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 transition-all ${
                  inputMode === 'review_id'
                    ? 'bg-primary text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
                }`}
              >
                <Hash className="w-3.5 h-3.5" />
                <span>Lookup by Review ID</span>
              </button>
              <button
                onClick={() => setInputMode('custom_text')}
                className={`px-3 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 transition-all ${
                  inputMode === 'custom_text'
                    ? 'bg-primary text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
                }`}
              >
                <Type className="w-3.5 h-3.5" />
                <span>Custom Review Text</span>
              </button>
            </div>

            <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-500 dark:text-slate-400">
              <span className="hidden sm:inline">Active:</span>
              <span
                className="px-2 py-0.5 rounded-md bg-primary/10 text-primary border border-primary/20 font-semibold max-w-[150px] sm:max-w-[220px] truncate"
                title={activeDatasetName}
              >
                {activeDatasetName}
              </span>
              <span className="text-[10px] text-slate-400">({activeDataset.length} rows)</span>
            </div>
          </div>

          {/* Mode 1: Lookup by Review ID */}
          {inputMode === 'review_id' && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-mono uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5">
                  Enter or Select Review ID from Dataset:
                </label>
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
                    <input
                      type="text"
                      list="active-dataset-review-ids"
                      value={selectedReviewId}
                      onChange={(e) => handleReviewIdChange(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleReviewIdChange(selectedReviewId, true);
                        }
                      }}
                      placeholder="e.g. FMCY_00003, REV-10491, 3..."
                      className="w-full rounded-xl bg-slate-50 dark:bg-black/40 border border-surface-border pl-9 pr-4 py-2.5 text-sm font-mono text-foreground focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                    />
                    <datalist id="active-dataset-review-ids">
                      {activeDataset.slice(0, 100).map((rev) => {
                        const rId = rev.Review_ID || rev.id || '';
                        return (
                          <option key={rId} value={rId}>
                            {rev.Product_Name || 'Product'} ({rev.Rating || 5}★)
                          </option>
                        );
                      })}
                    </datalist>
                  </div>

                  <button
                    type="button"
                    onClick={() => handleReviewIdChange(selectedReviewId, true)}
                    disabled={isLoading || !selectedReviewId.trim()}
                    className="px-3.5 py-2.5 rounded-xl bg-primary text-primary-foreground text-xs font-semibold hover:bg-primary-hover transition-all shrink-0 flex items-center gap-1.5 disabled:opacity-50 shadow-sm"
                  >
                    <Search className="w-3.5 h-3.5" />
                    <span>Load & Analyze</span>
                  </button>
                </div>
              </div>

              {/* Direct Dataset Dropdown Selector */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                    Or select from uploaded dataset ({activeDataset.length} available):
                  </span>
                </div>
                <select
                  value={selectedReviewId}
                  onChange={(e) => handleReviewIdChange(e.target.value, true)}
                  className="w-full rounded-xl bg-slate-50 dark:bg-black/40 border border-surface-border px-3 py-2 text-xs font-mono text-foreground focus:outline-none focus:border-primary transition-all cursor-pointer"
                >
                  <option value="">-- Choose a Review ID from uploaded dataset --</option>
                  {activeDataset.slice(0, 50).map((rev) => {
                    const rId = rev.Review_ID || rev.id || '';
                    const rProd = (rev.Product_Name || rev.product_category || 'Product').slice(0, 30);
                    const rRating = rev.Rating || rev.stars || 5;
                    const rSnippet = (rev.Review_Text || rev.review_text || '').slice(0, 50);
                    return (
                      <option key={rId} value={rId}>
                        {rId} — {rRating}★ — {rProd} (&quot;{rSnippet}...&quot;)
                      </option>
                    );
                  })}
                </select>
              </div>

              {/* Quick Preset Review ID Pills from Active Dataset */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                    Quick Select:
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    Showing {Math.min(10, activeDataset.length)} of {activeDataset.length}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {activeDataset.slice(0, 10).map((rev) => {
                    const rId = rev.Review_ID || rev.id || '';
                    const rRating = rev.Rating || rev.stars || 5;
                    return (
                      <button
                        key={rId}
                        type="button"
                        onClick={() => handleReviewIdChange(rId, true)}
                        className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-all border ${
                          selectedReviewId === rId
                            ? 'bg-primary/15 text-primary border-primary font-bold shadow-sm'
                            : 'glass-panel text-slate-600 dark:text-slate-400 hover:border-primary/50'
                        }`}
                      >
                        {rId} ({rRating}★)
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Matched Product Metadata Card */}
              {matchedReviewMeta ? (
                <div className="p-3 rounded-xl bg-slate-50/80 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-xs space-y-1.5">
                  <div className="flex items-center gap-1.5 pb-1 border-b border-surface-border text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold">
                    <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                    <span>Loaded from Dataset: Review ID <strong>{matchedReviewMeta.Review_ID || selectedReviewId}</strong></span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-900 dark:text-white font-sans">
                      {matchedReviewMeta.Product_Name || matchedReviewMeta.product_category || 'Product Review'}
                    </span>
                    <span className="font-mono text-amber-500 font-bold">
                      {'★'.repeat(Math.max(1, Math.min(5, matchedReviewMeta.Rating || matchedReviewMeta.stars || 5)))} (
                      {matchedReviewMeta.Rating || matchedReviewMeta.stars || 5}/5)
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-3">
                    <span>
                      Ground Truth: <strong>{matchedReviewMeta.Sentiments || matchedReviewMeta.sentiment || 'Unknown'}</strong>
                    </span>
                    <span>
                      Length:{' '}
                      <strong>
                        {matchedReviewMeta.word_count ||
                          (matchedReviewMeta.Review_Text || matchedReviewMeta.review_text || '').split(/\s+/).filter(Boolean).length}{' '}
                        words
                      </strong>
                    </span>
                  </div>
                </div>
              ) : (
                <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-600 dark:text-amber-400">
                  Review ID &quot;{selectedReviewId}&quot; not found in active dataset ({activeDatasetName}). Pick from the dataset pills above or enter custom text below.
                </div>
              )}
            </div>
          )}

          {/* Mode 2: Custom Text Review */}
          {inputMode === 'custom_text' && (
            <div>
              <label className="block text-xs font-mono uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5">
                Quick Preset Example Reviews:
              </label>
              <div className="flex flex-wrap gap-1.5 mb-2">
                {PRESET_REVIEWS.map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => {
                      setInputText(preset.text);
                      setSelectedReviewId('CUSTOM-PRESET');
                      setMatchedReviewMeta(null);
                    }}
                    className="px-2.5 py-1 rounded-lg text-xs font-medium glass-panel hover:border-primary/50 text-foreground transition-all"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Review Textarea Display / Editor */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-mono uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Review Text for Analysis:
              </label>
              <span className="text-[11px] font-mono text-slate-400">
                {inputText.split(/\s+/).filter(Boolean).length} words | {inputText.length} chars
              </span>
            </div>

            <textarea
              rows={4}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="e.g. The boots are stylish and warm, but they are not comfortable for all-day standing..."
              className="w-full rounded-xl bg-slate-50 dark:bg-black/40 border border-surface-border p-3.5 text-sm text-foreground focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all resize-none leading-relaxed"
            />
          </div>

          {/* Algorithm Indicator & Run Button */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-500 dark:text-slate-400">Algorithm:</span>
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl glass-panel border border-primary/30 text-xs font-mono font-bold text-primary bg-primary/5">
                <Cpu className="w-3.5 h-3.5" />
                <span>TF-IDF (1–3 n-grams) + Logistic Regression</span>
              </div>
            </div>

            <button
              onClick={handleRunAnalysis}
              disabled={isLoading || !inputText.trim()}
              className="px-5 py-2.5 rounded-xl font-bold text-xs sm:text-sm bg-primary hover:bg-primary/90 text-white shadow-glow-primary transition-all duration-200 flex items-center justify-center gap-2 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Decomposing Tokens...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Run Token Analysis</span>
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-500 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results & Model Probabilities */}
        <div className="lg:col-span-5 rounded-2xl glass-panel-elevated p-6 sm:p-7 border border-surface-border flex flex-col justify-between relative overflow-hidden min-h-[420px] shadow-sm">
          {isLoading ? (
            <div className="flex-1 flex flex-col items-center justify-center py-12 text-center">
              <ThinkingOrbCanvas />
              <div className="text-base font-bold font-display text-foreground mt-4">
                Computing Token Decomposition...
              </div>
              <p className="text-xs text-foreground-muted mt-1 max-w-xs">
                {modelChoice === 'baseline'
                  ? 'Calculating sublinear TF, IDF corpus weights, and logistic regression coefficients.'
                  : 'Encoding 384-dimensional dense semantic vectors and contextual token attributions.'}
              </p>
            </div>
          ) : tokenResult ? (
            <div className="space-y-5">
              {/* Primary Prediction Label & Confidence Gauge */}
              <div className="flex items-center justify-between pb-4 border-b border-surface-border">
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                    Classification Result
                  </div>
                  <div className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-lg font-extrabold font-display border ${sentimentColorClass}`}>
                    <span>{tokenResult.sentiment} Sentiment</span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400 mt-1">
                    Review: <strong>{tokenResult.review_id}</strong>
                  </div>
                </div>

                {/* Animated Radial Gauge */}
                <div className="relative w-18 h-18 flex items-center justify-center shrink-0">
                  <svg className="w-18 h-18 transform -rotate-90" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="38"
                      stroke="currentColor"
                      strokeWidth="8"
                      className="text-slate-200 dark:text-slate-800 fill-transparent"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="38"
                      stroke="currentColor"
                      strokeWidth="8"
                      strokeDasharray={2 * Math.PI * 38}
                      strokeDashoffset={2 * Math.PI * 38 * (1 - tokenResult.confidence)}
                      strokeLinecap="round"
                      className={`transition-all duration-700 fill-transparent ${
                        tokenResult.sentiment === 'Positive'
                          ? 'text-emerald-500'
                          : tokenResult.sentiment === 'Negative'
                          ? 'text-rose-500'
                          : 'text-amber-500'
                      }`}
                    />
                  </svg>
                  <div className="absolute flex flex-col items-center text-center">
                    <span className="text-sm font-bold font-mono text-slate-900 dark:text-white">
                      {confidencePercent}%
                    </span>
                    <span className="text-[8px] font-mono uppercase text-slate-400">Conf.</span>
                  </div>
                </div>
              </div>

              {/* 3-Class Probabilities */}
              <div>
                <div className="text-xs font-mono uppercase text-slate-500 dark:text-slate-400 mb-2 flex items-center justify-between">
                  <span>Class Probabilities</span>
                  <span className="text-[10px] text-slate-400">Sum = 1.0</span>
                </div>

                <div className="space-y-2 font-mono text-xs">
                  <div>
                    <div className="flex justify-between text-[11px] mb-1">
                      <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Positive</span>
                      <span className="font-bold text-slate-900 dark:text-white">
                        {Math.round(tokenResult.probabilities.Positive * 100)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        style={{ width: `${tokenResult.probabilities.Positive * 100}%` }}
                        className="h-full bg-emerald-500 transition-all duration-500 rounded-full"
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] mb-1">
                      <span className="text-amber-600 dark:text-amber-400 font-semibold">Neutral</span>
                      <span className="font-bold text-slate-900 dark:text-white">
                        {Math.round(tokenResult.probabilities.Neutral * 100)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        style={{ width: `${tokenResult.probabilities.Neutral * 100}%` }}
                        className="h-full bg-amber-500 transition-all duration-500 rounded-full"
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] mb-1">
                      <span className="text-rose-600 dark:text-rose-400 font-semibold">Negative</span>
                      <span className="font-bold text-slate-900 dark:text-white">
                        {Math.round(tokenResult.probabilities.Negative * 100)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        style={{ width: `${tokenResult.probabilities.Negative * 100}%` }}
                        className="h-full bg-rose-500 transition-all duration-500 rounded-full"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick Summary Pill Banner */}
              <div className="pt-2 border-t border-surface-border grid grid-cols-3 gap-2 text-center text-[11px] font-mono">
                <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-surface-border">
                  <div className="text-slate-400 text-[9px] uppercase">Tokens</div>
                  <div className="font-bold text-slate-900 dark:text-white">
                    {tokenResult.token_analysis?.total_tokens_found || 0}
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-surface-border">
                  <div className="text-slate-400 text-[9px] uppercase">Length</div>
                  <div className="font-bold text-slate-900 dark:text-white">
                    {tokenResult.length_bucket}
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-surface-border">
                  <div className="text-slate-400 text-[9px] uppercase">Engine</div>
                  <div className={`font-bold ${tokenResult.is_mock ? 'dark:text-amber-400 text-amber-700' : 'dark:text-emerald-400 text-emerald-700'}`}>
                    {tokenResult.is_mock ? 'Offline' : 'FastAPI'}
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {/* Collapsible API Request / Response Debug Drawer */}
          <div className="mt-4 pt-3 border-t border-surface-border">
            <button
              onClick={() => setDebugOpen(!debugOpen)}
              className="w-full flex items-center justify-between text-xs font-mono text-slate-400 hover:text-foreground transition-colors"
            >
              <span>Developer API Request / Response Payload</span>
              {debugOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {debugOpen && (
              <div className="mt-2.5 p-3 rounded-xl bg-black/60 font-mono text-[10px] text-slate-300 space-y-2 overflow-x-auto">
                <div>
                  <span className="text-primary font-bold"># POST /analyze_tokens</span>
                  <pre className="text-slate-400 mt-1">
                    {JSON.stringify(
                      { review_id: selectedReviewId, text: inputText, model_choice: modelChoice },
                      null,
                      2
                    )}
                  </pre>
                </div>
                <div className="border-t border-slate-700/50 pt-2">
                  <span className="text-emerald-400 font-bold"># Response JSON (200 OK)</span>
                  <pre className="text-slate-300 mt-1">{JSON.stringify(tokenResult, null, 2)}</pre>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Requirement 3 & 4: Deep Vocabulary Token & Score Inspection Matrix */}
      {tokenResult?.token_analysis && (
        <div className="rounded-2xl glass-panel border border-surface-border p-6 shadow-sm space-y-5">
          {/* Table Header & Controls */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-surface-border">
            <div>
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-500" />
                <h3 className="text-base font-bold font-display text-slate-900 dark:text-white">
                  TF-IDF Vocabulary Token &amp; Weight Decomposition
                </h3>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                Displays all matching vocabulary tokens, term frequencies (TF), inverse document frequencies (IDF), sublinear vector scores, and learned logistic regression weights.
              </p>
            </div>

            {/* Token Filter Pills */}
            <div className="flex items-center gap-1 p-1 rounded-xl bg-slate-100 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 text-xs">
              <button
                onClick={() => setTokenFilter('all')}
                className={`px-2.5 py-1 rounded-lg font-semibold transition-all ${
                  tokenFilter === 'all'
                    ? 'bg-primary text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
                }`}
              >
                All Tokens ({allTokens.length})
              </button>
              <button
                onClick={() => setTokenFilter('positive')}
                className={`px-2.5 py-1 rounded-lg font-semibold transition-all ${
                  tokenFilter === 'positive'
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
                }`}
              >
                Positive Drivers
              </button>
              <button
                onClick={() => setTokenFilter('negative')}
                className={`px-2.5 py-1 rounded-lg font-semibold transition-all ${
                  tokenFilter === 'negative'
                    ? 'bg-rose-600 text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
                }`}
              >
                Negative Drivers
              </button>
              {modelChoice === 'baseline' && (
                <button
                  onClick={() => setTokenFilter('ngrams')}
                  className={`px-2.5 py-1 rounded-lg font-semibold transition-all ${
                    tokenFilter === 'ngrams'
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-foreground'
                  }`}
                >
                  Bi-Grams Only
                </button>
              )}
            </div>
          </div>

          {/* TF-IDF Logit Summary Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-surface-border">
              <div className="text-[10px] text-slate-500 dark:text-slate-400">Total Tokens Found</div>
              <div className="text-base font-bold text-slate-900 dark:text-white mt-0.5">
                {tokenResult.token_analysis.total_tokens_found}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800/40">
              <div className="text-[10px] text-emerald-600 dark:text-emerald-400">Positive Logit Weight</div>
              <div className="text-base font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">
                +{tokenResult.token_analysis.total_positive_weight || 0}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-rose-50/50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-800/40">
              <div className="text-[10px] text-rose-600 dark:text-rose-400">Negative Logit Weight</div>
              <div className="text-base font-bold text-rose-600 dark:text-rose-400 mt-0.5">
                {tokenResult.token_analysis.total_negative_weight || 0}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-200 dark:border-indigo-800/40">
              <div className="text-[10px] text-indigo-600 dark:text-indigo-400">Net Logit Score</div>
              <div className="text-base font-bold text-indigo-600 dark:text-indigo-400 mt-0.5">
                {tokenResult.token_analysis.net_logit || 0}
              </div>
            </div>
          </div>

          {/* Tokens Inspection Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-surface-border text-slate-500 dark:text-slate-400 uppercase text-[10px] tracking-wider">
                  <th className="pb-3 pr-4">Token / Word</th>
                  <th className="pb-3 pr-4">Type</th>
                  {modelChoice === 'baseline' ? (
                    <>
                      <th className="pb-3 pr-4">Term Freq (TF)</th>
                      <th className="pb-3 pr-4">IDF Weight</th>
                      <th className="pb-3 pr-4">TF-IDF Score</th>
                      <th className="pb-3 pr-4">Model Weight (w)</th>
                      <th className="pb-3 pr-4">Net Contribution</th>
                    </>
                  ) : (
                    <>
                      <th className="pb-3 pr-4">Semantic Polarity</th>
                      <th className="pb-3 pr-4">Pos Anchor Sim</th>
                      <th className="pb-3 pr-4">Neg Anchor Sim</th>
                      <th className="pb-3 pr-4">Leave-One-Out Salience</th>
                    </>
                  )}
                  <th className="pb-3">Influence</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-surface-border/50 text-foreground">
                {filteredTokens.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-6 text-center text-slate-400 font-sans">
                      No matching tokens for this filter in the current review.
                    </td>
                  </tr>
                ) : (
                  filteredTokens.map((item, idx) => (
                    <tr key={`${item.token}-${idx}`} className="hover:bg-surface/50 transition-colors">
                      {/* Token Word */}
                      <td className="py-3 pr-4 font-bold text-slate-900 dark:text-white">
                        <span
                          className={`px-2 py-0.5 rounded-md ${
                            item.is_ngram
                              ? 'bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 border border-indigo-500/30'
                              : 'bg-slate-100 dark:bg-slate-800'
                          }`}
                        >
                          &quot;{item.token}&quot;
                        </span>
                      </td>

                      {/* Type */}
                      <td className="py-3 pr-4 text-slate-500 dark:text-slate-400 text-[11px]">
                        {item.is_ngram ? 'Preserved Bi-gram' : 'Unigram'}
                      </td>

                      {/* TF-IDF Columns */}
                      {modelChoice === 'baseline' ? (
                        <>
                          <td className="py-3 pr-4 text-slate-700 dark:text-slate-300 font-semibold">{item.tf}</td>
                          <td className="py-3 pr-4 text-slate-600 dark:text-slate-400">{item.idf || '2.4500'}</td>
                          <td className="py-3 pr-4 text-primary font-bold">{item.tfidf_score || '0.1200'}</td>
                          <td className="py-3 pr-4">
                            <span
                              className={`font-semibold ${
                                (item.coefficient || 0) > 0
                                  ? 'text-emerald-600 dark:text-emerald-400'
                                  : 'text-rose-600 dark:text-rose-400'
                              }`}
                            >
                              {(item.coefficient || 0) > 0 ? `+${item.coefficient}` : item.coefficient}
                            </span>
                          </td>
                          <td className="py-3 pr-4">
                            <span
                              className={`font-bold ${
                                (item.sentiment_contribution || 0) > 0
                                  ? 'text-emerald-600 dark:text-emerald-400'
                                  : 'text-rose-600 dark:text-rose-400'
                              }`}
                            >
                              {(item.sentiment_contribution || 0) > 0
                                ? `+${item.sentiment_contribution}`
                                : item.sentiment_contribution}
                            </span>
                          </td>
                        </>
                      ) : (
                        /* MiniLM Dense Columns */
                        <>
                          <td className="py-3 pr-4 font-bold">
                            <span
                              className={
                                (item.semantic_similarity || 0) > 0
                                  ? 'text-emerald-600 dark:text-emerald-400'
                                  : 'text-rose-600 dark:text-rose-400'
                              }
                            >
                              {(item.semantic_similarity || 0) > 0
                                ? `+${item.semantic_similarity}`
                                : item.semantic_similarity}
                            </span>
                          </td>
                          <td className="py-3 pr-4 text-slate-600 dark:text-slate-400">
                            {item.sim_to_positive || '+0.120'}
                          </td>
                          <td className="py-3 pr-4 text-slate-600 dark:text-slate-400">
                            {item.sim_to_negative || '+0.080'}
                          </td>
                          <td className="py-3 pr-4">
                            <span
                              className={`font-semibold ${
                                (item.importance_score || 0) > 0
                                  ? 'text-emerald-600 dark:text-emerald-400'
                                  : (item.importance_score || 0) < 0
                                  ? 'text-rose-600 dark:text-rose-400'
                                  : 'text-slate-400'
                              }`}
                            >
                              {(item.importance_score || 0) > 0
                                ? `+${item.importance_score}`
                                : item.importance_score}
                            </span>
                          </td>
                        </>
                      )}

                      {/* Influence Badge */}
                      <td className="py-3">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-bold inline-flex items-center gap-1 ${
                            item.influence === 'Positive'
                              ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
                              : item.influence === 'Negative'
                              ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20'
                              : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20'
                          }`}
                        >
                          {item.influence === 'Positive' ? (
                            <TrendingUp className="w-3 h-3" />
                          ) : item.influence === 'Negative' ? (
                            <TrendingDown className="w-3 h-3" />
                          ) : null}
                          {item.influence}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

        </div>
      )}
    </section>
  );
};
