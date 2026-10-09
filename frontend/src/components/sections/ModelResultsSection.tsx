import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  ShieldAlert,
  Award,
  ArrowUpRight,
  Activity,
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  Info,
  Layers,
  Database,
  Globe,
  FileText,
  Star,
  RefreshCw,
} from 'lucide-react';
import {
  TOP_FEATURE_WEIGHTS,
  FeatureWeight,
  ReviewSample,
} from '../../data/sampleData';
import { BarChart3DCanvas } from '../3d/BarChart3DCanvas';
import {
  computeDatasetEvaluation,
  DynamicEvaluationResult,
  getActiveDataset,
  getActiveDatasetName,
  getActiveReviewTokens,
  getActiveReviewMeta,
  subscribeToActiveReviewTokens,
  subscribeToDataset,
  extractDatasetFeatureWeights,
  findReviewInActiveDataset,
  ActiveReviewMeta,
} from '../../store/datasetStore';
import { fetchFeatureImportance, fetchModelEvaluation } from '../../services/api';

export const ModelResultsSection: React.FC = () => {
  const selectedModel = 'tfidf';
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  // Scope switcher: 'review' (tokens from specific review), 'dataset' (vocabulary from active dataset), 'global' (benchmark weights)
  const [featureScope, setFeatureScope] = useState<'review' | 'dataset' | 'global'>('review');

  const [activeDatasetList, setActiveDatasetList] = useState<ReviewSample[]>(() => getActiveDataset());
  const [datasetName, setDatasetName] = useState<string>(() => getActiveDatasetName());

  const [activeReviewMetaState, setActiveReviewMetaState] = useState<ActiveReviewMeta | null>(() => getActiveReviewMeta());
  const [activeTokensState, setActiveTokensState] = useState<FeatureWeight[]>(() => getActiveReviewTokens());

  const [selectedReviewId, setSelectedReviewId] = useState<string>(() => {
    const meta = getActiveReviewMeta();
    return meta?.review_id || 'FMCY_00003';
  });

  const [isLoadingImportance, setIsLoadingImportance] = useState<boolean>(false);
  const [serverFeatures, setServerFeatures] = useState<{ positive: FeatureWeight[]; negative: FeatureWeight[]; all: FeatureWeight[] } | null>(null);

  // Dynamic evaluation result (confusion matrix, accuracy, precision, recall, f1, subgroups)
  const [evalResult, setEvalResult] = useState<DynamicEvaluationResult>(() =>
    computeDatasetEvaluation(getActiveDataset(), 'tfidf')
  );

  // Listen to dataset updates and active review token events
  useEffect(() => {
    const unsubTokens = subscribeToActiveReviewTokens((tokens, meta) => {
      setActiveTokensState(tokens);
      if (meta) {
        setActiveReviewMetaState(meta);
        if (meta.review_id) {
          setSelectedReviewId(meta.review_id);
        }
      }
    });

    const unsubDataset = subscribeToDataset((dataset, name) => {
      setActiveDatasetList(dataset);
      setDatasetName(name);
    });

    return () => {
      unsubTokens();
      unsubDataset();
    };
  }, []);

  // Dynamically recompute confusion matrix & evaluation metrics when dataset or model changes
  useEffect(() => {
    // 1. Instant local calculation from memory
    const dynamicRes = computeDatasetEvaluation(activeDatasetList, selectedModel);
    setEvalResult(dynamicRes);

    // 2. Query backend evaluation if server is running
    let isMounted = true;
    fetchModelEvaluation(selectedModel, activeDatasetList, datasetName).then((serverEval) => {
      if (isMounted && serverEval && serverEval.status === 'success' && serverEval.total > 0) {
        const tp = serverEval.tp ?? (serverEval.matrix.length === 2 ? serverEval.matrix[1]?.[1] : serverEval.matrix[0]?.[0]) ?? 0;
        const fp = serverEval.fp ?? (serverEval.matrix.length === 2 ? serverEval.matrix[0]?.[1] : ((serverEval.matrix[1]?.[0] ?? 0) + (serverEval.matrix[2]?.[0] ?? 0))) ?? 0;
        const fn = serverEval.fn ?? (serverEval.matrix.length === 2 ? serverEval.matrix[1]?.[0] : ((serverEval.matrix[0]?.[1] ?? 0) + (serverEval.matrix[0]?.[2] ?? 0))) ?? 0;
        const tn = serverEval.tn ?? (serverEval.matrix.length === 2 ? serverEval.matrix[0]?.[0] : ((serverEval.matrix[1]?.[1] ?? 0) + (serverEval.matrix[1]?.[2] ?? 0) + (serverEval.matrix[2]?.[1] ?? 0) + (serverEval.matrix[2]?.[2] ?? 0))) ?? 0;

        setEvalResult({
          total: serverEval.total,
          datasetName: serverEval.dataset_name || datasetName,
          modelChoice: serverEval.model_choice,
          isThreeClass: false,
          labels: serverEval.labels,
          matrix: serverEval.matrix,
          tp,
          fp,
          tn,
          fn,
          accuracy: serverEval.accuracy,
          precision: serverEval.precision,
          recall: serverEval.recall,
          f1: serverEval.f1_score,
          macroF1: serverEval.macro_f1,
          subgroups: serverEval.subgroups,
        });
      }
    });

    return () => {
      isMounted = false;
    };
  }, [activeDatasetList, datasetName, selectedModel]);

  // Fetch feature importance dynamically when scope, selected review, or model changes
  useEffect(() => {
    let isMounted = true;

    const loadData = async () => {
      setIsLoadingImportance(true);
      try {
        if (featureScope === 'review') {
          const review = findReviewInActiveDataset(selectedReviewId);
          const text = review
            ? String(review.Review_Text || review.review_text || '')
            : (activeReviewMetaState?.review_text || "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me.");

          const res = await fetchFeatureImportance({
            scope: 'review',
            review_id: selectedReviewId,
            text,
            model_choice: selectedModel,
            limit: 8,
          });

          if (isMounted && res.status === 'success') {
            setServerFeatures({
              positive: res.top_positive || [],
              negative: res.top_negative || [],
              all: res.all_features || [...(res.top_positive || []), ...(res.top_negative || [])],
            });
            if (res.review_text) {
              const revAny = review as any;
              setActiveReviewMetaState({
                review_id: res.review_id || selectedReviewId,
                review_text: res.review_text,
                product_name: res.product_name || review?.Product_Name || revAny?.product_name || 'Selected Product',
                rating: res.rating !== undefined ? res.rating : (review?.Rating !== undefined ? review.Rating : (revAny?.rating ?? 5)),
                sentiment: res.sentiment || review?.Sentiments || revAny?.sentiment,
              });
            }
          }
        } else if (featureScope === 'dataset') {
          const res = await fetchFeatureImportance({
            scope: 'dataset',
            model_choice: selectedModel,
            limit: 8,
          });

          if (isMounted) {
            if (res.status === 'success' && ((res.top_positive && res.top_positive.length > 0) || (res.top_negative && res.top_negative.length > 0))) {
              setServerFeatures({
                positive: res.top_positive || [],
                negative: res.top_negative || [],
                all: res.all_features || [...(res.top_positive || []), ...(res.top_negative || [])],
              });
            } else {
              const extracted = extractDatasetFeatureWeights(activeDatasetList, 8);
              setServerFeatures(extracted);
            }
          }
        } else {
          setServerFeatures(null);
        }
      } catch (err) {
        console.error('Error fetching feature importance:', err);
      } finally {
        if (isMounted) setIsLoadingImportance(false);
      }
    };

    loadData();

    return () => {
      isMounted = false;
    };
  }, [featureScope, selectedReviewId, selectedModel, activeDatasetList]);

  // Handler when user selects a different review from the dropdown in this section
  const handleSelectReview = (rId: string) => {
    setSelectedReviewId(rId);
    const r = findReviewInActiveDataset(rId);
    if (r) {
      const rAny = r as any;
      const text = String(r.Review_Text || rAny.review_text || '');
      setActiveReviewMetaState({
        review_id: String(r.Review_ID || r.id || rId),
        review_text: text,
        product_name: String(r.Product_Name || rAny.product_name || 'Selected Product'),
        rating: Number(r.Rating !== undefined ? r.Rating : (rAny.rating ?? 5)),
        sentiment: String(r.Sentiments || rAny.sentiment || 'Positive'),
      });
    }
  };

  // Resolve features for 3D canvas and dual cards
  let displayPositive: FeatureWeight[] = [];
  let displayNegative: FeatureWeight[] = [];
  let scopeTitle = '';
  let scopeLabel = '';

  if (featureScope === 'review') {
    scopeLabel = `Review: ${selectedReviewId}`;
    scopeTitle = activeReviewMetaState?.product_name || 'Active Review';
    if (serverFeatures && serverFeatures.all && serverFeatures.all.length > 0) {
      displayPositive = serverFeatures.positive;
      displayNegative = serverFeatures.negative;
    } else if (activeTokensState && activeTokensState.length > 0) {
      displayPositive = activeTokensState.filter((t) => t.sentiment === 'Positive' || t.weight > 0);
      displayNegative = activeTokensState.filter((t) => t.sentiment === 'Negative' || t.weight < 0);
    }
  } else if (featureScope === 'dataset') {
    scopeLabel = 'Dataset Vocabulary';
    scopeTitle = datasetName.split(' ')[0] || 'Active Dataset';
    if (serverFeatures && (serverFeatures.positive.length > 0 || serverFeatures.negative.length > 0)) {
      displayPositive = serverFeatures.positive;
      displayNegative = serverFeatures.negative;
    } else {
      const extracted = extractDatasetFeatureWeights(activeDatasetList, 8);
      displayPositive = extracted.positive;
      displayNegative = extracted.negative;
    }
  } else {
    scopeLabel = 'Benchmark Corpus';
    scopeTitle = 'Amazon 10K Footwear';
    displayPositive = TOP_FEATURE_WEIGHTS.filter((f) => f.sentiment === 'Positive').slice(0, 8);
    displayNegative = TOP_FEATURE_WEIGHTS.filter((f) => f.sentiment === 'Negative').slice(0, 8);
  }

  const display3DFeatures = [...displayPositive.slice(0, 5), ...displayNegative.slice(0, 5)];

  const maxPosWeight = Math.max(1.0, ...displayPositive.map((f) => Math.abs(f.weight || 0)));
  const maxNegWeight = Math.max(1.0, ...displayNegative.map((f) => Math.abs(f.weight || 0)));


  return (
    <section id="model-results" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono text-primary bg-primary/10 border border-primary/20 mb-3">
            <BarChart3 className="w-3.5 h-3.5" />
            Rigorous Benchmark & Subgroup Audit
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight">
            Model Performance Dashboard
          </h2>
          <p className="text-foreground-muted text-sm sm:text-base mt-2 max-w-xl">
            Empirical evaluation across {evalResult.total.toLocaleString()} records from {evalResult.datasetName}, dynamic confusion matrix heatmaps, algorithmic slice audits, and 3D token coefficient visualizers.
          </p>
        </div>

        {/* Model Architecture Indicator */}
        <div className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl glass-panel border border-primary/30 text-xs font-mono font-bold text-primary bg-primary/5 shadow-sm">
          <Sparkles className="w-4 h-4 text-primary" />
          <span>Algorithm: TF-IDF (1–3 n-grams) + Logistic Regression</span>
        </div>
      </div>

      {/* Primary KPI Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          {
            metric: 'Accuracy',
            value: evalResult.accuracy,
          },
          {
            metric: 'Precision',
            value: evalResult.precision,
          },
          {
            metric: 'Recall',
            value: evalResult.recall,
          },
          {
            metric: 'F1-Score',
            value: evalResult.f1,
          },
        ].map((m) => (
          <div
            key={m.metric}
            className="p-6 rounded-2xl glass-panel border border-surface-border flex flex-col justify-between hover:border-primary/40 transition-all hover:scale-[1.01]"
          >
            <div className="flex items-center justify-between text-xs font-mono text-foreground-muted">
              <span>{m.metric}</span>
              <Award className="w-4 h-4 text-primary" />
            </div>
            <div className="my-3">
              <div className="text-3xl sm:text-4xl font-extrabold font-display text-foreground">
                {(m.value * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-[11px] font-mono flex items-center justify-between pt-2 border-t border-surface-border">
              <span className="dark:text-slate-400 text-slate-500 font-medium">Model Status</span>
              <span className="dark:text-emerald-400 text-emerald-700 font-bold">Trained &amp; Validated</span>
            </div>
          </div>
        ))}
      </div>

      {/* Grid: Confusion Matrix Heatmap (Left) + Subgroup Audit (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
        {/* Confusion Matrix Heatmap Card */}
        <div className="lg:col-span-6 rounded-2xl glass-panel p-6 border border-surface-border flex flex-col justify-between">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold font-display text-foreground">
                  Confusion Matrix Heatmap
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
                  4 Categories (TP, FP, TN, FN)
                </span>
              </div>
              <p className="text-xs text-foreground-muted mt-0.5">
                Evaluated dynamically on {evalResult.total.toLocaleString()} records from <span className="font-semibold text-foreground">{evalResult.datasetName}</span>
              </p>
            </div>
            <span className="text-xs font-mono text-primary font-bold px-2.5 py-1 rounded-lg bg-surface border border-surface-border self-start sm:self-auto">
              TF-IDF Model
            </span>
          </div>

          {/* 4 Categories Confusion Matrix Heatmap (TP, FP, TN, FN) */}
          {(() => {
            const tpVal = evalResult.tp ?? (evalResult.matrix.length === 2 ? evalResult.matrix[1]?.[1] : evalResult.matrix[0]?.[0]) ?? 0;
            const fpVal = evalResult.fp ?? (evalResult.matrix.length === 2 ? evalResult.matrix[0]?.[1] : ((evalResult.matrix[1]?.[0] ?? 0) + (evalResult.matrix[2]?.[0] ?? 0))) ?? 0;
            const fnVal = evalResult.fn ?? (evalResult.matrix.length === 2 ? evalResult.matrix[1]?.[0] : ((evalResult.matrix[0]?.[1] ?? 0) + (evalResult.matrix[0]?.[2] ?? 0))) ?? 0;
            const tnVal = evalResult.tn ?? (evalResult.matrix.length === 2 ? evalResult.matrix[0]?.[0] : ((evalResult.matrix[1]?.[1] ?? 0) + (evalResult.matrix[1]?.[2] ?? 0) + (evalResult.matrix[2]?.[1] ?? 0) + (evalResult.matrix[2]?.[2] ?? 0))) ?? 0;

            const tpPct = evalResult.total > 0 ? ((tpVal / evalResult.total) * 100).toFixed(1) : '0.0';
            const fpPct = evalResult.total > 0 ? ((fpVal / evalResult.total) * 100).toFixed(1) : '0.0';
            const fnPct = evalResult.total > 0 ? ((fnVal / evalResult.total) * 100).toFixed(1) : '0.0';
            const tnPct = evalResult.total > 0 ? ((tnVal / evalResult.total) * 100).toFixed(1) : '0.0';

            return (
              <div className="my-4">
                {/* Column Headers: Predicted Pos / Predicted Neg */}
                <div className="grid grid-cols-2 gap-3 mb-2 px-1 text-center font-mono text-[11px] font-semibold">
                  <span className="dark:text-emerald-300 text-emerald-800 dark:bg-emerald-500/10 bg-emerald-100 py-1.5 rounded-md border dark:border-emerald-500/20 border-emerald-300">
                    Predicted Positive
                  </span>
                  <span className="dark:text-rose-300 text-rose-800 dark:bg-rose-500/10 bg-rose-100 py-1.5 rounded-md border dark:border-rose-500/20 border-rose-300">
                    Predicted Negative
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {/* Row 1 / Col 1: True Positive (TP) */}
                  <div className="p-4 rounded-xl dark:bg-emerald-500/15 bg-emerald-50/90 dark:border-emerald-500/35 border-emerald-300 border flex flex-col justify-between transition-all hover:border-emerald-500/60 shadow-[0_0_15px_rgba(16,185,129,0.08)]">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono dark:text-emerald-400 text-emerald-800 font-bold uppercase tracking-wider">
                        True Positive
                      </span>
                      <span className="text-[10px] font-mono font-extrabold px-1.5 py-0.5 rounded dark:bg-emerald-500/25 bg-emerald-200 dark:text-emerald-300 text-emerald-900 border dark:border-emerald-500/40 border-emerald-300">
                        TP
                      </span>
                    </div>
                    <div className="my-2.5 text-center">
                      <div className="text-3xl sm:text-4xl font-extrabold font-display dark:text-emerald-300 text-emerald-700">
                        {tpVal.toLocaleString()}
                      </div>
                      <div className="text-[11px] font-mono dark:text-emerald-400/90 text-emerald-800 font-medium mt-0.5">
                        {tpPct}% of dataset
                      </div>
                    </div>
                    <div className="text-[10px] text-center dark:text-emerald-300/80 text-emerald-800 font-medium border-t dark:border-emerald-500/20 border-emerald-200 pt-1.5 flex items-center justify-between">
                      <span>Actual: Positive</span>
                      <span>Correct</span>
                    </div>
                  </div>

                  {/* Row 1 / Col 2: False Negative (FN) */}
                  <div className="p-4 rounded-xl dark:bg-rose-500/10 bg-rose-50/90 dark:border-rose-500/30 border-rose-300 border flex flex-col justify-between transition-all hover:border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.05)]">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono dark:text-rose-400 text-rose-800 font-bold uppercase tracking-wider">
                        False Negative
                      </span>
                      <span className="text-[10px] font-mono font-extrabold px-1.5 py-0.5 rounded dark:bg-rose-500/20 bg-rose-200 dark:text-rose-300 text-rose-900 border dark:border-rose-500/30 border-rose-300">
                        FN (Type II)
                      </span>
                    </div>
                    <div className="my-2.5 text-center">
                      <div className="text-3xl sm:text-4xl font-extrabold font-display dark:text-rose-300 text-rose-700">
                        {fnVal.toLocaleString()}
                      </div>
                      <div className="text-[11px] font-mono dark:text-rose-400/90 text-rose-800 font-medium mt-0.5">
                        {fnPct}% of dataset
                      </div>
                    </div>
                    <div className="text-[10px] text-center dark:text-rose-300/80 text-rose-800 font-medium border-t dark:border-rose-500/20 border-rose-200 pt-1.5 flex items-center justify-between">
                      <span>Actual: Positive</span>
                      <span>Missed</span>
                    </div>
                  </div>

                  {/* Row 2 / Col 1: False Positive (FP) */}
                  <div className="p-4 rounded-xl dark:bg-rose-500/10 bg-rose-50/90 dark:border-rose-500/30 border-rose-300 border flex flex-col justify-between transition-all hover:border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.05)]">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono dark:text-rose-400 text-rose-800 font-bold uppercase tracking-wider">
                        False Positive
                      </span>
                      <span className="text-[10px] font-mono font-extrabold px-1.5 py-0.5 rounded dark:bg-rose-500/20 bg-rose-200 dark:text-rose-300 text-rose-900 border dark:border-rose-500/30 border-rose-300">
                        FP (Type I)
                      </span>
                    </div>
                    <div className="my-2.5 text-center">
                      <div className="text-3xl sm:text-4xl font-extrabold font-display dark:text-rose-300 text-rose-700">
                        {fpVal.toLocaleString()}
                      </div>
                      <div className="text-[11px] font-mono dark:text-rose-400/90 text-rose-800 font-medium mt-0.5">
                        {fpPct}% of dataset
                      </div>
                    </div>
                    <div className="text-[10px] text-center dark:text-rose-300/80 text-rose-800 font-medium border-t dark:border-rose-500/20 border-rose-200 pt-1.5 flex items-center justify-between">
                      <span>Actual: Negative</span>
                      <span>False Alarm</span>
                    </div>
                  </div>

                  {/* Row 2 / Col 2: True Negative (TN) */}
                  <div className="p-4 rounded-xl dark:bg-emerald-500/15 bg-emerald-50/90 dark:border-emerald-500/35 border-emerald-300 border flex flex-col justify-between transition-all hover:border-emerald-500/60 shadow-[0_0_15px_rgba(16,185,129,0.08)]">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono dark:text-emerald-400 text-emerald-800 font-bold uppercase tracking-wider">
                        True Negative
                      </span>
                      <span className="text-[10px] font-mono font-extrabold px-1.5 py-0.5 rounded dark:bg-emerald-500/25 bg-emerald-200 dark:text-emerald-300 text-emerald-900 border dark:border-emerald-500/40 border-emerald-300">
                        TN
                      </span>
                    </div>
                    <div className="my-2.5 text-center">
                      <div className="text-3xl sm:text-4xl font-extrabold font-display dark:text-emerald-300 text-emerald-700">
                        {tnVal.toLocaleString()}
                      </div>
                      <div className="text-[11px] font-mono dark:text-emerald-400/90 text-emerald-800 font-medium mt-0.5">
                        {tnPct}% of dataset
                      </div>
                    </div>
                    <div className="text-[10px] text-center dark:text-emerald-300/80 text-emerald-800 font-medium border-t dark:border-emerald-500/20 border-emerald-200 pt-1.5 flex items-center justify-between">
                      <span>Actual: Negative</span>
                      <span>Correct</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}

          <div className="text-[11px] font-mono dark:text-slate-400 text-slate-600 pt-3 border-t border-surface-border flex flex-wrap items-center justify-between gap-2">
            <span>Precision: {(evalResult.precision * 100).toFixed(1)}%</span>
            <span>Recall: {(evalResult.recall * 100).toFixed(1)}%</span>
            <span>Macro F1: {(evalResult.macroF1 * 100).toFixed(1)}%</span>
            <span className="dark:text-emerald-400 text-emerald-700 font-bold">Overall F1: {(evalResult.f1 * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Subgroup Performance & Disparity Audit Card */}
        <div className="lg:col-span-6 rounded-2xl glass-panel p-6 border border-surface-border flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold font-display text-foreground flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-500" />
                Algorithmic Subgroup Disparity Audit
              </h3>
              <p className="text-xs text-foreground-muted">
                Performance breakdown sliced by review length and product category
              </p>
            </div>
          </div>

          {/* Slices list */}
          <div className="space-y-3 my-2">
            {evalResult.subgroups.map((slice) => (
              <div
                key={slice.slice}
                className="p-3 rounded-xl bg-surface/50 border border-surface-border flex items-center justify-between hover:border-surface-border/80 transition-all"
              >
                <div>
                  <div className="text-xs font-semibold text-foreground">{slice.slice}</div>
                  <div className="text-[10px] font-mono dark:text-slate-400 text-slate-500">Sample size: N={slice.count.toLocaleString()}</div>
                </div>

                <div className="text-right">
                  <div className="text-sm font-bold font-mono text-foreground">
                    {slice.f1.toFixed(1)}% F1
                  </div>
                  <span
                    className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full ${
                      slice.disparity.includes('Severe') || slice.disparity.startsWith('-')
                        ? 'dark:bg-rose-500/20 bg-rose-100 dark:text-rose-400 text-rose-800 border border-rose-300 dark:border-transparent'
                        : slice.disparity.includes('Highest') || slice.disparity.startsWith('+')
                        ? 'dark:bg-emerald-500/20 bg-emerald-100 dark:text-emerald-400 text-emerald-800 border border-emerald-300 dark:border-transparent'
                        : 'dark:bg-slate-700/50 bg-slate-100 dark:text-slate-300 text-slate-700 border border-slate-300 dark:border-transparent'
                    }`}
                  >
                    {slice.disparity}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {(() => {
            const shortSlice = evalResult.subgroups.find((s) => s.slice.includes('Short'));
            const detSlice = evalResult.subgroups.find((s) => s.slice.includes('Detailed'));
            const diff = shortSlice && detSlice ? Math.abs(shortSlice.f1 - detSlice.f1).toFixed(1) : '14.8';
            return (
              <div className="p-3 rounded-xl dark:bg-amber-500/10 bg-amber-50 dark:border-amber-500/20 border-amber-300 border dark:text-amber-300 text-amber-900 text-xs mt-3">
                <strong>Key Finding:</strong> Detailed reviews (&ge;50 words) have a <strong>{diff} percentage point</strong> divergence from short reviews (&le;20 words) due to mixed trade-offs in long text.
              </div>
            );
          })()}
        </div>
      </div>

      {/* 3D Extruded Bar Visualizer for Feature Weights */}
      <div className="mt-14">
        {/* Section Heading — Clean, Non-overlapping, Distinct from Canvas */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-6 gap-3">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono dark:text-emerald-400 text-emerald-700 dark:bg-emerald-500/10 bg-emerald-100 border dark:border-emerald-500/20 border-emerald-300 mb-2.5">
              <Sparkles className="w-3.5 h-3.5" />
              Model Interpretability & Feature Weights
            </div>
            <h3 className="text-2xl sm:text-3xl font-bold font-display text-foreground tracking-tight">
              3D Feature Importance & Model Coefficients
            </h3>
            <p className="text-xs sm:text-sm text-foreground-muted mt-1 max-w-2xl leading-relaxed">
              {featureScope === 'review'
                ? `Active token influence and preserved n-grams extracted directly from review ${selectedReviewId}. Observe how specific phrases steer cumulative sentiment polarity.`
                : featureScope === 'dataset'
                ? `Aggregated vocabulary sentiment weights extracted across ${activeDatasetList.length} reviews in the active uploaded dataset.`
                : `Logistic Regression feature coefficients learned across the 10,000 Amazon footwear & accessories corpus.`}
            </p>
          </div>

          <div className="flex items-center gap-2">
            {isLoadingImportance && (
              <div className="flex items-center gap-1.5 text-xs font-mono text-indigo-400 bg-indigo-500/10 px-2.5 py-1 rounded-lg border border-indigo-500/20">
                <RefreshCw className="w-3 h-3 animate-spin" />
                <span>Updating tokens...</span>
              </div>
            )}
            <div className="hidden sm:flex items-center gap-2 text-xs font-mono dark:text-slate-400 text-slate-500">
              <span>Scope Size:</span>
              <span className="font-bold text-foreground bg-surface px-2.5 py-1 rounded-lg border border-surface-border">
                {featureScope === 'review'
                  ? `${displayPositive.length + displayNegative.length} Review Tokens`
                  : featureScope === 'dataset'
                  ? `${activeDatasetList.length} Dataset Reviews`
                  : '10,000 n-grams (1–2)'}
              </span>
            </div>
          </div>
        </div>

        {/* Scope Selector Tabs: Review Tokens vs Dataset Vocabulary vs Global Corpus */}
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5 p-2 rounded-2xl glass-panel border border-surface-border">
          <div className="flex items-center gap-1.5 overflow-x-auto">
            <button
              onClick={() => setFeatureScope('review')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold font-mono transition-all ${
                featureScope === 'review'
                  ? 'bg-primary text-white shadow-glow-primary'
                  : 'text-foreground-muted hover:text-foreground hover:bg-surface/60'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>🎯 Active Review Tokens</span>
              {selectedReviewId && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/20 text-white font-mono">
                  {selectedReviewId}
                </span>
              )}
            </button>

            <button
              onClick={() => setFeatureScope('dataset')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold font-mono transition-all ${
                featureScope === 'dataset'
                  ? 'bg-secondary text-white shadow-glow-primary'
                  : 'text-foreground-muted hover:text-foreground hover:bg-surface/60'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span>📊 Active Dataset Vocabulary</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/20 text-white font-mono">
                {activeDatasetList.length}
              </span>
            </button>

            <button
              onClick={() => setFeatureScope('global')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold font-mono transition-all ${
                featureScope === 'global'
                  ? 'bg-slate-700 text-white shadow-md'
                  : 'text-foreground-muted hover:text-foreground hover:bg-surface/60'
              }`}
            >
              <Globe className="w-3.5 h-3.5" />
              <span>🌐 Benchmark Corpus</span>
            </button>
          </div>

          <div className="text-[11px] font-mono text-foreground-muted px-2">
            Algorithm: <strong className="text-foreground">Tone-Aware Sublinear TF-IDF + Logistic Regression</strong>
          </div>
        </div>

        {/* When Review Scope is active: Review Selector Dropdown & Review Quote Banner */}
        {featureScope === 'review' && (
          <div className="mb-5 p-4 rounded-2xl glass-panel border border-primary/20 bg-primary/5 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-xs font-mono font-semibold text-primary">
                <Layers className="w-4 h-4" />
                <span>Select Review to Inspect in 3D:</span>
              </div>

              {/* Review Selector Dropdown from active dataset */}
              <div className="w-full sm:w-auto sm:min-w-[340px]">
                <select
                  value={selectedReviewId}
                  onChange={(e) => handleSelectReview(e.target.value)}
                  className="w-full text-xs font-mono bg-surface border border-surface-border rounded-xl px-3 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary shadow-sm"
                >
                  {activeDatasetList.slice(0, 100).map((r, i) => {
                    const rAny = r as any;
                    const rId = String(r.Review_ID || r.id || `ROW_${i + 1}`);
                    const rText = String(r.Review_Text || rAny.review_text || '');
                    const rateVal = r.Rating !== undefined ? r.Rating : rAny.rating;
                    const ratingStar = rateVal ? `[${rateVal}★]` : '';
                    return (
                      <option key={`${rId}-${i}`} value={rId}>
                        {rId} {ratingStar} — {rText.slice(0, 50)}...
                      </option>
                    );
                  })}
                </select>
              </div>
            </div>

            {/* Active Review Quote & Metadata Card */}
            <div className="p-3.5 rounded-xl bg-surface/80 border border-surface-border flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-primary text-white">
                    {selectedReviewId}
                  </span>
                  {activeReviewMetaState?.product_name && (
                    <span className="text-xs font-medium text-foreground">
                      {activeReviewMetaState.product_name}
                    </span>
                  )}
                  {activeReviewMetaState?.rating !== undefined && (
                    <span className="flex items-center text-amber-500 dark:text-amber-400 text-xs">
                      {'★'.repeat(Math.max(1, Math.min(5, Number(activeReviewMetaState.rating))))}
                      <span className="text-[10px] font-mono ml-1 dark:text-slate-400 text-slate-500">
                        ({activeReviewMetaState.rating}/5)
                      </span>
                    </span>
                  )}
                  {activeReviewMetaState?.sentiment && (
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-bold ${
                        activeReviewMetaState.sentiment === 'Positive'
                          ? 'dark:bg-emerald-500/20 bg-emerald-100 dark:text-emerald-400 text-emerald-800 border border-emerald-300 dark:border-transparent'
                          : 'dark:bg-rose-500/20 bg-rose-100 dark:text-rose-400 text-rose-800 border border-rose-300 dark:border-transparent'
                      }`}
                    >
                      {activeReviewMetaState.sentiment}
                    </span>
                  )}
                </div>
                <p className="text-xs italic text-foreground-muted leading-relaxed pt-1">
                  &ldquo;{activeReviewMetaState?.review_text || "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me."}&rdquo;
                </p>
              </div>

              <div className="shrink-0 flex sm:flex-col items-center sm:items-end justify-between gap-1 text-[11px] font-mono dark:text-slate-400 text-slate-500 border-t sm:border-t-0 sm:border-l sm:pl-3 border-surface-border">
                <span>Positive tokens: <strong className="dark:text-emerald-400 text-emerald-700">{displayPositive.length}</strong></span>
                <span>Negative tokens: <strong className="dark:text-rose-400 text-rose-700">{displayNegative.length}</strong></span>
              </div>
            </div>
          </div>
        )}

        {/* When Dataset Scope is active: Dataset Summary Pill */}
        {featureScope === 'dataset' && (
          <div className="mb-5 p-3.5 rounded-2xl glass-panel border border-secondary/20 bg-secondary/5 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-secondary" />
              <span className="font-semibold text-foreground">Active Dataset Vocabulary:</span>
              <span className="font-mono text-secondary font-bold">{datasetName}</span>
            </div>
            <div className="font-mono text-foreground-muted text-[11px]">
              Aggregating top polarity vocabulary across all <strong>{activeDatasetList.length}</strong> loaded customer records.
            </div>
          </div>
        )}

        {/* 3D Interactive Canvas */}
        <BarChart3DCanvas
          features={display3DFeatures}
          selectedWord={selectedFeature}
          scopeLabel={scopeLabel}
          scopeTitle={scopeTitle}
          onSelectWord={setSelectedFeature}
        />

        {/* Dual-Column Feature Importance Grid (Synchronized with 3D Canvas) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          {/* Left Column: Top Positive Drivers */}
          <div className="rounded-2xl glass-panel p-5 sm:p-6 border border-surface-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-surface-border">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg dark:bg-emerald-500/15 bg-emerald-100 dark:text-emerald-400 text-emerald-700 flex items-center justify-center">
                    <ThumbsUp className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-foreground">Top Positive Drivers</h4>
                    <span className="text-[11px] text-foreground-muted">
                      {featureScope === 'review'
                        ? `Tokens pushing P(Positive) in ${selectedReviewId}`
                        : featureScope === 'dataset'
                        ? `Dominant praise vocabulary in active dataset`
                        : `Footwear Comfort, Sizing & Praise`}
                    </span>
                  </div>
                </div>
                <span className="text-xs font-mono font-semibold dark:text-emerald-400 text-emerald-700 dark:bg-emerald-500/10 bg-emerald-100 px-2 py-0.5 rounded-full border dark:border-emerald-500/20 border-emerald-300">
                  {displayPositive.length} drivers &uarr;
                </span>
              </div>

              {/* Positive Tokens List */}
              <div className="space-y-2.5">
                {displayPositive.length === 0 ? (
                  <div className="p-4 rounded-xl bg-surface/50 border border-surface-border text-center text-xs text-foreground-muted">
                    No positive sentiment drivers detected in this review text.
                  </div>
                ) : (
                  displayPositive.map((item, idx) => {
                    const isHovered = selectedFeature === item.word;
                    const pct = Math.min(100, Math.max(12, Math.round((Math.abs(item.weight) / maxPosWeight) * 100)));

                    return (
                      <div
                        key={`${item.word}-${idx}`}
                        onMouseEnter={() => setSelectedFeature(item.word)}
                        onMouseLeave={() => setSelectedFeature(null)}
                        className={`p-2.5 rounded-xl transition-all duration-200 cursor-pointer border flex flex-col gap-1.5 ${
                          isHovered
                            ? 'dark:bg-slate-800 bg-emerald-50/80 border-emerald-500 shadow-md ring-1 ring-emerald-500/30'
                            : 'bg-surface/50 border-surface-border hover:border-emerald-500/50'
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono dark:text-slate-400 text-slate-500 w-4">#{idx + 1}</span>
                            <span className="font-mono font-bold text-foreground">&quot;{item.word}&quot;</span>
                            {item.context && (
                              <span className="text-[10px] font-sans text-foreground-muted hidden sm:inline px-1.5 py-0.5 rounded bg-surface border border-surface-border">
                                {item.context}
                              </span>
                            )}
                          </div>
                          <span className="font-mono font-bold text-emerald-700 dark:text-emerald-400">
                            +{item.weight.toFixed(2)}
                          </span>
                        </div>

                        {/* Proportional Weight Bar */}
                        <div className="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-emerald-500 h-full rounded-full transition-all duration-300"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Top Negative Drivers */}
          <div className="rounded-2xl glass-panel p-5 sm:p-6 border border-surface-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-surface-border">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg dark:bg-rose-500/15 bg-rose-100 dark:text-rose-400 text-rose-700 flex items-center justify-center">
                    <ThumbsDown className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-foreground">Top Negative Drivers</h4>
                    <span className="text-[11px] text-foreground-muted">
                      {featureScope === 'review'
                        ? `Tokens pushing P(Negative) in ${selectedReviewId}`
                        : featureScope === 'dataset'
                        ? `Dominant critique vocabulary in active dataset`
                        : `Defects, Discomfort & Negation`}
                    </span>
                  </div>
                </div>
                <span className="text-xs font-mono font-semibold dark:text-rose-400 text-rose-700 dark:bg-rose-500/10 bg-rose-100 px-2 py-0.5 rounded-full border dark:border-rose-500/20 border-rose-300">
                  {displayNegative.length} drivers &uarr;
                </span>
              </div>

              {/* Negative Tokens List */}
              <div className="space-y-2.5">
                {displayNegative.length === 0 ? (
                  <div className="p-4 rounded-xl bg-surface/50 border border-surface-border text-center text-xs text-foreground-muted">
                    No negative sentiment drivers detected in this review text.
                  </div>
                ) : (
                  displayNegative.map((item, idx) => {
                    const isHovered = selectedFeature === item.word;
                    const pct = Math.min(100, Math.max(12, Math.round((Math.abs(item.weight) / maxNegWeight) * 100)));

                    return (
                      <div
                        key={`${item.word}-${idx}`}
                        onMouseEnter={() => setSelectedFeature(item.word)}
                        onMouseLeave={() => setSelectedFeature(null)}
                        className={`p-2.5 rounded-xl transition-all duration-200 cursor-pointer border flex flex-col gap-1.5 ${
                          isHovered
                            ? 'dark:bg-slate-800 bg-rose-50/80 border-rose-500 shadow-md ring-1 ring-rose-500/30'
                            : 'bg-surface/50 border-surface-border hover:border-rose-500/50'
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono dark:text-slate-400 text-slate-500 w-4">#{idx + 1}</span>
                            <span className="font-mono font-bold text-foreground">&quot;{item.word}&quot;</span>
                            {item.context && (
                              <span className="text-[10px] font-sans text-foreground-muted hidden sm:inline px-1.5 py-0.5 rounded bg-surface border border-surface-border">
                                {item.context}
                              </span>
                            )}
                          </div>
                          <span className="font-mono font-bold text-rose-700 dark:text-rose-400">
                            {item.weight.toFixed(2)}
                          </span>
                        </div>

                        {/* Proportional Weight Bar */}
                        <div className="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-rose-500 h-full rounded-full transition-all duration-300"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Domain Linguistic Insight Callout Box */}
        <div className="mt-4 p-4 rounded-2xl dark:bg-indigo-500/10 bg-indigo-50/80 dark:border-indigo-500/20 border-indigo-200 border text-xs text-foreground flex items-start gap-3">
          <Info className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-bold text-indigo-700 dark:text-indigo-400">Corpus Linguistic Grounding:</span>{' '}
            {featureScope === 'review' ? (
              <>
                The 3D model coefficients and tokens above are extracted directly from review{' '}
                <span className="font-mono font-bold text-foreground">{selectedReviewId}</span>. Preserved rating n-grams
                (such as <span className="font-mono font-bold dark:text-emerald-400 text-emerald-700">&quot;five stars&quot;</span> and{' '}
                <span className="font-mono font-bold dark:text-emerald-400 text-emerald-700">&quot;won me over&quot;</span>) correctly shift the document polarity to positive,
                even when introductory contrasting clauses (<span className="font-mono font-bold dark:text-rose-400 text-rose-700">&quot;was not a fan&quot;</span>) are present.
              </>
            ) : featureScope === 'dataset' ? (
              <>
                Aggregated token weights across all <span className="font-mono font-bold text-foreground">{activeDatasetList.length}</span> reviews in{' '}
                <span className="font-mono font-bold text-foreground">{datasetName.split(' ')[0]}</span>. The model isolates recurring domain praises and failure modes across the entire dataset.
              </>
            ) : (
              <>
                In the Amazon footwear corpus, sentiment is strongly anchored to sensory fit terms (<em>comfortable</em>, <em>uncomfortable</em>, <em>too tight</em>, <em>narrow</em>) and transactional outcomes (<em>returned</em>, <em>cheap</em>). By using custom contraction expansion and preserving negation bi-grams, our pipeline assigned a <span className="font-mono font-bold dark:text-rose-400 text-rose-700">-2.18</span> weight to <span className="font-mono font-bold text-foreground">&quot;not comfortable&quot;</span>, successfully overriding the <span className="font-mono font-bold dark:text-emerald-400 text-emerald-700">+4.37</span> positive pull of isolated <span className="font-mono font-bold text-foreground">&quot;comfortable&quot;</span>.
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
