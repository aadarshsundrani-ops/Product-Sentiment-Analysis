import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Play, CheckCircle2, Copy, Check, Sparkles, Layers, RotateCcw, ArrowUpRight } from 'lucide-react';
import { fireSafeConfetti } from '../../utils/confetti';
import { PYTHON_CODE_TABS } from '../../data/sampleData';

const PIPELINE_LOGS = [
  '[STEP 1/5] Initializing NLP Environment (Python 3.9, scikit-learn 1.6, NumPy, Pandas)...',
  '[STEP 2/5] Loading customer reviews and applying text cleaning & negation preservation...',
  '[STEP 3/5] Vectorizing sublinear TF-IDF 1–3 n-grams with contrastive discourse weighting (vocab=35,000)...',
  '[STEP 4/5] Training L2-regularized Logistic Regression classifier (C=3.0, solver="liblinear")...',
  '[STEP 4/5] Running stratified evaluation on held-out test split (N=1,567 reviews)...',
  '[STEP 5/5] Running algorithmic subgroup audit across review lengths and product categories...',
  '[STEP 5/5] Computing token coefficient weights on critical negation & concession test instances...',
  '================================================================================',
  'TRAINING COMPLETE: TF-IDF Model Accuracy = 94.82% | Macro F1 = 94.80%',
  'DISPARITY AUDIT: Detailed reviews F1 = 89.15% | Short reviews F1 = 96.85% [Balanced]',
  'Model dumped to outputs/baseline_model.joblib [OK]',
];

export const CodeRunnerSection: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'preprocessing' | 'models' | 'evaluation'>('preprocessing');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [isCompleted, setIsCompleted] = useState(false);
  const [copied, setCopied] = useState(false);
  const terminalEndRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const currentSnippet = PYTHON_CODE_TABS[activeTab];

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(currentSnippet.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRunPipeline = () => {
    if (isRunning) return;
    if (intervalRef.current) clearInterval(intervalRef.current);

    setIsRunning(true);
    setIsCompleted(false);
    setProgress(0);
    setLogs(['$ python src/run_pipeline.py --sample-size 10000']);

    let logIndex = 0;
    intervalRef.current = setInterval(() => {
      if (logIndex < PIPELINE_LOGS.length) {
        setLogs((prev) => [...prev, PIPELINE_LOGS[logIndex]]);
        logIndex++;
        setProgress(Math.round((logIndex / PIPELINE_LOGS.length) * 100));
      } else {
        if (intervalRef.current) clearInterval(intervalRef.current);
        intervalRef.current = null;
        setIsRunning(false);
        setIsCompleted(true);
        fireSafeConfetti({
          particleCount: 35,
          origin: { x: 0.5, y: 0.65 },
        });
      }
    }, 320);
  };

  const handleResetPipeline = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    setIsRunning(false);
    setIsCompleted(false);
    setProgress(0);
    setLogs([]);
  };

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <section id="code-runner" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono text-primary bg-primary/10 border border-primary/20 mb-3">
            <Terminal className="w-3.5 h-3.5" />
            Executable Pipeline Architecture
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight">
            Run the Python ML Pipeline
          </h2>
          <p className="text-foreground-muted text-sm sm:text-base mt-2 max-w-xl">
            Inspect the actual Python implementation used to preprocess tokens, train TF-IDF and
            Sentence-Transformers models, and run subgroup audits. Click &ldquo;Run Pipeline&rdquo; to simulate live execution.
          </p>
        </div>

        {/* Action Buttons: Run & Reset */}
        <div className="flex items-center gap-3 self-start md:self-auto">
          {logs.length > 0 && (
            <button
              onClick={handleResetPipeline}
              className="px-4 py-3 rounded-xl font-medium text-xs glass-panel hover:border-primary/50 text-foreground transition-all flex items-center gap-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
              <span>Reset Logs</span>
            </button>
          )}

          <button
            onClick={handleRunPipeline}
            disabled={isRunning}
            className="px-6 py-3 rounded-xl font-bold text-sm bg-gradient-to-r from-primary to-secondary hover:opacity-95 text-white shadow-glow-primary transition-all duration-200 flex items-center gap-2 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Executing Pipeline ({progress}%)...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>Run Pipeline Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Grid: Code Viewer (Left) + Terminal / 3D Progress (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Syntax Highlighted Code Viewer */}
        <div className="lg:col-span-7 rounded-2xl glass-panel border border-surface-border overflow-hidden flex flex-col">
          {/* Tabs Bar */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-surface-border bg-surface/60">
            <div className="flex items-center gap-1 sm:gap-2">
              {(['preprocessing', 'models', 'evaluation'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                    activeTab === tab
                      ? 'bg-primary/20 text-primary border border-primary/40 font-bold'
                      : 'text-foreground-muted hover:text-foreground hover:bg-surface'
                  }`}
                >
                  {tab === 'preprocessing'
                    ? 'preprocessing.py'
                    : tab === 'models'
                    ? 'models.py'
                    : 'audit.py'}
                </button>
              ))}
            </div>

            <button
              onClick={handleCopyCode}
              className="p-1.5 rounded-lg hover:bg-surface text-foreground-muted hover:text-foreground transition-colors"
              title="Copy code"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>

          {/* Tab description */}
          <div className="px-5 py-2.5 bg-surface/30 border-b border-surface-border text-xs text-foreground-muted flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-primary shrink-0" />
            <span>{currentSnippet.description}</span>
          </div>

          {/* Code Viewer Panel */}
          <div className="p-5 font-mono text-xs overflow-x-auto flex-1 bg-black/40 text-slate-200 leading-relaxed max-h-[460px]">
            <pre>
              <code>{currentSnippet.code}</code>
            </pre>
          </div>
        </div>

        {/* Right Column: Execution Terminal & 3D Progress Visualization */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Circular / Ring Progress Card */}
          <div className="rounded-2xl glass-panel p-6 border border-surface-border flex items-center justify-between relative overflow-hidden">
            <div>
              <div className="text-xs font-mono uppercase text-foreground-muted tracking-wider mb-1">
                Pipeline Execution Status
              </div>
              <div className="text-xl font-bold font-display text-foreground flex items-center gap-2">
                {isCompleted ? (
                  <span className="text-emerald-400 flex items-center gap-1.5">
                    <CheckCircle2 className="w-5 h-5" /> Completed
                  </span>
                ) : isRunning ? (
                  <span className="text-primary flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-primary animate-ping" />
                    Running Steps...
                  </span>
                ) : (
                  <span className="text-foreground-muted">Ready to Run</span>
                )}
              </div>
              <div className="text-xs text-foreground-muted mt-1">
                {isRunning
                  ? 'Simulating cross-validation & LIME explainer...'
                  : isCompleted
                  ? 'Outputs dumped to /outputs directory'
                  : 'Click "Run Pipeline" to stream live logs'}
              </div>
            </div>

            {/* Circular Progress Gauge */}
            <div className="relative w-20 h-20 flex items-center justify-center shrink-0">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="currentColor"
                  strokeWidth="8"
                  className="text-surface-border fill-transparent"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="currentColor"
                  strokeWidth="8"
                  strokeDasharray={2 * Math.PI * 40}
                  strokeDashoffset={2 * Math.PI * 40 * (1 - progress / 100)}
                  strokeLinecap="round"
                  className={`transition-all duration-300 fill-transparent ${
                    isCompleted ? 'text-emerald-400' : 'text-primary'
                  }`}
                />
              </svg>
              <div className="absolute text-sm font-mono font-bold text-foreground">
                {progress}%
              </div>
            </div>
          </div>

          {/* Terminal Log Streamer */}
          <div className="rounded-2xl glass-panel border border-surface-border overflow-hidden flex flex-col flex-1 min-h-[300px]">
            <div className="px-4 py-2.5 bg-black/60 border-b border-surface-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
                <span className="text-[11px] font-mono text-slate-400 ml-2">bash — python pipeline</span>
              </div>
              <span className="text-[10px] font-mono text-slate-500">simulated / live runner</span>
            </div>

            <div className="p-4 bg-black/80 font-mono text-[11px] leading-relaxed text-slate-300 flex-1 overflow-y-auto max-h-[320px] space-y-1.5">
              {logs.length === 0 ? (
                <div className="text-slate-500 italic py-8 text-center">
                  Pipeline terminal is idle. Click &ldquo;Run Pipeline Simulation&rdquo; above to stream execution steps.
                </div>
              ) : (
                logs.map((line, idx) => (
                  <div
                    key={idx}
                    className={`${
                      line.includes('COMPLETE')
                        ? 'text-emerald-400 font-bold'
                        : line.includes('DISPARITY')
                        ? 'text-amber-400 font-semibold'
                        : line.startsWith('$')
                        ? 'text-primary font-bold'
                        : 'text-slate-300'
                    }`}
                  >
                    {line}
                  </div>
                ))
              )}
              <div ref={terminalEndRef} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
