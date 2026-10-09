import React, { useState } from 'react';
import { Cpu, RefreshCw, Zap } from 'lucide-react';
import { Architecture3DCanvas, PipelineMode } from '../3d/Architecture3DCanvas';

export const ArchitectureSection: React.FC = () => {
  const [mode, setMode] = useState<PipelineMode>('inference');
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  return (
    <section id="architecture" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header — Crystal Clear Contrast & Responsive Spacing */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-5">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-semibold text-indigo-600 dark:text-cyan-400 bg-indigo-50 dark:bg-cyan-500/10 border border-indigo-200 dark:border-cyan-500/20 mb-3 shadow-sm">
            <Cpu className="w-3.5 h-3.5" />
            System Architecture & Data Flow
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight text-slate-900 dark:text-white">
            High-Performance ML Infrastructure
          </h2>
          <p className="text-slate-600 dark:text-slate-300 text-sm sm:text-base mt-2 max-w-2xl leading-relaxed font-normal">
            Toggle between the live REST Inference Pipeline and the offline Training & Audit Pipeline
            to view data packet flow through each architectural layer in 3D.
          </p>
        </div>

        {/* Pipeline Mode Toggle — Accessible & Non-wrapping */}
        <div className="w-full sm:w-auto flex p-1.5 rounded-2xl bg-slate-100 dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 shadow-sm shrink-0">
          <button
            onClick={() => {
              setMode('inference');
              setSelectedStep(null);
            }}
            className={`flex-1 sm:flex-initial px-4 py-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 ${
              mode === 'inference'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            Inference / Serving Pipeline
          </button>
          <button
            onClick={() => {
              setMode('training');
              setSelectedStep(null);
            }}
            className={`flex-1 sm:flex-initial px-4 py-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 ${
              mode === 'training'
                ? 'bg-cyan-600 dark:bg-cyan-500 text-white shadow-md'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Training & Audit Pipeline
          </button>
        </div>
      </div>

      {/* 3D Animated Architecture Diagram Canvas */}
      <div className="mb-6">
        <Architecture3DCanvas
          mode={mode}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
        />
      </div>

      {/* Layer Description Cards — Fully Synchronized & High Contrast */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {mode === 'inference' ? (
          <>
            <div
              onMouseEnter={() => setSelectedStep(0)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(0)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 0
                  ? 'ring-2 ring-sky-500 bg-sky-50/90 dark:bg-slate-800/90 border-sky-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-sky-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-sky-100 dark:bg-sky-500/20 text-sky-800 dark:text-sky-300 flex items-center justify-center font-mono font-bold text-xs border border-sky-300 dark:border-sky-500/30">
                  01
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-sky-100 dark:bg-sky-500/20 text-sky-800 dark:text-sky-300 border border-sky-300 dark:border-sky-500/30">
                  React 18 + R3F
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">Client Presentation</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                Vite + React 18 frontend with interactive 3D WebGL centerpiece and real-time review input.
              </p>
            </div>

            <div
              onMouseEnter={() => setSelectedStep(1)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(1)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 1
                  ? 'ring-2 ring-indigo-500 bg-indigo-50/90 dark:bg-slate-800/90 border-indigo-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-indigo-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-500/20 text-indigo-800 dark:text-indigo-300 flex items-center justify-center font-mono font-bold text-xs border border-indigo-300 dark:border-indigo-500/30">
                  02
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-indigo-100 dark:bg-indigo-500/20 text-indigo-800 dark:text-indigo-300 border border-indigo-300 dark:border-indigo-500/30">
                  Uvicorn REST
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">FastAPI Gateway</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                Asynchronous ASGI server handling CORS, Pydantic schemas, and sub-25ms response latency.
              </p>
            </div>

            <div
              onMouseEnter={() => setSelectedStep(2)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(2)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 2
                  ? 'ring-2 ring-purple-500 bg-purple-50/90 dark:bg-slate-800/90 border-purple-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-purple-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-500/20 text-purple-800 dark:text-purple-300 flex items-center justify-center font-mono font-bold text-xs border border-purple-300 dark:border-purple-500/30">
                  03
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-purple-100 dark:bg-purple-500/20 text-purple-800 dark:text-purple-300 border border-purple-300 dark:border-purple-500/30">
                  TF-IDF / MiniLM
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">Vectorize & Predict</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                Sublinear TF-IDF or MiniLM transformer projection into trained L2 Logistic Regression.
              </p>
            </div>

            <div
              onMouseEnter={() => setSelectedStep(3)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(3)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 3
                  ? 'ring-2 ring-emerald-500 bg-emerald-50/90 dark:bg-slate-800/90 border-emerald-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-emerald-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 flex items-center justify-center font-mono font-bold text-xs border border-emerald-300 dark:border-emerald-500/30">
                  04
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30">
                  LIME / Attr
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">Attributions & Stream</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                Word-level LIME attributions, class confidence percentiles, and structured JSON output.
              </p>
            </div>
          </>
        ) : (
          <>
            <div
              onMouseEnter={() => setSelectedStep(0)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(0)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 0
                  ? 'ring-2 ring-sky-500 bg-sky-50/90 dark:bg-slate-800/90 border-sky-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-sky-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-sky-100 dark:bg-sky-500/20 text-sky-800 dark:text-sky-300 flex items-center justify-center font-mono font-bold text-xs border border-sky-300 dark:border-sky-500/30">
                  01
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-sky-100 dark:bg-sky-500/20 text-sky-800 dark:text-sky-300 border border-sky-300 dark:border-sky-500/30">
                  10K Corpus
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">Corpus Ingestion</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                10,000 real Amazon product reviews parsed, rating 1-5 binarized, and 80/20 train/test split.
              </p>
            </div>

            <div
              onMouseEnter={() => setSelectedStep(1)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(1)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 1
                  ? 'ring-2 ring-indigo-500 bg-indigo-50/90 dark:bg-slate-800/90 border-indigo-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-indigo-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-500/20 text-indigo-800 dark:text-indigo-300 flex items-center justify-center font-mono font-bold text-xs border border-indigo-300 dark:border-indigo-500/30">
                  02
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-indigo-100 dark:bg-indigo-500/20 text-indigo-800 dark:text-indigo-300 border border-indigo-300 dark:border-indigo-500/30">
                  Tokenization
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">Negation Preservation</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                Contraction mapping (&apos;didn&apos;t&apos; &rarr; &apos;did not&apos;) and preservation of negation bi-grams.
              </p>
            </div>

            <div
              onMouseEnter={() => setSelectedStep(2)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(2)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 2
                  ? 'ring-2 ring-purple-500 bg-purple-50/90 dark:bg-slate-800/90 border-purple-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-purple-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-500/20 text-purple-800 dark:text-purple-300 flex items-center justify-center font-mono font-bold text-xs border border-purple-300 dark:border-purple-500/30">
                  03
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-purple-100 dark:bg-purple-500/20 text-purple-800 dark:text-purple-300 border border-purple-300 dark:border-purple-500/30">
                  Dual Bench
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">Model Benchmarking</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                Dual pipeline training: Tone-Aware TF-IDF baseline (94.82% acc) vs MiniLM embeddings (84.49% acc).
              </p>
            </div>

            <div
              onMouseEnter={() => setSelectedStep(3)}
              onMouseLeave={() => setSelectedStep(null)}
              onClick={() => setSelectedStep(3)}
              className={`p-5 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                selectedStep === 3
                  ? 'ring-2 ring-amber-500 bg-amber-50/90 dark:bg-slate-800/90 border-amber-400 shadow-md scale-[1.02]'
                  : 'bg-white/90 dark:bg-slate-900/80 border-slate-200 dark:border-slate-800 hover:border-amber-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-lg bg-amber-100 dark:bg-amber-500/20 text-amber-900 dark:text-amber-300 flex items-center justify-center font-mono font-bold text-xs border border-amber-300 dark:border-amber-500/30">
                  04
                </div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-500/20 text-amber-900 dark:text-amber-300 border border-amber-300 dark:border-amber-500/30">
                  Disparity Audit
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-white text-sm font-display">Subgroup Audit & LIME</h4>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
                Algorithmic disparity audit across review length buckets and local token explainability cards.
              </p>
            </div>
          </>
        )}
      </div>
    </section>
  );
};

