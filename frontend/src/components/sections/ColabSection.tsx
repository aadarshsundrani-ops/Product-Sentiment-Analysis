import React, { useState } from 'react';
import { ExternalLink, Code2, ChevronLeft, ChevronRight, Download, BookOpen, X, ArrowLeft, Check, Sparkles } from 'lucide-react';
import { COLAB_NOTEBOOK_INFO } from '../../data/sampleData';

export const ColabSection: React.FC = () => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const cells = COLAB_NOTEBOOK_INFO.cells;

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % cells.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + cells.length) % cells.length);
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = COLAB_NOTEBOOK_INFO.downloadUrl || '/sentiment_tfidf_pipeline.ipynb';
    link.download = COLAB_NOTEBOOK_INFO.title || 'sentiment_tfidf_pipeline.ipynb';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setDownloaded(true);
    setTimeout(() => setDownloaded(false), 2500);
  };

  return (
    <section id="colab" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="rounded-3xl glass-panel-elevated p-8 sm:p-12 border border-surface-border relative overflow-hidden">
        {/* Glow ambient background accent */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-amber-500/10 via-primary/10 to-transparent rounded-full blur-3xl pointer-events-none" />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative z-10">
          {/* Left Column: Colab Badge & Description */}
          <div className="lg:col-span-6 space-y-6">
            {/* Colab Official-style Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-mono font-semibold">
              <span className="text-base">🟡</span>
              <span>Official Google Colaboratory Integration</span>
            </div>

            <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight text-foreground">
              Reproduce & Experiment in Google Colab
            </h2>

            <p className="text-sm sm:text-base text-foreground-muted leading-relaxed">
              {COLAB_NOTEBOOK_INFO.description}
            </p>

            <div className="space-y-2 text-xs text-foreground-muted font-mono">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Runs on free Google Colab Cloud GPU/CPU with 1-click execution</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Includes interactive LIME HTML explanation widgets inline</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Reproducible random seed (42) and automatic Amazon 10K dataset download</span>
              </div>
            </div>

            {/* Action Buttons: Open in Colab + Download .ipynb + Interactive Viewer */}
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <a
                href={COLAB_NOTEBOOK_INFO.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl font-bold text-sm bg-gradient-to-r from-amber-500 via-orange-500 to-amber-600 text-white shadow-lg hover:shadow-amber-500/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <img
                  src="https://colab.research.google.com/img/colab_favicon_256px.png"
                  alt="Colab Logo"
                  className="w-5 h-5"
                />
                <span>Launch Google Colab</span>
                <ExternalLink className="w-4 h-4 ml-1" />
              </a>

              <button
                onClick={handleDownload}
                className="inline-flex items-center gap-2 px-4 py-3 rounded-xl font-semibold text-xs glass-panel hover:border-primary/50 text-foreground transition-all hover:scale-[1.02]"
              >
                {downloaded ? <Check className="w-4 h-4 text-emerald-400" /> : <Download className="w-4 h-4 text-primary" />}
                <span>{downloaded ? 'Downloaded .ipynb!' : 'Download .ipynb Notebook'}</span>
              </button>

              <button
                onClick={() => setIsModalOpen(true)}
                className="inline-flex items-center gap-2 px-4 py-3 rounded-xl font-semibold text-xs glass-panel hover:border-secondary/50 text-secondary transition-all hover:scale-[1.02]"
              >
                <BookOpen className="w-4 h-4" />
                <span>Preview Full Notebook</span>
              </button>
            </div>
          </div>

          {/* Right Column: Interactive Notebook Cell Output Carousel */}
          <div className="lg:col-span-6 rounded-2xl glass-panel border border-surface-border overflow-hidden flex flex-col shadow-xl">
            {/* Carousel Header */}
            <div className="px-4 py-3 dark:bg-black/60 bg-slate-900 border-b dark:border-surface-border border-slate-800 flex items-center justify-between text-white">
              <div className="flex items-center gap-2">
                <Code2 className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-mono font-bold text-white">
                  {cells[currentSlide].title}
                </span>
              </div>

              <div className="flex items-center gap-1.5">
                <button
                  onClick={prevSlide}
                  className="p-1 rounded-md hover:bg-white/10 text-slate-300 hover:text-white transition-colors"
                  aria-label="Previous Cell"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="text-[10px] font-mono text-slate-300 px-1">
                  {currentSlide + 1} / {cells.length}
                </span>
                <button
                  onClick={nextSlide}
                  className="p-1 rounded-md hover:bg-white/10 text-slate-300 hover:text-white transition-colors"
                  aria-label="Next Cell"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Code Cell Content */}
            <div className="p-5 font-mono text-xs bg-black/80 text-slate-200 leading-relaxed overflow-x-auto min-h-[200px]">
              <pre>
                <code>{cells[currentSlide].content}</code>
              </pre>
            </div>

            {/* Cell Footer */}
            <div className="px-4 py-2.5 bg-surface/50 border-t border-surface-border flex items-center justify-between text-[11px] text-foreground-muted font-mono">
              <span>Status: Cell executed in 2.1s</span>
              <span className="dark:text-emerald-400 text-emerald-700 font-bold">Memory: 412 MB</span>
            </div>
          </div>
        </div>
      </div>

      {/* Full-Screen Interactive Notebook Viewer Modal with Clear 'Move Back' Navigation */}
      {isModalOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-xl flex items-center justify-center p-4 sm:p-6"
          onClick={() => setIsModalOpen(false)}
        >
          <div
            className="w-full max-w-4xl max-h-[85vh] rounded-3xl glass-panel-elevated border border-surface-border overflow-hidden flex flex-col shadow-2xl bg-background"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Top Bar with Clear 'Move Back' button */}
            <div className="px-6 py-4 border-b border-surface-border flex items-center justify-between bg-surface/80">
              <button
                onClick={() => setIsModalOpen(false)}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20 text-xs font-semibold transition-all"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>← Back to Showcase</span>
              </button>

              <div className="text-xs font-mono font-bold text-foreground">
                sentiment_tfidf_pipeline.ipynb
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleDownload}
                  className="px-3 py-1.5 rounded-xl glass-panel text-xs font-semibold text-foreground flex items-center gap-1.5 hover:border-primary/50 transition-colors"
                >
                  <Download className="w-3.5 h-3.5 text-primary" />
                  <span>Download</span>
                </button>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="p-1.5 rounded-xl hover:bg-surface text-foreground-muted hover:text-foreground transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Content: Step by step notebook walkthrough */}
            <div className="p-6 overflow-y-auto space-y-6 text-sm text-foreground">
              <div className="p-4 rounded-2xl dark:bg-amber-500/10 bg-amber-50 dark:border-amber-500/30 border-amber-300 border text-xs dark:text-amber-300 text-amber-900 flex items-center justify-between">
                <span>
                  <strong>Google Colab Quickstart:</strong> Download this notebook, then open{' '}
                  <a
                    href="https://colab.research.google.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline font-bold"
                  >
                    Google Colab
                  </a>{' '}
                  and select <em>Upload &rarr; sentiment_tfidf_pipeline.ipynb</em> to run.
                </span>
                <a
                  href="https://colab.research.google.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-3 py-1 rounded-lg bg-amber-500 text-black font-bold text-xs whitespace-nowrap ml-3"
                >
                  Launch Colab ↗
                </a>
              </div>

              {cells.map((cell, idx) => (
                <div key={idx} className="rounded-xl border border-surface-border overflow-hidden">
                  <div className="px-4 py-2 bg-surface text-xs font-mono font-bold border-b border-surface-border text-foreground-muted flex items-center justify-between">
                    <span>{cell.title}</span>
                    <span className="text-emerald-400 font-normal">Executed [OK]</span>
                  </div>
                  <pre className="p-4 bg-black/70 font-mono text-xs text-slate-200 overflow-x-auto leading-relaxed">
                    <code>{cell.content}</code>
                  </pre>
                </div>
              ))}
            </div>

            {/* Modal Bottom Bar with Close / Back Option */}
            <div className="px-6 py-3 border-t border-surface-border bg-surface/50 flex items-center justify-between">
              <span className="text-xs font-mono text-foreground-muted">
                Press ESC or click &ldquo;Back to Showcase&rdquo; to return.
              </span>
              <button
                onClick={() => setIsModalOpen(false)}
                className="px-4 py-2 rounded-xl bg-primary text-white font-semibold text-xs shadow-glow-primary hover:bg-primary/90 transition-all"
              >
                Back to Showcase
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
