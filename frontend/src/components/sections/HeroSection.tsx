import React from 'react';
import { ArrowDown, Play, ExternalLink, Cpu, BarChart3, ShieldCheck } from 'lucide-react';
import { HeroSentimentCenterpiece, SentimentState } from '../3d/HeroSentimentCenterpiece';

interface HeroSectionProps {
  currentSentiment: SentimentState;
  onSentimentChange: (s: SentimentState) => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  currentSentiment,
  onSentimentChange,
}) => {
  return (
    <section
      id="hero"
      className="relative min-h-screen pt-28 pb-16 flex flex-col justify-between overflow-hidden"
    >
      {/* Background ambient neon glow spheres */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-tr from-primary/20 via-secondary/15 to-accent/10 rounded-full blur-[140px] pointer-events-none -z-10" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full flex-1 flex flex-col justify-center">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Column: Headlines & Call to Actions */}
          <div className="lg:col-span-7 space-y-6 text-center lg:text-left z-10">
            {/* Pill Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-surface-border glass-panel text-xs font-mono font-medium text-foreground-muted">
              <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>State-of-the-Art NLP Benchmark</span>
              <span className="text-surface-border">|</span>
              <span className="text-primary font-semibold">Amazon 10K Corpus</span>
            </div>

            {/* Main Title with Animated Gradient Reveal */}
            <h1 className="text-4xl sm:text-5xl xl:text-6xl font-extrabold font-display tracking-tight leading-[1.1]">
              Product Sentiment <br />
              <span className="gradient-text-hero">
                Analysis & Auditing
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-lg sm:text-xl text-foreground-muted max-w-2xl font-normal leading-relaxed mx-auto lg:mx-0">
              An AI-powered NLP system that classifies customer reviews into{' '}
              <strong className="dark:text-emerald-400 text-emerald-700 font-semibold">Positive</strong>,{' '}
              <strong className="dark:text-rose-400 text-rose-700 font-semibold">Negative</strong>, or{' '}
              <strong className="dark:text-amber-400 text-amber-700 font-semibold">Neutral</strong> with confidence scoring,
              rigorous subgroup auditing, and explainable token attributions.
            </p>

            {/* Interactive Sentiment State Pill Switcher */}
            <div className="pt-2">
              <div className="text-xs font-mono uppercase text-foreground-muted mb-2 tracking-wider">
                Preview Reactive 3D Sentiment State:
              </div>
              <div className="inline-flex p-1 rounded-xl glass-panel border border-surface-border gap-1">
                {(['Positive', 'Neutral', 'Negative'] as SentimentState[]).map((s) => (
                  <button
                    key={s}
                    onClick={() => onSentimentChange(s)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                      currentSentiment === s
                        ? s === 'Positive'
                          ? 'bg-emerald-500 text-white shadow-glow-positive'
                          : s === 'Negative'
                          ? 'bg-rose-500 text-white shadow-glow-negative'
                          : 'bg-amber-500 text-white shadow-glow-neutral'
                        : 'text-foreground-muted hover:text-foreground'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {/* CTA Action Buttons */}
            <div className="flex flex-wrap items-center justify-center lg:justify-start gap-4 pt-4">
              <a
                href="#prediction-interface"
                className="px-6 py-3.5 rounded-xl font-semibold text-sm bg-primary hover:bg-primary/90 text-white shadow-glow-primary transition-all duration-200 flex items-center gap-2 hover:scale-[1.02] active:scale-[0.98]"
              >
                <Play className="w-4 h-4 fill-white" />
                Try the Live Demo
              </a>
              <a
                href="#colab"
                className="px-6 py-3.5 rounded-xl font-semibold text-sm glass-panel hover:border-primary/40 text-foreground transition-all duration-200 flex items-center gap-2 hover:scale-[1.02]"
              >
                <ExternalLink className="w-4 h-4 text-primary" />
                View on Google Colab
              </a>
            </div>

            {/* Metric Highlights Ticker */}
            <div className="pt-6 grid grid-cols-3 gap-3 max-w-lg border-t border-surface-border">
              <div>
                <div className="text-2xl font-bold font-display text-primary">94.8%</div>
                <div className="text-xs text-foreground-muted">Test Accuracy</div>
              </div>
              <div>
                <div className="text-2xl font-bold font-display text-secondary">10,000</div>
                <div className="text-xs text-foreground-muted">Audited Reviews</div>
              </div>
              <div>
                <div className="text-2xl font-bold font-display dark:text-emerald-400 text-emerald-700">&lt;25ms</div>
                <div className="text-xs text-foreground-muted">Inference Latency</div>
              </div>
            </div>
          </div>

          {/* Right Column: 3D Centerpiece Canvas */}
          <div className="lg:col-span-5 h-[420px] sm:h-[480px] lg:h-[540px] w-full flex items-center justify-center relative">
            <HeroSentimentCenterpiece sentiment={currentSentiment} interactive={true} />
          </div>
        </div>
      </div>

      {/* Bouncing Scroll Down Indicator */}
      <div className="flex flex-col items-center justify-center pt-8 text-foreground-muted animate-bounce pointer-events-none">
        <span className="text-xs font-mono uppercase tracking-widest mb-1 text-slate-400">Scroll to Explore</span>
        <ArrowDown className="w-4 h-4 text-primary" />
      </div>
    </section>
  );
};
