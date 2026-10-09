import React, { useState } from 'react';
import { GitCommit, ChevronDown, ChevronUp, Layers, CheckCircle2, BookOpen } from 'lucide-react';
import { METHODOLOGY_STEPS } from '../../data/sampleData';

export const MethodologySection: React.FC = () => {
  const [expandedStep, setExpandedStep] = useState<string | null>('02');

  const toggleStep = (step: string) => {
    setExpandedStep((prev) => (prev === step ? null : step));
  };

  return (
    <section id="methodology" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono text-primary bg-primary/10 border border-primary/20 mb-3">
          <BookOpen className="w-3.5 h-3.5" />
          End-to-End Scientific Rigor
        </div>
        <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight">
          7-Step ML Methodology Timeline
        </h2>
        <p className="text-foreground-muted text-sm sm:text-base max-w-xl mx-auto mt-2">
          From raw text parsing and negation protection to dual model benchmarking, algorithmic subgroup
          fairness auditing, and containerized deployment.
        </p>
      </div>

      {/* Stepper Timeline */}
      <div className="relative border-l-2 border-surface-border ml-4 sm:ml-32 md:ml-40 space-y-8 pl-6 sm:pl-8">
        {METHODOLOGY_STEPS.map((item) => {
          const isExpanded = expandedStep === item.step;
          return (
            <div key={item.step} className="relative group">
              {/* Step indicator badge on the timeline line */}
              <div
                className={`absolute -left-[35px] sm:-left-[43px] top-1 w-8 h-8 rounded-full border-2 flex items-center justify-center font-mono text-xs font-bold transition-all duration-300 ${
                  isExpanded
                    ? 'bg-primary border-primary text-white shadow-glow-primary scale-110'
                    : 'bg-background border-surface-border text-foreground-muted group-hover:border-primary group-hover:text-primary'
                }`}
              >
                {item.step}
              </div>

              {/* Step content card */}
              <div
                className={`rounded-2xl glass-panel p-6 border transition-all duration-300 ${
                  isExpanded
                    ? 'border-primary/50 shadow-lg bg-surface-elevated'
                    : 'border-surface-border hover:border-surface-border/80'
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 cursor-pointer" onClick={() => toggleStep(item.step)}>
                  <div>
                    <span className="text-[11px] font-mono text-primary font-bold uppercase tracking-wider">
                      {item.subtitle}
                    </span>
                    <h3 className="text-lg font-bold font-display text-foreground mt-0.5">
                      {item.title}
                    </h3>
                  </div>

                  <button
                    className="self-start sm:self-center px-3 py-1 rounded-lg text-xs font-medium text-foreground-muted hover:text-foreground flex items-center gap-1.5 glass-panel transition-colors"
                  >
                    <span>{isExpanded ? 'Collapse' : 'Learn more'}</span>
                    {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </button>
                </div>

                <p className="text-sm text-foreground-muted mt-3 leading-relaxed">
                  {item.description}
                </p>

                {/* Expandable Technical Detail */}
                {isExpanded && (
                  <div className="mt-4 pt-4 border-t border-surface-border text-xs dark:text-slate-300 text-slate-700 leading-relaxed dark:bg-black/30 bg-slate-100 p-4 rounded-xl font-mono border dark:border-slate-800 border-slate-200">
                    <span className="text-primary font-bold"># Technical Specification: </span>
                    {item.details}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};
