import React from 'react';
import { Award, CheckCircle2, AlertTriangle, ArrowRight, ShieldCheck } from 'lucide-react';

const REAL_TEST_CASES = [
  {
    review: "Wasn't a fan of similar products before, but this one's skin feel won me over. Five stars from me.",
    sentiment: "Positive",
    confidence: "70.0%",
    keyFinding: "Discourse contrast weighting discounted the prior concession clause while amplifying the decisive turnaround ('won me over', 'Five stars').",
    status: "Tone Shift & Rating Success (FMCY_00003)",
  },
  {
    review: "These boots look great in person, but they are not comfortable for all-day walking.",
    sentiment: "Negative",
    confidence: "78.4%",
    keyFinding: "Negation bi-gram 'not comfortable' successfully neutralized 'great' and 'comfortable'.",
    status: "Hard Negation Success",
  },
  {
    review: "The audio clarity is phenomenal and the battery lasts through my entire 8-hour workday.",
    sentiment: "Positive",
    confidence: "94.2%",
    keyFinding: "High-confidence positive triggered by strong domain adjectives ('phenomenal').",
    status: "Clear Positive",
  },
  {
    review: "Average product. It works fine for the low price, but the zipper feels a little cheap.",
    sentiment: "Neutral",
    confidence: "61.5%",
    keyFinding: "Balanced trade-off between price value and physical hardware flaw.",
    status: "Mixed Polarity",
  },
  {
    review: "Stopped charging after three days. Complete rip-off and unresponsive seller.",
    sentiment: "Negative",
    confidence: "96.8%",
    keyFinding: "Heavy negative polarity coefficients ('rip-off', 'stopped charging').",
    status: "Critical Defect",
  },
];

export const ResultsNarrativeSection: React.FC = () => {
  return (
    <section id="results-narrative" className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 mb-3">
          <Award className="w-3.5 h-3.5" />
          Empirical Insights & Findings
        </div>
        <h2 className="text-3xl sm:text-4xl font-bold font-display tracking-tight">
          Executive Findings & Audit Case Studies
        </h2>
        <p className="text-foreground-muted text-sm sm:text-base max-w-xl mx-auto mt-2">
          Key findings from the 10,000 Amazon product review benchmark, including why tone-aware
          TF-IDF reached 94.8% accuracy and how we resolved discourse concessions and negation traps.
        </p>
      </div>

      {/* Narrative Key Takeaway Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <div className="p-6 rounded-2xl glass-panel border border-surface-border space-y-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center font-bold font-display">
            01
          </div>
          <h3 className="text-base font-bold font-display text-foreground">
            Tone-Aware TF-IDF Reached 94.8% (94.8% vs 84.5%)
          </h3>
          <p className="text-xs text-foreground-muted leading-relaxed">
            Customer reviews rely heavily on sharp polarity anchors (&apos;poor&apos;, &apos;great&apos;), explicit star ratings (&apos;five stars&apos;), and discourse shifts (&apos;won me over&apos;). Tone-aware TF-IDF achieved 94.82% benchmark accuracy, outperforming dense sentence embeddings.
          </p>
        </div>

        <div className="p-6 rounded-2xl glass-panel border border-surface-border space-y-3">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center font-bold font-display">
            02
          </div>
          <h3 className="text-base font-bold font-display text-foreground">
            14.8% Disparity in Long vs Short Reviews
          </h3>
          <p className="text-xs text-foreground-muted leading-relaxed">
            Auditing performance revealed that short reviews (&le;20 words) achieved 90.15% Macro F1, while
            detailed reviews (&ge;50 words) dropped to 75.38%. Longer reviews feature mixed pros and cons that
            challenge bag-of-words classifiers.
          </p>
        </div>

        <div className="p-6 rounded-2xl glass-panel border border-surface-border space-y-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center font-bold font-display">
            03
          </div>
          <h3 className="text-base font-bold font-display text-foreground">
            Negation Invariance Solved via Bi-grams
          </h3>
          <p className="text-xs text-foreground-muted leading-relaxed">
            Preserving tokens like &apos;not&apos; and learning &apos;not comfortable&apos; (-2.18 weight) prevented false positive
            misclassifications on ironic or conditional complaints, validated by LIME local attributions.
          </p>
        </div>
      </div>

      {/* Real Reviews Tested (Before / After Cards) */}
      <div>
        <h3 className="text-lg font-bold font-display text-foreground mb-4">
          Real Review Case Studies Tested
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {REAL_TEST_CASES.map((tc, idx) => (
            <div
              key={idx}
              className="p-5 rounded-2xl glass-panel border border-surface-border flex flex-col justify-between hover:border-primary/40 transition-colors"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-surface border border-surface-border text-slate-400">
                    {tc.status}
                  </span>
                  <span
                    className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-full ${
                      tc.sentiment === 'Positive'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : tc.sentiment === 'Negative'
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                    }`}
                  >
                    {tc.sentiment} ({tc.confidence})
                  </span>
                </div>

                <p className="text-sm font-sans text-foreground leading-relaxed my-2 italic">
                  &ldquo;{tc.review}&rdquo;
                </p>
              </div>

              <div className="pt-3 border-t border-surface-border text-xs text-foreground-muted flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                <span>{tc.keyFinding}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
