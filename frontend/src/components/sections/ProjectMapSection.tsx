import React, { useState } from 'react';
import { ProjectMapCanvas } from '../3d/ProjectMapCanvas';
import { PIPELINE_NODES } from '../../data/pipelineNodes';
import {
  GitBranch,
  Database,
  Filter,
  Cpu,
  Zap,
  BarChart3,
  ArrowDown,
  ChevronRight,
  ExternalLink,
  Sparkles
} from 'lucide-react';

const NODE_ICONS: Record<string, React.ReactNode> = {
  dataset: <Database className="w-4 h-4" />,
  preprocessing: <Filter className="w-4 h-4" />,
  training: <Cpu className="w-4 h-4" />,
  prediction: <Zap className="w-4 h-4" />,
  results: <BarChart3 className="w-4 h-4" />,
};

const NODE_DETAILS: Record<string, { summary: string; kpi: string }> = {
  dataset: {
    summary: '10K raw reviews from UCSD/SNAP; neutral 3-star filtered for clean balanced binary labels.',
    kpi: '8,039 Balanced Samples',
  },
  preprocessing: {
    summary: 'Contraction expansion & negation preservation binding "not comfortable" into negative bi-grams.',
    kpi: '-2.18 Negation Weight',
  },
  training: {
    summary: 'Tone-Aware Sublinear TF-IDF + Logistic Regression benchmarked against Sentence-Transformers MiniLM.',
    kpi: '94.82% Test Accuracy',
  },
  prediction: {
    summary: 'Asynchronous FastAPI gateway executing live inference & LIME attributions in under 25ms.',
    kpi: '<25ms Response Latency',
  },
  results: {
    summary: 'Subgroup audit uncovering Macro F1 drops on detailed reviews and hardware terms.',
    kpi: '14.8% Disparity Uncovered',
  },
};

export const ProjectMapSection: React.FC = () => {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const scrollToSection = (targetId: string) => {
    try {
      const el = document.getElementById(targetId);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
      }
    } catch (e) {
      console.warn('Scroll navigation error:', e);
    }
  };

  return (
    <section id="project-map" className="py-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Section Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono text-primary bg-primary/10 border border-primary/20 mb-3">
          <GitBranch className="w-3.5 h-3.5" />
          Interactive Pipeline Architecture
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold font-display tracking-tight text-foreground">
          3D End-to-End Workflow Map
        </h2>
        <p className="text-foreground-muted text-sm sm:text-base max-w-2xl mx-auto mt-2">
          Explore the end-to-end sentiment classification lifecycle. Click or hover on any orbital 3D node
          or stage card below to jump directly to its code, live interactive demo, or audit charts.
        </p>
      </div>

      {/* 1. Staggered 3D WebGL Orbital Canvas */}
      <ProjectMapCanvas
        hoveredNode={hoveredNode}
        onHoverNode={setHoveredNode}
        onSelectNode={scrollToSection}
      />

      {/* 2. Synchronized 5-Stage Interactive Workflow Cards Ribbon */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-3 px-1">
          <span className="text-xs font-mono uppercase tracking-wider text-foreground-muted flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-primary" />
            Pipeline Stages (Click any card to jump)
          </span>
          <span className="hidden sm:inline-block text-[11px] font-mono text-foreground-muted">
            5 Sequential NLP Steps
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
          {PIPELINE_NODES.map((node, index) => {
            const isHovered = hoveredNode === node.id;
            const details = NODE_DETAILS[node.id];
            const icon = NODE_ICONS[node.id];

            return (
              <div
                key={node.id}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                onClick={() => scrollToSection(node.targetId)}
                className={`group cursor-pointer rounded-2xl p-4 transition-all duration-300 relative border-2 flex flex-col justify-between ${
                  isHovered
                    ? 'dark:bg-slate-800 bg-white border-primary shadow-xl shadow-primary/10 -translate-y-1 ring-2 ring-primary/30'
                    : 'dark:bg-slate-900/90 bg-white/90 border-slate-200 dark:border-slate-800 hover:border-primary/50 shadow-sm'
                }`}
              >
                {/* Top card row: Step Badge & Icon */}
                <div>
                  <div className="flex items-center justify-between mb-2.5">
                    <span
                      className="text-[11px] font-mono font-bold px-2 py-0.5 rounded-md border"
                      style={{
                        color: node.color,
                        borderColor: `${node.color}40`,
                        backgroundColor: `${node.color}15`,
                      }}
                    >
                      STEP {node.step}
                    </span>

                    <div
                      className="w-7 h-7 rounded-lg flex items-center justify-center transition-transform group-hover:scale-110"
                      style={{
                        color: node.color,
                        backgroundColor: `${node.color}15`,
                      }}
                    >
                      {icon}
                    </div>
                  </div>

                  {/* Stage Title */}
                  <h3 className="text-sm font-bold font-display text-foreground group-hover:text-primary transition-colors flex items-center justify-between">
                    <span>{node.shortName}</span>
                  </h3>

                  {/* KPI pill */}
                  <div className="mt-1 mb-2">
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full dark:bg-slate-800 bg-slate-100 dark:text-slate-300 text-slate-700 border dark:border-slate-700 border-slate-200">
                      {details.kpi}
                    </span>
                  </div>

                  {/* Description */}
                  <p className="text-xs text-foreground-muted leading-relaxed line-clamp-3">
                    {details.summary}
                  </p>
                </div>

                {/* Bottom link: Jump button */}
                <div className="mt-3 pt-2.5 border-t dark:border-slate-800/80 border-slate-100 flex items-center justify-between text-xs font-medium text-primary">
                  <span className="group-hover:underline text-[11px]">Jump to Section</span>
                  <ChevronRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
