import React from 'react';
import { Github, Sparkles, Terminal, Sun, Moon, ExternalLink, Code } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

const TECH_BADGES = [
  'React 18',
  'Three.js / WebGL',
  '@react-three/fiber',
  'Tailwind CSS',
  'FastAPI (Python)',
  'scikit-learn',
  'Sentence-Transformers',
  'LIME Explainability',
  'Vite',
  'TypeScript',
];

export const Footer: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <footer className="border-t border-surface-border bg-background/50 backdrop-blur-md pt-16 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 pb-12 border-b border-surface-border">
          {/* Brand & Project Info */}
          <div className="md:col-span-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-primary to-secondary flex items-center justify-center text-white shadow-glow-primary">
                <Sparkles className="w-4 h-4" />
              </div>
              <span className="font-display font-bold text-lg text-foreground tracking-tight">
                Product Sentiment Analysis <span className="text-primary text-xs font-mono uppercase">3D AI</span>
              </span>
            </div>

            <p className="text-xs sm:text-sm text-foreground-muted max-w-md leading-relaxed">
              An interactive 3D WebGL showcase of an end-to-end Machine Learning NLP system.
              Engineered with rigorous subgroup auditing, negation preservation, and real-time
              confidence scoring.
            </p>

            {/* Tech Badges */}
            <div className="flex flex-wrap gap-1.5 pt-2">
              {TECH_BADGES.map((badge) => (
                <span
                  key={badge}
                  className="px-2.5 py-1 rounded-md text-[10px] font-mono bg-surface border border-surface-border text-foreground-muted"
                >
                  {badge}
                </span>
              ))}
            </div>
          </div>

          {/* Quick Anchor Navigation */}
          <div className="md:col-span-3 space-y-3">
            <div className="text-xs font-mono uppercase tracking-wider text-foreground font-semibold">
              Sections
            </div>
            <ul className="space-y-2 text-xs text-foreground-muted font-medium">
              <li>
                <a href="#hero" className="hover:text-primary transition-colors">Hero Overview</a>
              </li>
              <li>
                <a href="#dataset-upload" className="hover:text-primary transition-colors">Dataset Ingestion</a>
              </li>
              <li>
                <a href="#prediction-interface" className="hover:text-primary transition-colors">Live Prediction Demo</a>
              </li>
              <li>
                <a href="#model-results" className="hover:text-primary transition-colors">Subgroup Audit & Metrics</a>
              </li>
              <li>
                <a href="#colab" className="hover:text-primary transition-colors">Google Colab Pipeline</a>
              </li>
            </ul>
          </div>

          {/* Credits & Theme Toggle Reminder */}
          <div className="md:col-span-3 space-y-4">
            <div className="text-xs font-mono uppercase tracking-wider text-foreground font-semibold">
              Project & Resources
            </div>

            <div className="space-y-2 text-xs text-foreground-muted">
              <a
                href="#colab"
                className="flex items-center gap-1.5 hover:text-foreground transition-colors"
              >
                <ExternalLink className="w-3.5 h-3.5 text-primary" />
                <span>Google Colab Notebook</span>
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 hover:text-foreground transition-colors"
              >
                <Github className="w-3.5 h-3.5 text-primary" />
                <span>GitHub Repository</span>
              </a>
            </div>

            {/* Small Theme Toggle Reminder */}
            <div className="pt-2">
              <div className="text-[11px] text-foreground-muted mb-2 font-mono">Theme Preference:</div>
              <button
                onClick={toggleTheme}
                className="px-3 py-1.5 rounded-xl glass-panel text-xs flex items-center gap-2 text-foreground hover:border-primary/50 transition-all"
              >
                {isDark ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-indigo-600" />}
                <span>Active: {isDark ? 'Dark Mode' : 'Light Mode'} (Click to switch)</span>
              </button>
            </div>
          </div>
        </div>

        {/* Bottom copyright row */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between text-xs text-foreground-muted font-mono gap-4">
          <div>
            Built for Machine Learning & NLP Sentiment Auditing Showcase &bull; 2026
          </div>
          <div>
            FastAPI REST Server + Three.js WebGL &bull; High Performance
          </div>
        </div>
      </div>
    </footer>
  );
};
