import React, { useState, useEffect } from 'react';
import { Sun, Moon, Sparkles, Terminal, Activity, Github, ExternalLink, Menu, X, Mic } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import { checkBackendHealth } from '../../services/api';

interface NavbarProps {
  onOpenVoiceChat?: () => void;
}

const NAV_LINKS = [
  { label: 'Overview', href: '#hero' },
  { label: 'Dataset', href: '#dataset-upload' },
  { label: 'Live AI Demo', href: '#prediction-interface' },
  { label: 'Metrics', href: '#model-results' },
  { label: 'Colab', href: '#colab' },
];

export const Navbar: React.FC<NavbarProps> = ({ onOpenVoiceChat }) => {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === 'dark';
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    checkBackendHealth().then((status) => setBackendOnline(status));
    const interval = setInterval(() => {
      checkBackendHealth().then((status) => setBackendOnline(status));
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'py-3 bg-background/80 backdrop-blur-xl border-b border-surface-border shadow-lg'
          : 'py-5 bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Logo / Brand */}
        <a href="#hero" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-primary to-secondary flex items-center justify-center text-white shadow-glow-primary group-hover:scale-105 transition-transform duration-200">
            <Sparkles className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <span className="font-display font-bold text-base sm:text-lg tracking-tight flex items-center gap-2">
              Sentiment<span className="text-primary font-mono text-sm uppercase px-1.5 py-0.5 rounded bg-primary/10 border border-primary/20">3D AI</span>
            </span>
          </div>
        </a>

        {/* Desktop Nav Links */}
        <div className="hidden lg:flex items-center gap-1 xl:gap-2 text-sm font-medium text-foreground-muted">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="px-3 py-1.5 rounded-lg hover:text-foreground hover:bg-surface transition-colors"
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* Right Actions: Voice Assistant, Backend Status & Theme Toggle */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* AI Voice Assistant Trigger Button */}
          {onOpenVoiceChat && (
            <button
              onClick={onOpenVoiceChat}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-primary/10 via-primary/20 to-secondary/20 hover:from-primary hover:to-secondary border border-primary/30 text-foreground hover:text-white transition-all duration-300 shadow-sm group hover:scale-105"
              title="Open Voice Chatbot Assistant"
              aria-label="Open Voice Chatbot Assistant"
            >
              <Mic className="w-4 h-4 text-primary group-hover:text-white group-hover:animate-pulse" />
              <span className="text-xs font-semibold hidden md:inline">Voice AI</span>
            </button>
          )}

          {/* Live Backend Indicator */}
          <div
            className={`hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono border backdrop-blur-md transition-colors ${
              backendOnline
                ? 'dark:bg-emerald-500/10 bg-emerald-100/90 dark:text-emerald-400 text-emerald-800 dark:border-emerald-500/30 border-emerald-300'
                : 'dark:bg-amber-500/10 bg-amber-100/90 dark:text-amber-400 text-amber-800 dark:border-amber-500/30 border-amber-300'
            }`}
            title={backendOnline ? 'Connected to local FastAPI backend' : 'Backend offline: Using built-in NLP fallback'}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
              }`}
            />
            {backendOnline ? 'API Connected' : 'Offline Engine'}
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle Theme"
            className="w-10 h-10 rounded-xl glass-panel flex items-center justify-center text-foreground hover:border-primary/50 transition-all duration-300 hover:scale-105 shadow-sm"
          >
            {isDark ? (
              <Sun className="w-5 h-5 text-amber-400 transition-transform hover:rotate-45" />
            ) : (
              <Moon className="w-5 h-5 text-indigo-600 transition-transform hover:-rotate-12" />
            )}
          </button>

          {/* Mobile menu trigger */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden w-10 h-10 rounded-xl glass-panel flex items-center justify-center text-foreground"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-background/95 backdrop-blur-2xl border-b border-surface-border px-6 py-6 space-y-3 shadow-2xl">
          <div className="flex items-center justify-between pb-3 border-b border-surface-border">
            <span className="text-xs font-mono uppercase text-foreground-muted">Navigation Menu</span>
            <div
              className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono border ${
                backendOnline
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              {backendOnline ? 'FastAPI Online' : 'Client NLP Engine'}
            </div>
          </div>

          {onOpenVoiceChat && (
            <button
              onClick={() => {
                setMobileMenuOpen(false);
                onOpenVoiceChat();
              }}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-gradient-to-r from-primary to-secondary text-white font-medium text-sm shadow-md"
            >
              <Mic className="w-4 h-4 animate-pulse" />
              <span>Ask Voice AI Assistant</span>
            </button>
          )}

          <div className="grid grid-cols-2 gap-2 pt-2">
            {NAV_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className="px-3 py-2 rounded-lg text-sm text-foreground hover:bg-surface font-medium"
              >
                {link.label}
              </a>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
};
