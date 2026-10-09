import React, { useState, useEffect } from 'react';
import { ArrowUp, Mic, Sparkles, MessageSquare } from 'lucide-react';
import { Navbar } from './components/sections/Navbar';
import { HeroSection } from './components/sections/HeroSection';
import { DatasetUploadSection } from './components/sections/DatasetUploadSection';
import { PredictionSection } from './components/sections/PredictionSection';
import { ModelResultsSection } from './components/sections/ModelResultsSection';
import { ColabSection } from './components/sections/ColabSection';
import { Footer } from './components/sections/Footer';
import { SentimentState } from './components/3d/HeroSentimentCenterpiece';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { VoiceChatbotModal } from './components/ui/VoiceChatbotModal';

export const App: React.FC = () => {
  const [currentSentiment, setCurrentSentiment] = useState<SentimentState>('Positive');
  const [showBackToTop, setShowBackToTop] = useState(false);
  const [isVoiceChatOpen, setIsVoiceChatOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 300);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-400 relative">
      {/* Sticky Glassmorphic Navbar with Voice Assistant trigger */}
      <Navbar onOpenVoiceChat={() => setIsVoiceChatOpen(true)} />

      {/* Main Single-Page Showcase Scroll */}
      <main>
        {/* 1. Hero Landing Section with Reactive 3D Centerpiece */}
        <ErrorBoundary>
          <HeroSection
            currentSentiment={currentSentiment}
            onSentimentChange={setCurrentSentiment}
          />
        </ErrorBoundary>

        {/* 2. Dataset Upload Section ("Bring Your Own Data" & 3D Ingestion) */}
        <ErrorBoundary>
          <DatasetUploadSection />
        </ErrorBoundary>

        {/* 3. AI Sentiment Prediction Interface (Live AI Demo - Direct Transition) */}
        <ErrorBoundary>
          <PredictionSection onSentimentResult={setCurrentSentiment} />
        </ErrorBoundary>

        {/* 6. Model Results Dashboard (Heatmap, Curves, Subgroup Audit, 3D Bar) */}
        <ErrorBoundary>
          <ModelResultsSection />
        </ErrorBoundary>

        {/* 7. Google Colab Section & Output Carousel (Directly from Metrics) */}
        <ColabSection />
      </main>

      {/* 11. Footer */}
      <Footer />

      {/* Floating AI Voice Assistant Launcher Button */}
      <button
        onClick={() => setIsVoiceChatOpen(true)}
        aria-label="Open AI Voice Assistant"
        className="fixed bottom-6 right-6 z-40 px-4 py-3 rounded-2xl bg-gradient-to-r from-primary via-indigo-600 to-secondary text-white shadow-xl shadow-primary/30 hover:shadow-primary/50 flex items-center gap-2.5 transition-all duration-300 hover:scale-105 group border border-white/20"
      >
        <div className="relative">
          <Mic className="w-5 h-5 text-white group-hover:animate-pulse" />
          <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
        </div>
        <div className="flex flex-col text-left">
          <span className="text-xs font-bold leading-tight flex items-center gap-1">
            Voice AI <Sparkles className="w-3 h-3 text-amber-300 inline" />
          </span>
          <span className="text-[10px] font-mono opacity-80 leading-none">Ask ML & Data</span>
        </div>
      </button>

      {/* Always-Available Floating "Back to Top" Button */}
      {showBackToTop && (
        <button
          onClick={scrollToTop}
          aria-label="Move Back to Top"
          className="fixed bottom-20 right-6 z-40 px-3.5 py-2.5 rounded-2xl glass-panel-elevated border border-primary/40 text-foreground hover:bg-primary hover:text-white shadow-xl flex items-center gap-2 transition-all duration-300 hover:scale-105"
        >
          <ArrowUp className="w-4 h-4" />
          <span className="text-xs font-mono font-semibold">Back to Top</span>
        </button>
      )}

      {/* Voice Chatbot Modal Component */}
      <VoiceChatbotModal
        isOpen={isVoiceChatOpen}
        onClose={() => setIsVoiceChatOpen(false)}
      />
    </div>
  );
};

export default App;
