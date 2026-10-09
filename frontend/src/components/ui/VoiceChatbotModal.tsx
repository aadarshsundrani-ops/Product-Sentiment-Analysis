import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Send,
  Sparkles,
  Bot,
  User,
  X,
  RotateCcw,
  MessageSquare,
  Square,
  AlertCircle
} from 'lucide-react';
import {
  ChatMessage,
  askAssistant,
  VoiceSpeaker,
  VoiceRecognizer,
  SUGGESTED_QUESTIONS,
} from '../../services/chatService';

interface VoiceChatbotModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const VoiceChatbotModal: React.FC<VoiceChatbotModalProps> = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: "👋 Hi! I'm your SentimentAI Voice Assistant. Ask me anything via voice or text about our TF-IDF machine learning pipeline, the 10,000 Amazon reviews dataset, our 14.8% disparity audit, negation preservation ('not comfortable'), or test any product review!",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(true);
  const [statusNotice, setStatusNotice] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
      setTimeout(() => inputRef.current?.focus(), 200);
    } else {
      VoiceSpeaker.stop();
      VoiceRecognizer.stopListening();
      setIsListening(false);
      setIsSpeaking(false);
    }
  }, [isOpen]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, isListening]);

  // Clean up speech on unmount
  useEffect(() => {
    return () => {
      VoiceSpeaker.stop();
      VoiceRecognizer.stopListening();
    };
  }, []);

  const handleSendMessage = async (textToSend?: string, fromVoice = false) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isVoiceInput: fromVoice,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);
    setStatusNotice(null);

    try {
      const replyText = await askAssistant(text);

      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: replyText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Spoken voice assignment
      if (voiceOutputEnabled && VoiceSpeaker.isSupported()) {
        setIsSpeaking(true);
        VoiceSpeaker.speak(
          replyText,
          () => setIsSpeaking(false),
          () => setIsSpeaking(false)
        );
      }
    } catch (err) {
      console.error('Chat error:', err);
      const fallbackMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: "I encountered an issue processing that query, but our tone-aware TF-IDF model achieved 94.82% test accuracy on held-out Amazon reviews. Feel free to ask about our algorithms, dataset, or disparity audit!",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, fallbackMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      VoiceRecognizer.stopListening();
      setIsListening(false);
      setStatusNotice(null);
      return;
    }

    if (!VoiceRecognizer.isSupported()) {
      setStatusNotice('Microphone speech recognition is not supported in this browser. Please type your query.');
      setTimeout(() => setStatusNotice(null), 4000);
      return;
    }

    // Stop speaking if assistant is currently talking
    VoiceSpeaker.stop();
    setIsSpeaking(false);

    setIsListening(true);
    setStatusNotice('🎙️ Listening... Please speak your question now.');

    VoiceRecognizer.startListening(
      (transcript) => {
        setIsListening(false);
        setStatusNotice(null);
        if (transcript.trim()) {
          handleSendMessage(transcript, true);
        }
      },
      (err) => {
        console.warn('Speech error:', err);
        setIsListening(false);
        setStatusNotice('Microphone access interrupted or not granted. You can type below.');
        setTimeout(() => setStatusNotice(null), 4000);
      },
      () => {
        setIsListening(false);
      }
    );
  };

  const handleStopSpeaking = () => {
    VoiceSpeaker.stop();
    setIsSpeaking(false);
  };

  const handleReplaySpeech = (text: string) => {
    if (!VoiceSpeaker.isSupported()) return;
    setIsSpeaking(true);
    VoiceSpeaker.speak(
      text,
      () => setIsSpeaking(false),
      () => setIsSpeaking(false)
    );
  };

  const clearChat = () => {
    VoiceSpeaker.stop();
    setIsSpeaking(false);
    setMessages([
      {
        id: 'welcome',
        sender: 'assistant',
        text: "👋 Chat cleared. What would you like to explore next? You can ask about our ML algorithms, the dataset, or speak your question directly using the microphone!",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-end sm:justify-center p-0 sm:p-4 bg-black/70 backdrop-blur-md animate-fade-in">
      <div
        className="w-full sm:max-w-xl h-[88vh] sm:h-[630px] dark:bg-slate-900 bg-slate-50 border-2 dark:border-slate-700 border-slate-300 rounded-t-3xl sm:rounded-3xl shadow-2xl flex flex-col overflow-hidden relative"
        role="dialog"
        aria-labelledby="chatbot-title"
      >
        {/* Header */}
        <div className="px-5 py-3.5 border-b-2 dark:border-slate-800 border-slate-200 dark:bg-slate-900/95 bg-white/95 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-600 flex items-center justify-center text-white shadow-md">
                <Sparkles className="w-5 h-5 animate-pulse text-amber-300" />
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 rounded-full border-2 border-white dark:border-slate-900 animate-ping" />
              <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 rounded-full border-2 border-white dark:border-slate-900" />
            </div>
            <div>
              <h3 id="chatbot-title" className="text-sm font-bold dark:text-white text-slate-900 flex items-center gap-2">
                SentimentAI Assistant
                <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-full dark:bg-indigo-950 dark:text-indigo-300 dark:border dark:border-indigo-700/60 bg-indigo-50 text-indigo-700 border border-indigo-200">
                  Voice Enabled
                </span>
              </h3>
              <p className="text-[12px] dark:text-slate-300 text-slate-600 font-medium">
                Dual NLP Benchmark • 10K Reviews • Disparity Audit
              </p>
            </div>
          </div>

          {/* Header Action buttons */}
          <div className="flex items-center gap-1.5">
            {/* Speaking visualizer / stop button */}
            {isSpeaking && (
              <button
                onClick={handleStopSpeaking}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-rose-500/15 text-rose-600 dark:text-rose-400 border border-rose-500/40 text-xs font-mono font-semibold animate-pulse hover:bg-rose-500/25 transition-colors"
                title="Stop spoken audio"
              >
                <Square className="w-3 h-3 fill-current" />
                <span>Stop Voice</span>
              </button>
            )}

            {/* Voice toggle button */}
            <button
              onClick={() => {
                const nextState = !voiceOutputEnabled;
                setVoiceOutputEnabled(nextState);
                if (!nextState) handleStopSpeaking();
              }}
              className={`p-2 rounded-xl transition-colors ${
                voiceOutputEnabled
                  ? 'text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800'
                  : 'text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800'
              }`}
              title={voiceOutputEnabled ? 'Voice is ON (click to mute)' : 'Voice is MUTED (click to enable)'}
              aria-label="Toggle voice output"
            >
              {voiceOutputEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4 text-rose-500" />}
            </button>

            {/* Clear conversation */}
            <button
              onClick={clearChat}
              className="p-2 rounded-xl text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors"
              title="Clear conversation"
              aria-label="Clear chat"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            {/* Close modal */}
            <button
              onClick={onClose}
              className="p-2 rounded-xl text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors ml-1"
              aria-label="Close voice assistant"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Status notification banner (e.g. listening / speech error) */}
        {statusNotice && (
          <div className="px-4 py-2 dark:bg-indigo-950/70 bg-indigo-100 border-b dark:border-indigo-800/60 border-indigo-200 text-xs dark:text-indigo-200 text-indigo-900 font-medium flex items-center gap-2 animate-fade-in shrink-0">
            {isListening ? (
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
            )}
            <span>{statusNotice}</span>
          </div>
        )}

        {/* Messages scroll area */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4 font-sans text-sm dark:bg-slate-950/60 bg-slate-100/70">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start gap-3 ${
                msg.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {msg.sender === 'assistant' && (
                <div className="w-8 h-8 rounded-xl bg-indigo-600 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-sm">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-md relative group ${
                  msg.sender === 'user'
                    ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white font-medium rounded-tr-sm'
                    : 'dark:bg-slate-800 bg-white border dark:border-slate-700/80 border-slate-200 rounded-tl-sm'
                }`}
              >
                {/* Voice input indicator badge */}
                {msg.isVoiceInput && (
                  <div className="flex items-center gap-1 text-[11px] text-indigo-200 font-mono font-semibold uppercase tracking-wider mb-1.5">
                    <Mic className="w-3 h-3 text-amber-300 animate-pulse" />
                    <span>Spoken Voice Input</span>
                  </div>
                )}

                {/* High contrast, reading-friendly message content */}
                <p
                  className={`text-[14.5px] leading-relaxed whitespace-pre-wrap ${
                    msg.sender === 'user'
                      ? 'text-white'
                      : 'dark:text-slate-100 text-slate-900 font-normal selection:bg-indigo-500 selection:text-white'
                  }`}
                >
                  {msg.text}
                </p>

                {/* Message footer: timestamp and replay button */}
                <div
                  className={`flex items-center justify-between mt-2 pt-1 border-t text-[11px] font-mono ${
                    msg.sender === 'user'
                      ? 'border-white/20 text-indigo-100'
                      : 'dark:border-slate-700/60 border-slate-200 dark:text-slate-400 text-slate-500'
                  }`}
                >
                  <span>{msg.timestamp}</span>

                  {/* Replay audio button for assistant responses */}
                  {msg.sender === 'assistant' && (
                    <button
                      onClick={() => handleReplaySpeech(msg.text)}
                      className="ml-3 dark:text-indigo-400 text-indigo-600 hover:underline flex items-center gap-1 font-semibold transition-colors"
                      title="Listen to this explanation again"
                    >
                      <Volume2 className="w-3.5 h-3.5" />
                      <span>Re-listen</span>
                    </button>
                  )}
                </div>
              </div>

              {msg.sender === 'user' && (
                <div className="w-8 h-8 rounded-xl bg-slate-700 dark:bg-slate-700 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-sm">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {/* Typing / Thinking indicator */}
          {isLoading && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-indigo-600 text-white flex items-center justify-center shrink-0 shadow-sm">
                <Bot className="w-4 h-4 animate-spin" />
              </div>
              <div className="dark:bg-slate-800 bg-white border dark:border-slate-700/80 border-slate-200 rounded-2xl px-4 py-3 rounded-tl-sm flex items-center gap-2 shadow-sm">
                <span className="w-2 h-2 rounded-full bg-indigo-600 animate-bounce [animation-delay:-0.3s]" />
                <span className="w-2 h-2 rounded-full bg-indigo-600 animate-bounce [animation-delay:-0.15s]" />
                <span className="w-2 h-2 rounded-full bg-indigo-600 animate-bounce" />
                <span className="text-xs dark:text-slate-300 text-slate-700 font-semibold ml-2 font-mono">
                  Consulting Sentiment Knowledge Base...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Suggested question chips */}
        <div className="px-4 py-2.5 border-t-2 dark:border-slate-800 border-slate-200 dark:bg-slate-900 bg-white overflow-x-auto no-scrollbar flex items-center gap-2 shrink-0">
          <span className="text-[11px] font-mono font-bold dark:text-slate-400 text-slate-600 shrink-0 uppercase tracking-wider">
            Quick Prompts:
          </span>
          {SUGGESTED_QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleSendMessage(q)}
              className="text-xs px-3 py-1.5 rounded-full dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-200 dark:hover:text-white bg-slate-100 hover:bg-indigo-50 text-slate-800 hover:text-indigo-700 border dark:border-slate-700 border-slate-300 hover:border-indigo-400 font-medium transition-all shrink-0 whitespace-nowrap shadow-xs"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <div className="p-3.5 dark:bg-slate-900 bg-white border-t-2 dark:border-slate-800 border-slate-200 shrink-0">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-center gap-2.5"
          >
            {/* Voice microphone button */}
            <button
              type="button"
              onClick={toggleListening}
              className={`p-3 rounded-2xl transition-all duration-300 relative group flex items-center justify-center shrink-0 ${
                isListening
                  ? 'bg-rose-600 text-white shadow-lg shadow-rose-600/40 scale-105 animate-pulse'
                  : 'dark:bg-slate-800 dark:text-indigo-400 dark:hover:bg-indigo-600 dark:hover:text-white bg-indigo-50 text-indigo-700 hover:bg-indigo-600 hover:text-white border-2 dark:border-slate-700 border-indigo-200 shadow-sm'
              }`}
              title={isListening ? 'Stop listening' : 'Click to speak your question'}
              aria-label="Toggle voice input"
            >
              {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              {isListening && (
                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-rose-400 animate-ping" />
              )}
            </button>

            {/* High contrast text input */}
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder={isListening ? '🎙️ Listening... speak now' : 'Ask about algorithms, dataset, disparity, or test a review...'}
                disabled={isListening}
                className="w-full dark:bg-slate-800 bg-white border-2 dark:border-slate-700 border-slate-300 rounded-xl px-4 py-2.5 text-sm dark:text-white text-slate-900 placeholder:text-slate-400 dark:placeholder:text-slate-500 font-normal focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
              />
            </div>

            {/* Send button */}
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="p-3 rounded-2xl bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md shrink-0 flex items-center justify-center"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>

          {/* Web speech info badge */}
          <div className="mt-2 flex items-center justify-between text-[11px] dark:text-slate-400 text-slate-600 px-1 font-mono font-medium">
            <span>Click Mic to speak • Spoken audio {voiceOutputEnabled ? 'ENABLED' : 'MUTED'}</span>
            <span>Web Speech API (Safari & Chrome)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
