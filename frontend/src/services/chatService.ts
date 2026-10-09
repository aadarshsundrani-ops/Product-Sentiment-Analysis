export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  isVoiceInput?: boolean;
}

const CANDIDATE_URLS = [
  import.meta.env.VITE_API_BASE_URL,
  'http://127.0.0.1:8000',
  'http://127.0.0.1:8001',
].filter(Boolean) as string[];

export const SUGGESTED_QUESTIONS = [
  "What ML / DL algorithms were used?",
  "Tell me about the Amazon dataset and balance.",
  "Why is there a 14.8% disparity on detailed reviews?",
  "How does your negation tokenizer solve 'not comfortable'?",
  "What are the top positive and negative word weights?",
  "How was LIME used for local explainability?",
  "What are the 6 product categories in the dataset?",
  "Where can I find the Google Colab notebook?",
];

export const PROJECT_KNOWLEDGE: Record<string, string> = {
  algorithms:
    "We trained a production-grade Tone-Aware Sublinear TF-IDF pipeline (1–3 n-grams, 35,000 vocabulary features, discourse contrast weighting, and explicit rating anchors) coupled with an L2-regularized Logistic Regression classifier (C=3.0, solver='liblinear'). The model achieves 94.82% test accuracy and 94.80% Macro F1 on held-out reviews because customer reviews are heavily anchored by sharp lexical polarity tokens and explicit rating discourse.",

  dataset:
    "We used the public Amazon Customer Reviews Corpus (McAuley Lab at UCSD / Stanford SNAP; Hugging Face juliensimon/amazon-shoe-reviews). It contains 10,000 raw customer reviews. We binarized 4–5 star reviews as Positive and 1–2 star reviews as Negative, while deliberately excluding neutral 3-star reviews (1,961 samples) to eliminate label ambiguity. This left 8,039 balanced reviews (50.04% Positive, 49.96% Negative) across 6 categories.",

  categories:
    "The dataset spans 6 footwear & accessory categories: Casual Footwear (75.2%, 6,045 samples), Boots (6.0%, 482 samples), Sandals (5.9%, 474 samples), Dress Shoes (5.1%, 410 samples), Bags & Accessories (4.5%, 362 samples), and Sneakers (3.4%, 273 samples). Casual footwear dominates the distribution, while Bags & Accessories showed the lowest classification performance due to unique hardware terminology.",

  disparity:
    "Our algorithmic subgroup audit on the tone-aware model demonstrated robust cross-slice generalization: Short reviews (≤20 words) achieved 96.85% Macro F1, Medium reviews achieved 94.42% F1, and Detailed reviews achieved 89.15% Macro F1. Long reviews contain mixed clauses (e.g., 'Great design, but the sole unglued'), which are properly balanced by our discourse weighting. In category audits, Boots (94.97% F1) and Casual Footwear (95.12% F1) perform exceptionally well.",

  negation:
    "Standard off-the-shelf NLP pipelines strip stop-words like 'not', converting 'not comfortable' into 'comfortable'—causing catastrophic false-positive misclassifications. We built a custom tokenizer that expands contractions ('don't' -> 'do not', 'wasn't' -> 'was not'), preserves negation bi-grams, handles contrastive discourse concessions ('Wasn't a fan... but this one won me over'), and anchors explicit rating phrases ('five stars', '1 star'). As a result, our model assigned strong positive attribution to turnaround phrases like 'won me over' (+4.85) and negative coefficients to 'waste of money' (-5.12).",

  features:
    "Our top positive word coefficients in Logistic Regression are: 'great' (+9.21), 'love' (+6.84), 'five stars' (+5.24), 'won me over' (+4.85), 'comfortable' (+4.37), 'perfect' (+3.71), and 'good' (+3.33). Our top negative word coefficients are: 'not' (-8.56), 'one star' (-6.50), 'waste of money' (-5.12), 'disappointed' (-4.06), 'cheap' (-3.49), 'too' (-3.30), 'poor' (-3.52), and 'returned' (-3.10).",

  lime:
    "We integrated token attribution to verify model predictions on tricky test instances. For example, on 'These boots look great but are not comfortable at all', 'not comfortable' receives strong negative attribution, overriding 'great' and confirming the negative prediction with high confidence.",

  metrics:
    "On held-out test reviews (N=1,567), our tone-aware TF-IDF model achieved: Accuracy: 94.82%, Precision: 95.14%, Recall: 94.48%, and Macro F1: 94.80%. The confusion matrix shows 742 True Positives, 744 True Negatives, 38 False Positives, and 43 False Negatives.",

  preprocessing:
    "Our text preprocessing pipeline consists of: 1) HTML entity unescaping, 2) Lowercasing, 3) Contraction expansion (e.g., 'won't' -> 'will not'), 4) Negation preservation logic that binds 'not', 'never', and 'no' to succeeding adjectives, 5) Punctuation stripping while maintaining sentiment-bearing compound tokens, and 6) Assigning length buckets: Short (≤20 words), Medium (21-49 words), and Detailed (≥50 words).",

  architecture:
    "The system consists of: 1) A React 18 + Vite + Three.js WebGL frontend with responsive 3D visualizations, 2) A high-performance FastAPI asynchronous REST gateway on port 8000/8001 with sub-25ms latency, 3) Scikit-learn TF-IDF inference pipelines, and 4) An automated client-side offline fallback engine with full domain knowledge.",

  colab:
    "The complete reproducible Jupyter / Google Colab notebook is available at sentiment_analysis_pipeline.ipynb. You can preview all executed code cells and outputs directly in our interactive Colab section or download the .ipynb file with 1 click.",

  overview:
    "This project is 'Product Sentiment Analysis' — an end-to-end Machine Learning and NLP system built to demonstrate production-grade sentiment classification, interpretability with token coefficients, and algorithmic fairness auditing on customer reviews. It features pure TF-IDF n-gram vectorization, negation preservation, interactive 3D visualizations, and this two-way voice assistant.",

  voice:
    "I use the Web Speech API for two-way audio assignment: SpeechRecognition (or webkitSpeechRecognition) to transcribe your voice questions into text, and SpeechSynthesis to speak responses out loud using natural language voices. You can speak to me or type questions at any time!"
};

export async function askAssistant(message: string): Promise<string> {
  const cleanMsg = message.trim();
  if (!cleanMsg) {
    return "How can I help you today? Ask me about our ML algorithms, dataset, subgroup audit, negation handling, or test any product review!";
  }

  // Try backend first
  for (const url of CANDIDATE_URLS) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(`${url}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: cleanMsg }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        if (data.response) return data.response;
      }
    } catch {
      // Continue to next candidate
    }
  }

  // Offline NLP Domain Intent Classifier
  const lower = cleanMsg.toLowerCase();

  if (/(algorithm|model|architecture|logistic|minilm|transformer|tf-idf|tfidf|bert|neural)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.algorithms;
  }
  if (/(category|categories|shoes|boots|sandals|casual|bags)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.categories;
  }
  if (/(dataset|data|amazon|sample|rows|balance|star|split|train|test)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.dataset;
  }
  if (/(disparity|subgroup|audit|detailed|short|length|fairness|bias|gap)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.disparity;
  }
  if (/(negation|not comfortable|not|contraction|preprocess|clean|tokenizer|handling)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.negation;
  }
  if (/(preprocess|cleaning|tokeniz|pipeline|steps|bucket)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.preprocessing;
  }
  if (/(feature|coefficient|weight|positive word|negative word|driver|vocabulary)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.features;
  }
  if (/(lime|explain|interpretability|local|attribution|why)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.lime;
  }
  if (/(accuracy|metric|precision|recall|f1|score|confusion matrix|heatmap|performance)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.metrics;
  }
  if (/(colab|notebook|jupyter|ipynb|download|code)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.colab;
  }
  if (/(fastapi|backend|rest|api|server|deploy|fullstack|stack)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.architecture;
  }
  if (/(voice|speech|listen|speak|audio|mic|microphone|assignment)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.voice;
  }
  if (/(project|overview|what is|summary|goal|purpose|showcase)/i.test(lower)) {
    return PROJECT_KNOWLEDGE.overview;
  }

  // If asking to test or classify a review
  if (lower.split(/\s+/).length >= 4) {
    return `That sounds like a product review! In our pipeline, this would be tokenized with negation preservation and evaluated through our TF-IDF + Logistic Regression model. You can test it live in the "Live AI Demo" section, where it calculates class probabilities, confidence, and LIME attributions in under 25 milliseconds.`;
  }

  return "I'm the SentimentAI Project Voice Assistant. I can explain our ML architectures (TF-IDF vs MiniLM), the 10,000 Amazon reviews dataset, our 14.8% review length disparity audit, negation preservation, or feature weights. What would you like to know?";
}

// Web Speech API Voice Synthesis (Text-to-Speech)
export class VoiceSpeaker {
  private static synth: SpeechSynthesis | null = typeof window !== 'undefined' ? window.speechSynthesis : null;
  private static currentUtterance: SpeechSynthesisUtterance | null = null;

  public static isSupported(): boolean {
    return typeof window !== 'undefined' && 'speechSynthesis' in window;
  }

  public static speak(text: string, onEnd?: () => void, onError?: () => void) {
    if (!this.synth) return;
    this.stop();

    // Clean markdown/symbols from text before speaking
    const cleanText = text
      .replace(/[#*`_~]/g, '')
      .replace(/\(P\(Pos\)[^)]*\)/g, '')
      .replace(/\[OK\]/g, 'okay')
      .replace(/≤/g, 'less than or equal to')
      .replace(/≥/g, 'greater than or equal to')
      .replace(/%/g, ' percent')
      .replace(/\s+/g, ' ')
      .trim();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;

    // Pick natural English voice if available
    const voices = this.synth.getVoices();
    const preferredVoice =
      voices.find((v) => (v.name.includes('Samantha') || v.name.includes('Daniel') || v.name.includes('Google US') || v.name.includes('Natural')) && v.lang.startsWith('en')) ||
      voices.find((v) => v.lang.startsWith('en'));

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.onend = () => {
      this.currentUtterance = null;
      if (onEnd) onEnd();
    };

    utterance.onerror = () => {
      this.currentUtterance = null;
      if (onError) onError();
    };

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
  }

  public static stop() {
    if (this.synth) {
      this.synth.cancel();
      this.currentUtterance = null;
    }
  }

  public static isSpeaking(): boolean {
    return !!(this.synth && (this.synth.speaking || this.synth.pending));
  }
}

// Web Speech API Voice Recognition (Speech-to-Text)
export class VoiceRecognizer {
  private static recognition: any = null;

  public static isSupported(): boolean {
    return typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);
  }

  public static startListening(
    onResult: (transcript: string) => void,
    onError: (err: any) => void,
    onEnd: () => void
  ) {
    if (!this.isSupported()) {
      onError(new Error('Speech recognition not supported in this browser.'));
      return;
    }

    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    this.stopListening();

    const rec = new SpeechRec();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = 'en-US';

    rec.onresult = (event: any) => {
      if (event.results && event.results[0] && event.results[0][0]) {
        const transcript = event.results[0][0].transcript;
        onResult(transcript);
      }
    };

    rec.onerror = (event: any) => {
      onError(event.error);
    };

    rec.onend = () => {
      onEnd();
    };

    this.recognition = rec;
    try {
      rec.start();
    } catch (e) {
      onError(e);
    }
  }

  public static stopListening() {
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch {
        // Ignore
      }
      this.recognition = null;
    }
  }
}
