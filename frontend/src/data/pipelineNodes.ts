export interface NodeInfo {
  id: string;
  step: string;
  name: string;
  shortName: string;
  sub: string;
  targetId: string;
  position: [number, number, number];
  labelYOffset: number; // Staggered alternating offset to eliminate overlap
  color: string;
}

export const PIPELINE_NODES: NodeInfo[] = [
  {
    id: 'dataset',
    step: '01',
    name: '01. Dataset',
    shortName: 'Dataset Ingestion',
    sub: '10K Amazon Reviews',
    targetId: 'dataset-upload',
    position: [-4.4, -0.2, 0],
    labelYOffset: -0.85,
    color: '#38bdf8', // Cyan
  },
  {
    id: 'preprocessing',
    step: '02',
    name: '02. Preprocess',
    shortName: 'Negation Preprocessing',
    sub: 'Contraction & Bi-grams',
    targetId: 'code-runner',
    position: [-2.2, 0.35, 0],
    labelYOffset: 0.85,
    color: '#818cf8', // Indigo
  },
  {
    id: 'training',
    step: '03',
    name: '03. Model Train',
    shortName: 'Dual Benchmarking',
    sub: 'TF-IDF vs MiniLM',
    targetId: 'code-runner',
    position: [0, -0.25, 0],
    labelYOffset: -0.85,
    color: '#c084fc', // Purple
  },
  {
    id: 'prediction',
    step: '04',
    name: '04. Predict API',
    shortName: 'FastAPI Gateway',
    sub: 'Sub-25ms REST Inference',
    targetId: 'prediction-interface',
    position: [2.2, 0.35, 0],
    labelYOffset: 0.85,
    color: '#f472b6', // Pink
  },
  {
    id: 'results',
    step: '05',
    name: '05. Audit & Results',
    shortName: 'Fairness Audit',
    sub: '94.8% F1 & Disparity',
    targetId: 'model-results',
    position: [4.4, -0.2, 0],
    labelYOffset: -0.85,
    color: '#34d399', // Emerald
  },
];
