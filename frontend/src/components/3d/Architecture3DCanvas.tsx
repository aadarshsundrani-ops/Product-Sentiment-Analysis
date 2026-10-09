import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../context/ThemeContext';
import { ErrorBoundary } from '../ui/ErrorBoundary';

export type PipelineMode = 'inference' | 'training';

export interface ArchNode {
  id: number;
  step: string;
  title: string;
  badge: string;
  desc: string;
  color: string;
  pos: [number, number, number];
  lightBadgeClass: string;
  darkBadgeClass: string;
}

const INFERENCE_NODES: ArchNode[] = [
  {
    id: 0,
    step: '01',
    title: 'Client WebGL UI',
    badge: 'React 18 + R3F',
    desc: 'Review input & 3D state',
    color: '#0284c7',
    pos: [-4.2, 0.7, 0],
    lightBadgeClass: 'bg-sky-100 text-sky-800 border-sky-300',
    darkBadgeClass: 'bg-sky-500/20 text-sky-300 border-sky-500/40',
  },
  {
    id: 1,
    step: '02',
    title: 'FastAPI Gateway',
    badge: 'Uvicorn REST',
    desc: 'CORS & Schema Check',
    color: '#6366f1',
    pos: [-1.4, -0.7, 0],
    lightBadgeClass: 'bg-indigo-100 text-indigo-800 border-indigo-300',
    darkBadgeClass: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',
  },
  {
    id: 2,
    step: '03',
    title: 'Inference Engine',
    badge: 'TF-IDF / MiniLM',
    desc: 'Vectorize & Classify',
    color: '#9333ea',
    pos: [1.4, 0.7, 0],
    lightBadgeClass: 'bg-purple-100 text-purple-800 border-purple-300',
    darkBadgeClass: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
  },
  {
    id: 3,
    step: '04',
    title: 'Explainability & API',
    badge: 'LIME / Attr',
    desc: 'Attributions & Stream',
    color: '#059669',
    pos: [4.2, -0.7, 0],
    lightBadgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
    darkBadgeClass: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  },
];

const TRAINING_NODES: ArchNode[] = [
  {
    id: 0,
    step: '01',
    title: 'Raw Review Data',
    badge: '10K Amazon Corpus',
    desc: 'Footwear & Accessories',
    color: '#0284c7',
    pos: [-4.2, 0.7, 0],
    lightBadgeClass: 'bg-sky-100 text-sky-800 border-sky-300',
    darkBadgeClass: 'bg-sky-500/20 text-sky-300 border-sky-500/40',
  },
  {
    id: 1,
    step: '02',
    title: 'Negation Pipeline',
    badge: 'Contraction Expander',
    desc: 'Preserves "not", bi-grams',
    color: '#6366f1',
    pos: [-1.4, -0.7, 0],
    lightBadgeClass: 'bg-indigo-100 text-indigo-800 border-indigo-300',
    darkBadgeClass: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',
  },
  {
    id: 2,
    step: '03',
    title: 'Benchmark Training',
    badge: 'TF-IDF vs MiniLM',
    desc: 'L2 Reg Logistic Model',
    color: '#9333ea',
    pos: [1.4, 0.7, 0],
    lightBadgeClass: 'bg-purple-100 text-purple-800 border-purple-300',
    darkBadgeClass: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
  },
  {
    id: 3,
    step: '04',
    title: 'Subgroup Auditing',
    badge: 'Length & Slice Audit',
    desc: 'Disparity & LIME Analysis',
    color: '#d97706',
    pos: [4.2, -0.7, 0],
    lightBadgeClass: 'bg-amber-100 text-amber-900 border-amber-300',
    darkBadgeClass: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
  },
];

// Flowing animated photon pulse along the bus conduit
const FlowingPackets: React.FC<{
  nodes: ArchNode[];
  isDark: boolean;
}> = ({ nodes, isDark }) => {
  const packetRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (!packetRef.current || !nodes || nodes.length < 2) return;
    const t = (state.clock.getElapsedTime() * 0.35) % 1;
    const totalSegments = nodes.length - 1;
    const segment = Math.min(Math.floor(t * totalSegments), totalSegments - 1);
    const segT = t * totalSegments - segment;

    const startNode = nodes[segment];
    const endNode = nodes[segment + 1];
    if (!startNode || !endNode) return;

    const start = new THREE.Vector3(...startNode.pos);
    const end = new THREE.Vector3(...endNode.pos);

    packetRef.current.position.lerpVectors(start, end, segT);
    // Smooth arching wave between staggered node heights
    packetRef.current.position.y += Math.sin(segT * Math.PI) * 0.45;
    packetRef.current.position.z = 0.28;
  });

  return (
    <mesh ref={packetRef}>
      <sphereGeometry args={[0.18, 16, 16]} />
      <meshStandardMaterial
        color={isDark ? '#38bdf8' : '#0284c7'}
        emissive={isDark ? '#38bdf8' : '#0284c7'}
        emissiveIntensity={1.4}
        roughness={0.2}
      />
    </mesh>
  );
};

// Safe Connecting Conduit Line
const ConnectingLine: React.FC<{ nodes: ArchNode[]; isDark: boolean }> = ({ nodes, isDark }) => {
  const lineObject = useMemo(() => {
    const points = nodes.map((n) => new THREE.Vector3(...n.pos));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: isDark ? '#6366f1' : '#94a3b8',
      transparent: true,
      opacity: 0.65,
    });
    return new THREE.Line(geometry, material);
  }, [nodes, isDark]);

  return <primitive object={lineObject} />;
};

interface Architecture3DCanvasProps {
  mode: PipelineMode;
  selectedStep?: number | null;
  onSelectStep?: (step: number | null) => void;
}

export const Architecture3DCanvas: React.FC<Architecture3DCanvasProps> = ({
  mode,
  selectedStep,
  onSelectStep,
}) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [internalHover, setInternalHover] = useState<number | null>(null);

  const nodes = mode === 'inference' ? INFERENCE_NODES : TRAINING_NODES;
  const activeStepId = selectedStep !== undefined && selectedStep !== null ? selectedStep : internalHover;
  const activeNode = activeStepId !== null ? nodes.find((n) => n.id === activeStepId) || null : null;

  const handleHover = (id: number | null) => {
    setInternalHover(id);
    if (onSelectStep) {
      onSelectStep(id);
    }
  };

  return (
    <ErrorBoundary>
      <div className="w-full h-[320px] md:h-[370px] relative rounded-3xl overflow-hidden dark:bg-slate-900/90 bg-white/95 border-2 dark:border-slate-800 border-slate-200 shadow-xl">
        {/* Canvas Header Overlay: Responsive & Non-overlapping */}
        <div className="absolute top-3.5 left-4 right-4 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider dark:text-slate-200 text-slate-800 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-3 py-1.5 rounded-xl border dark:border-slate-700 border-slate-300 shadow-sm pointer-events-auto">
            <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
            <span className="font-semibold">
              {mode === 'inference' ? 'REST Inference Conduit' : 'Training & Audit Pipeline'}
            </span>
          </div>

          {activeNode ? (
            <div className="text-xs font-mono px-3.5 py-1.5 rounded-xl bg-slate-900/95 text-white border border-indigo-500 shadow-lg pointer-events-auto flex items-center gap-2 animate-fade-in">
              <span className="text-slate-400">Layer {activeNode.step}:</span>
              <span className="font-bold text-white">&quot;{activeNode.title}&quot;</span>
              <span className="text-slate-500 hidden sm:inline">•</span>
              <span className="text-[11px] text-cyan-300 font-sans hidden sm:inline">{activeNode.desc}</span>
            </div>
          ) : (
            <div className="hidden sm:flex items-center gap-2 text-[11px] font-mono dark:text-slate-300 text-slate-600 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-3 py-1.5 rounded-xl border dark:border-slate-700 border-slate-300 shadow-sm pointer-events-auto">
              <span>Hover or tap any architectural node to inspect</span>
            </div>
          )}
        </div>

        {/* Legend pill bottom-left */}
        <div className="absolute bottom-3 left-4 z-10 flex items-center gap-2 pointer-events-none">
          <div className="flex items-center gap-1.5 text-[11px] font-mono dark:text-slate-300 text-slate-700 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-2.5 py-1 rounded-lg border dark:border-slate-700 border-slate-300 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-sky-500 animate-ping" />
            <span>Active Data Packet Flow</span>
          </div>
        </div>

        {/* 3D WebGL Canvas */}
        <Canvas
          camera={{ position: [0, 0, 9.4], fov: 40 }}
          gl={{ antialias: true, alpha: true, powerPreference: 'default' }}
        >
          <ambientLight intensity={isDark ? 0.85 : 1.25} />
          <pointLight position={[0, 5, 4]} intensity={1.6} color="#818cf8" />
          <directionalLight position={[6, 9, 6]} intensity={1.2} />

          {/* Ground Grid for Depth */}
          <gridHelper
            args={[14, 14, isDark ? '#334155' : '#cbd5e1', isDark ? '#1e293b' : '#f1f5f9']}
            position={[0, -2.4, 0]}
          />

          {/* Connecting Bus Conduit */}
          <ConnectingLine nodes={nodes} isDark={isDark} />

          {/* Animated Photon Pulse */}
          <FlowingPackets nodes={nodes} isDark={isDark} />

          {/* Staggered, Non-overlapping Architecture Nodes */}
          {nodes.map((node) => {
            const isSelected = activeStepId === node.id;

            return (
              <group key={node.title} position={node.pos}>
                {/* 3D Glass Box Pedestal */}
                <mesh
                  onPointerOver={(e) => {
                    e.stopPropagation();
                    document.body.style.cursor = 'pointer';
                    handleHover(node.id);
                  }}
                  onPointerOut={() => {
                    document.body.style.cursor = 'auto';
                    handleHover(null);
                  }}
                  onClick={() => handleHover(node.id)}
                >
                  <boxGeometry args={[1.75, 1.1, 0.16]} />
                  <meshStandardMaterial
                    color={isDark ? '#0f172a' : '#ffffff'}
                    emissive={node.color}
                    emissiveIntensity={isSelected ? 0.7 : isDark ? 0.25 : 0.08}
                    roughness={0.2}
                    metalness={0.2}
                    transparent
                    opacity={0.95}
                  />
                </mesh>

                {/* Glowing Outline / Wireframe Frame */}
                <mesh>
                  <boxGeometry args={[1.78, 1.14, 0.18]} />
                  <meshBasicMaterial
                    color={isSelected ? '#818cf8' : node.color}
                    wireframe
                    transparent
                    opacity={isSelected ? 0.95 : isDark ? 0.6 : 0.35}
                  />
                </mesh>

                {/* High-Contrast, Crystal Clear 3D HTML Card */}
                <Html position={[0, 0, 0.16]} center distanceFactor={9.2}>
                  <div
                    onMouseEnter={() => handleHover(node.id)}
                    onMouseLeave={() => handleHover(null)}
                    onClick={() => handleHover(node.id)}
                    className={`w-[140px] text-center select-none cursor-pointer p-2.5 rounded-xl border shadow-lg backdrop-blur-md transition-all duration-200 pointer-events-auto ${
                      isSelected
                        ? 'ring-2 ring-indigo-400 scale-105 z-30 ' +
                          (isDark
                            ? 'bg-slate-900 border-indigo-400 shadow-indigo-500/30'
                            : 'bg-white border-indigo-500 shadow-indigo-200')
                        : isDark
                        ? 'bg-slate-900/95 border-slate-700/80 text-white hover:border-slate-500 shadow-black/40'
                        : 'bg-white/95 border-slate-300 text-slate-900 hover:border-slate-400 shadow-slate-200'
                    }`}
                  >
                    {/* Step + Badge Row */}
                    <div className="flex items-center justify-center gap-1 mb-1">
                      <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-200">
                        {node.step}
                      </span>
                      <span
                        className={`text-[9px] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border truncate max-w-[95px] ${
                          isDark ? node.darkBadgeClass : node.lightBadgeClass
                        }`}
                      >
                        {node.badge}
                      </span>
                    </div>

                    {/* Title — Crystal Clear High Contrast */}
                    <div
                      className={`text-xs font-bold font-display leading-tight truncate ${
                        isDark ? 'text-white' : 'text-slate-900'
                      }`}
                    >
                      {node.title}
                    </div>

                    {/* Description — Readable Slate */}
                    <div
                      className={`text-[10px] mt-1 leading-snug line-clamp-2 ${
                        isDark ? 'text-slate-300' : 'text-slate-600'
                      }`}
                    >
                      {node.desc}
                    </div>
                  </div>
                </Html>
              </group>
            );
          })}

          <OrbitControls
            enableZoom={false}
            enablePan={false}
            maxPolarAngle={Math.PI / 2.05}
            minPolarAngle={Math.PI / 3.5}
            rotateSpeed={0.45}
          />
        </Canvas>
      </div>
    </ErrorBoundary>
  );
};

