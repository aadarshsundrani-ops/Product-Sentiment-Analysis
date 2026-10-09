import React, { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../context/ThemeContext';
import { TOP_FEATURE_WEIGHTS, FeatureWeight } from '../../data/sampleData';
import { ErrorBoundary } from '../ui/ErrorBoundary';

interface Bar3DProps {
  item: FeatureWeight;
  position: [number, number, number];
  isDark: boolean;
  isHovered: boolean;
  maxAbsWeight?: number;
  onHover: (item: FeatureWeight | null) => void;
}

const Bar3D: React.FC<Bar3DProps> = ({ item, position, isDark, isHovered, maxAbsWeight = 6.4, onHover }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const baseRef = useRef<THREE.Mesh>(null);

  const safeMax = Math.max(0.5, maxAbsWeight);
  const height = Math.max(0.45, Math.min(3.2, (Math.abs(item.weight) / safeMax) * 2.6));
  const isPositive = item.sentiment === 'Positive' || item.weight > 0;
  const barColor = isPositive
    ? isDark ? '#10b981' : '#059669'
    : isDark ? '#f43f5e' : '#e11d48';

  const yPos = position[1] + height / 2;

  useFrame((_, delta) => {
    if (meshRef.current) {
      const targetScaleX = isHovered ? 1.25 : 1.0;
      const targetScaleZ = isHovered ? 1.25 : 1.0;
      meshRef.current.scale.x = THREE.MathUtils.lerp(meshRef.current.scale.x, targetScaleX, delta * 9);
      meshRef.current.scale.z = THREE.MathUtils.lerp(meshRef.current.scale.z, targetScaleZ, delta * 9);
    }
  });

  return (
    <group position={[position[0], 0, position[2]]}>
      {/* Ground Pedestal / Base Plate */}
      <mesh ref={baseRef} position={[0, 0.04, 0]}>
        <cylinderGeometry args={[0.38, 0.42, 0.08, 16]} />
        <meshStandardMaterial
          color={isDark ? '#1e293b' : '#e2e8f0'}
          emissive={barColor}
          emissiveIntensity={isHovered ? 0.6 : 0.15}
          roughness={0.4}
        />
      </mesh>

      {/* Extruded 3D Bar */}
      <mesh
        ref={meshRef}
        position={[0, yPos, 0]}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
          onHover(item);
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto';
          onHover(null);
        }}
      >
        <boxGeometry args={[0.45, height, 0.45]} />
        <meshStandardMaterial
          color={barColor}
          emissive={barColor}
          emissiveIntensity={isHovered ? 0.85 : isDark ? 0.35 : 0.2}
          roughness={0.25}
          metalness={0.4}
        />
      </mesh>

      {/* Non-overlapping 3D Word Tag positioned cleanly above pedestal */}
      <Html position={[0, 0.12, 0.55]} center distanceFactor={8.5}>
        <div
          onClick={() => onHover(item)}
          className={`cursor-pointer transition-all duration-200 text-center select-none px-2.5 py-1 rounded-lg border shadow-md font-mono whitespace-nowrap ${
            isHovered
              ? 'scale-110 ring-2 ring-indigo-400 bg-slate-900 text-white font-bold border-indigo-400 z-30'
              : isDark
              ? 'bg-slate-900/90 text-slate-200 border-slate-700/80 hover:border-slate-500'
              : 'bg-white/95 text-slate-800 border-slate-300 hover:border-indigo-400'
          }`}
          style={{ minWidth: '70px' }}
        >
          <div className="text-[11px] font-bold leading-tight flex items-center justify-center gap-1">
            <span
              className={`w-1.5 h-1.5 rounded-full ${isPositive ? 'bg-emerald-400' : 'bg-rose-400'}`}
            />
            <span>{item.word}</span>
          </div>
          <div
            className={`text-[9px] font-semibold mt-0.5 ${
              isPositive ? 'text-emerald-500 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'
            }`}
          >
            {item.weight > 0 ? `+${item.weight.toFixed(2)}` : item.weight.toFixed(2)}
          </div>
        </div>
      </Html>
    </group>
  );
};

interface BarChart3DCanvasProps {
  features?: FeatureWeight[];
  selectedWord?: string | null;
  scopeTitle?: string;
  scopeLabel?: string;
  onSelectWord?: (word: string | null) => void;
}

export const BarChart3DCanvas: React.FC<BarChart3DCanvasProps> = ({
  features,
  selectedWord,
  scopeTitle,
  scopeLabel,
  onSelectWord,
}) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [internalHovered, setInternalHovered] = useState<FeatureWeight | null>(null);

  // Use provided features if available; fallback to TOP_FEATURE_WEIGHTS
  const sourceFeatures = features && features.length > 0 ? features : TOP_FEATURE_WEIGHTS;

  const posFeatures = sourceFeatures
    .filter((f) => f.sentiment === 'Positive' || f.weight > 0)
    .slice(0, 5);
  const negFeatures = sourceFeatures
    .filter((f) => f.sentiment === 'Negative' || f.weight < 0)
    .slice(0, 5);

  const maxAbsWeight = Math.max(
    0.5,
    ...sourceFeatures.map((f) => Math.abs(f.weight || 0))
  );

  const activeHoveredItem =
    internalHovered ||
    (selectedWord ? sourceFeatures.find((f) => f.word === selectedWord) || null : null);

  const handleHover = (item: FeatureWeight | null) => {
    setInternalHovered(item);
    if (onSelectWord) {
      onSelectWord(item ? item.word : null);
    }
  };

  return (
    <ErrorBoundary>
      <div className="w-full h-[320px] md:h-[370px] relative rounded-3xl overflow-hidden dark:bg-slate-900/90 bg-white/95 border-2 dark:border-slate-800 border-slate-200 shadow-xl">
        {/* Overlay header: Responsive & Non-overlapping */}
        <div className="absolute top-3.5 left-4 right-4 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider dark:text-slate-200 text-slate-800 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-3 py-1.5 rounded-xl border dark:border-slate-700 border-slate-300 shadow-sm pointer-events-auto">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
            <span className="font-semibold">{scopeLabel || 'Interactive 3D Token Grid'}</span>
            {scopeTitle && (
              <span className="text-[10px] text-indigo-500 dark:text-indigo-400 font-bold border-l pl-2 dark:border-slate-700 border-slate-300 truncate max-w-[200px]">
                {scopeTitle}
              </span>
            )}
          </div>

          {activeHoveredItem ? (
            <div className="text-xs font-mono px-3.5 py-1.5 rounded-xl bg-slate-900/95 text-white border border-indigo-500 shadow-lg pointer-events-auto flex items-center gap-2 animate-fade-in">
              <span className="text-slate-400">Word:</span>
              <span className="font-bold text-white">&quot;{activeHoveredItem.word}&quot;</span>
              <span className="text-slate-500">|</span>
              <span
                className={`font-bold ${
                  activeHoveredItem.weight > 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {activeHoveredItem.weight > 0
                  ? `+${activeHoveredItem.weight.toFixed(2)}`
                  : activeHoveredItem.weight.toFixed(2)}
              </span>
              {activeHoveredItem.context && (
                <>
                  <span className="text-slate-500 hidden sm:inline">•</span>
                  <span className="text-[11px] text-indigo-300 font-sans hidden sm:inline">
                    {activeHoveredItem.context}
                  </span>
                </>
              )}
            </div>
          ) : (
            <div className="hidden sm:flex items-center gap-2 text-[11px] font-mono dark:text-slate-300 text-slate-600 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-3 py-1.5 rounded-xl border dark:border-slate-700 border-slate-300 shadow-sm pointer-events-auto">
              <span>Drag to rotate • Hover bars to inspect</span>
            </div>
          )}
        </div>

        {/* Legend pills bottom-left */}
        <div className="absolute bottom-3 left-4 z-10 flex items-center gap-2 pointer-events-none">
          <div className="flex items-center gap-1.5 text-[11px] font-mono dark:text-slate-300 text-slate-700 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-2.5 py-1 rounded-lg border dark:border-slate-700 border-slate-300 shadow-sm">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
            <span>Positive Drivers ({posFeatures.length})</span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] font-mono dark:text-slate-300 text-slate-700 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-2.5 py-1 rounded-lg border dark:border-slate-700 border-slate-300 shadow-sm">
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-500" />
            <span>Negative Drivers ({negFeatures.length})</span>
          </div>
        </div>

        {/* 3D WebGL Canvas */}
        <Canvas
          camera={{ position: [0, 4.8, 8.2], fov: 42 }}
          gl={{ antialias: true, alpha: true, powerPreference: 'default' }}
        >
          <ambientLight intensity={isDark ? 0.8 : 1.2} />
          <directionalLight position={[6, 9, 6]} intensity={1.3} />
          <pointLight position={[-4, 4, 3]} intensity={0.9} color="#818cf8" />

          {/* Ground Grid */}
          <gridHelper
            args={[12, 12, isDark ? '#334155' : '#cbd5e1', isDark ? '#1e293b' : '#f1f5f9']}
            position={[0, 0, 0]}
          />

          {/* Back Row: Top Positive Features (Z = -1.1) */}
          <group position={[0, 0, -1.1]}>
            {posFeatures.map((item, idx) => {
              const count = Math.max(1, posFeatures.length);
              const xPos = (idx - (count - 1) / 2) * 1.35;
              return (
                <Bar3D
                  key={`${item.word}-${idx}`}
                  item={item}
                  position={[xPos, 0, 0]}
                  isDark={isDark}
                  isHovered={activeHoveredItem?.word === item.word}
                  maxAbsWeight={maxAbsWeight}
                  onHover={handleHover}
                />
              );
            })}
          </group>

          {/* Front Row: Top Negative Features (Z = 1.1) */}
          <group position={[0, 0, 1.1]}>
            {negFeatures.map((item, idx) => {
              const count = Math.max(1, negFeatures.length);
              const xPos = (idx - (count - 1) / 2) * 1.35;
              return (
                <Bar3D
                  key={`${item.word}-${idx}`}
                  item={item}
                  position={[xPos, 0, 0]}
                  isDark={isDark}
                  isHovered={activeHoveredItem?.word === item.word}
                  maxAbsWeight={maxAbsWeight}
                  onHover={handleHover}
                />
              );
            })}
          </group>

          <OrbitControls
            enableZoom={false}
            enablePan={false}
            maxPolarAngle={Math.PI / 2.05}
            minPolarAngle={Math.PI / 4.2}
            rotateSpeed={0.45}
          />
        </Canvas>
      </div>
    </ErrorBoundary>
  );
};

