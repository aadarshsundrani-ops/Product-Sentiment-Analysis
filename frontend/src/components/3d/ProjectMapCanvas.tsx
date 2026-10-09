import React, { useRef, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../context/ThemeContext';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { PIPELINE_NODES, NodeInfo } from '../../data/pipelineNodes';

const PipelineNode: React.FC<{
  node: NodeInfo;
  isHovered: boolean;
  onHover: (id: string | null) => void;
  onClick: (targetId: string) => void;
  isDark: boolean;
}> = ({ node, isHovered, onHover, onClick, isDark }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const outerRingRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (ringRef.current) {
      ringRef.current.rotation.z += delta * 1.6;
    }
    if (outerRingRef.current) {
      outerRingRef.current.rotation.x += delta * 1.2;
      outerRingRef.current.rotation.y += delta * 0.9;
    }
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.7;
      const targetScale = isHovered ? 1.3 : 1.0;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), delta * 8);
    }
  });

  return (
    <group position={node.position}>
      {/* Interactive Core 3D Geometry */}
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
          onHover(node.id);
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto';
          onHover(null);
        }}
        onClick={(e) => {
          e.stopPropagation();
          onClick(node.targetId);
        }}
      >
        <octahedronGeometry args={[0.42, 0]} />
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={isHovered ? 1.0 : isDark ? 0.45 : 0.25}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Orbiting Halo Ring 1 */}
      <mesh ref={ringRef}>
        <torusGeometry args={[0.62, 0.02, 16, 48]} />
        <meshBasicMaterial
          color={node.color}
          transparent
          opacity={isHovered ? 0.95 : 0.45}
        />
      </mesh>

      {/* Orbiting Halo Ring 2 (Gimbal) */}
      <mesh ref={outerRingRef}>
        <torusGeometry args={[0.72, 0.015, 12, 36]} />
        <meshBasicMaterial
          color={isDark ? '#ffffff' : '#334155'}
          transparent
          opacity={isHovered ? 0.7 : 0.2}
        />
      </mesh>

      {/* Staggered Vertical Label — Alternates top/bottom to completely prevent overlapping */}
      <Html position={[0, node.labelYOffset, 0]} center distanceFactor={10}>
        <div
          onClick={(e) => {
            e.stopPropagation();
            onClick(node.targetId);
          }}
          className={`cursor-pointer transition-all duration-300 text-center select-none px-3 py-1.5 rounded-xl border-2 shadow-lg backdrop-blur-xl ${
            isHovered
              ? 'scale-110 border-indigo-500 bg-indigo-600 text-white shadow-indigo-500/40 ring-2 ring-indigo-400/50'
              : isDark
              ? 'border-slate-700 bg-slate-900/90 text-slate-100 hover:border-slate-500 hover:bg-slate-800'
              : 'border-slate-300 bg-white/95 text-slate-900 hover:border-indigo-400 hover:bg-slate-50'
          }`}
          style={{ minWidth: '110px', maxWidth: '140px' }}
        >
          <div className="flex items-center justify-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: node.color }}
            />
            <span className="text-xs font-bold font-mono tracking-tight whitespace-nowrap">
              {node.name}
            </span>
          </div>
          <div
            className={`text-[10px] font-medium leading-tight mt-0.5 whitespace-nowrap ${
              isHovered ? 'text-indigo-100' : isDark ? 'text-slate-400' : 'text-slate-600'
            }`}
          >
            {node.sub}
          </div>
        </div>
      </Html>
    </group>
  );
};

// Safe Connecting Laser Conduit Mesh
const ConnectingLaser: React.FC<{ isDark: boolean }> = ({ isDark }) => {
  const lineMesh = useMemo(() => {
    const points = PIPELINE_NODES.map((n) => new THREE.Vector3(...n.position));
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const lineMaterial = new THREE.LineBasicMaterial({
      color: isDark ? '#6366f1' : '#4f46e5',
      transparent: true,
      opacity: 0.65,
      linewidth: 2,
    });
    return new THREE.Line(lineGeometry, lineMaterial);
  }, [isDark]);

  return <primitive object={lineMesh} />;
};

// Animated Data Packet traveling through the pipeline
const TravelingDataPacket: React.FC<{ speed?: number }> = ({ speed = 0.28 }) => {
  const packetRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!packetRef.current) return;
    const t = (clock.getElapsedTime() * speed) % 1;
    const totalSegments = PIPELINE_NODES.length - 1;
    const scaledT = t * totalSegments;
    const index = Math.floor(scaledT);
    const segT = scaledT - index;

    const p1 = PIPELINE_NODES[index].position;
    const p2 = PIPELINE_NODES[Math.min(index + 1, totalSegments)].position;

    packetRef.current.position.x = p1[0] + (p2[0] - p1[0]) * segT;
    packetRef.current.position.y = p1[1] + (p2[1] - p1[1]) * segT;
    packetRef.current.position.z = p1[2] + (p2[2] - p1[2]) * segT;
  });

  return (
    <mesh ref={packetRef}>
      <sphereGeometry args={[0.09, 16, 16]} />
      <meshBasicMaterial color="#38bdf8" />
    </mesh>
  );
};

interface ProjectMapCanvasProps {
  hoveredNode?: string | null;
  onHoverNode?: (id: string | null) => void;
  onSelectNode?: (targetId: string) => void;
}

export const ProjectMapCanvas: React.FC<ProjectMapCanvasProps> = ({
  hoveredNode: externalHoveredNode,
  onHoverNode: externalOnHover,
  onSelectNode: externalOnSelect,
}) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [internalHovered, setInternalHovered] = useState<string | null>(null);

  const activeHoveredNode = externalHoveredNode !== undefined ? externalHoveredNode : internalHovered;
  const setHoveredNode = externalOnHover || setInternalHovered;

  const handleNodeClick = (targetId: string) => {
    if (externalOnSelect) {
      externalOnSelect(targetId);
    } else {
      try {
        const el = document.getElementById(targetId);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth' });
        }
      } catch (e) {
        console.warn('Scroll navigation:', e);
      }
    }
  };

  return (
    <ErrorBoundary>
      <div className="w-full h-[270px] sm:h-[300px] relative rounded-3xl overflow-hidden dark:bg-slate-900/90 bg-white/95 border-2 dark:border-slate-800 border-slate-200 shadow-xl">
        {/* Top Header Status Bar */}
        <div className="absolute top-3.5 left-4 right-4 z-10 flex items-center justify-between pointer-events-none">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider dark:text-slate-200 text-slate-800 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-3 py-1.5 rounded-xl border dark:border-slate-700 border-slate-300 shadow-sm pointer-events-auto">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
            <span className="font-semibold">Interactive Pipeline Map</span>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 text-[11px] font-mono dark:text-slate-300 text-slate-600 bg-white/90 dark:bg-slate-800/90 backdrop-blur-md px-3 py-1.5 rounded-xl border dark:border-slate-700 border-slate-300 shadow-sm pointer-events-auto">
            <span>Click any node to jump</span>
            <span className="text-indigo-500 font-bold">↓</span>
          </div>
        </div>

        {/* 3D WebGL Canvas */}
        <Canvas
          camera={{ position: [0, 0, 8.8], fov: 42 }}
          gl={{ antialias: true, alpha: true, powerPreference: 'default' }}
        >
          <ambientLight intensity={isDark ? 0.9 : 1.3} />
          <pointLight position={[0, 4, 3]} intensity={1.8} color="#818cf8" />

          <Float speed={1.2} rotationIntensity={0.08} floatIntensity={0.25}>
            <ConnectingLaser isDark={isDark} />
            <TravelingDataPacket speed={0.3} />
            {PIPELINE_NODES.map((node) => (
              <PipelineNode
                key={node.id}
                node={node}
                isHovered={activeHoveredNode === node.id}
                onHover={setHoveredNode}
                onClick={handleNodeClick}
                isDark={isDark}
              />
            ))}
          </Float>
        </Canvas>
      </div>
    </ErrorBoundary>
  );
};
