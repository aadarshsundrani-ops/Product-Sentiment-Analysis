import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useTheme } from '../../context/ThemeContext';
import { ErrorBoundary } from '../ui/ErrorBoundary';

interface DataIngestionCanvasProps {
  isIngesting?: boolean;
}

const IngestionParticles: React.FC<{ isIngesting: boolean; isDark: boolean }> = ({
  isIngesting,
  isDark,
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const count = 180;

  // Stable parametric definitions for all particles
  const { positions, angles, speeds, baseRadii, colors } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const ang = new Float32Array(count);
    const spd = new Float32Array(count);
    const rad = new Float32Array(count);
    const cols = new Float32Array(count * 3);

    const c1 = new THREE.Color(isDark ? '#38bdf8' : '#0284c7');
    const c2 = new THREE.Color(isDark ? '#818cf8' : '#4f46e5');

    for (let i = 0; i < count; i++) {
      ang[i] = Math.random() * Math.PI * 2;
      spd[i] = 0.8 + Math.random() * 0.8;
      rad[i] = 0.4 + Math.random() * 1.1;

      const y = (Math.random() - 0.5) * 3.8;
      const funnelScale = 0.4 + Math.abs(y) * 0.4;
      const r = rad[i] * funnelScale;

      pos[i * 3] = Math.cos(ang[i]) * r;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = Math.sin(ang[i]) * r;

      const mixed = c1.clone().lerp(c2, Math.random());
      cols[i * 3] = mixed.r;
      cols[i * 3 + 1] = mixed.g;
      cols[i * 3 + 2] = mixed.b;
    }

    return { positions: pos, angles: ang, speeds: spd, baseRadii: rad, colors: cols };
  }, [count, isDark]);

  useFrame((_, delta) => {
    if (!pointsRef.current) return;
    const posAttr = pointsRef.current.geometry.attributes.position;
    const array = posAttr.array as Float32Array;
    const speedMultiplier = isIngesting ? 2.8 : 0.9;

    for (let i = 0; i < count; i++) {
      // Stream downward
      array[i * 3 + 1] -= delta * speeds[i] * speedMultiplier;

      // Wrap cleanly from bottom back to top
      if (array[i * 3 + 1] < -1.8) {
        array[i * 3 + 1] = 1.9;
        angles[i] = Math.random() * Math.PI * 2;
      }

      // Funnel constriction
      const currentY = array[i * 3 + 1];
      const funnelScale = 0.35 + Math.abs(currentY) * 0.45;
      angles[i] += delta * 1.2;

      const r = baseRadii[i] * funnelScale;
      array[i * 3] = Math.cos(angles[i]) * r;
      array[i * 3 + 2] = Math.sin(angles[i]) * r;
    }

    posAttr.needsUpdate = true;
    pointsRef.current.rotation.y += delta * 0.3;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={isIngesting ? 0.075 : 0.05}
        vertexColors
        transparent
        opacity={isDark ? 0.85 : 0.65}
        blending={isDark ? THREE.AdditiveBlending : THREE.NormalBlending}
      />
    </points>
  );
};

// Funnel Target Model Core
const FunnelCore: React.FC<{ isIngesting: boolean; isDark: boolean }> = ({ isIngesting, isDark }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * (isIngesting ? 2.2 : 0.7);
      meshRef.current.rotation.z += delta * (isIngesting ? 1.4 : 0.4);
    }
    if (ringRef.current) {
      ringRef.current.rotation.x += delta * 1.1;
    }
  });

  return (
    <group position={[0, -0.2, 0]}>
      {/* Central Funnel Core Node */}
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[0.55, 1]} />
        <meshStandardMaterial
          color={isDark ? '#6366f1' : '#4f46e5'}
          emissive={isDark ? '#38bdf8' : '#818cf8'}
          emissiveIntensity={isIngesting ? 0.85 : isDark ? 0.4 : 0.2}
          wireframe={true}
        />
      </mesh>

      {/* Orbiting Funnel Ring */}
      <mesh ref={ringRef}>
        <torusGeometry args={[0.85, 0.02, 16, 48]} />
        <meshBasicMaterial
          color={isDark ? '#38bdf8' : '#0284c7'}
          transparent
          opacity={isIngesting ? 0.9 : 0.4}
        />
      </mesh>
    </group>
  );
};

export const DataIngestionCanvas: React.FC<DataIngestionCanvasProps> = ({ isIngesting = false }) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <ErrorBoundary>
      <div className="w-full h-full min-h-[180px] relative pointer-events-none">
        <Canvas
          camera={{ position: [0, 0, 4.2], fov: 45 }}
          gl={{ antialias: true, alpha: true, powerPreference: 'default' }}
        >
          <ambientLight intensity={isDark ? 0.6 : 1.0} />
          <pointLight position={[0, 2, 2]} intensity={1.5} color="#38bdf8" />
          <IngestionParticles isIngesting={isIngesting} isDark={isDark} />
          <FunnelCore isIngesting={isIngesting} isDark={isDark} />
        </Canvas>
      </div>
    </ErrorBoundary>
  );
};
