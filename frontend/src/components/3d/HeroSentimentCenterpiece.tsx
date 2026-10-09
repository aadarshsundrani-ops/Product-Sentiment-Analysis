import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../context/ThemeContext';

export type SentimentState = 'Positive' | 'Negative' | 'Neutral';

interface HeroCenterpieceProps {
  sentiment?: SentimentState;
  interactive?: boolean;
}

// Procedural Sentiment Avatar: A floating high-tech speech bubble/sphere with expressive reactive face features
const SentimentAvatar: React.FC<{ sentiment: SentimentState; isDark: boolean }> = ({
  sentiment,
  isDark,
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const leftEyeRef = useRef<THREE.Mesh>(null);
  const rightEyeRef = useRef<THREE.Mesh>(null);
  const mouthRef = useRef<THREE.Mesh>(null);
  const ringRef1 = useRef<THREE.Mesh>(null);
  const ringRef2 = useRef<THREE.Mesh>(null);

  // Target colors based on sentiment
  const sentimentColors = useMemo(() => {
    switch (sentiment) {
      case 'Positive':
        return {
          core: isDark ? '#10b981' : '#059669',
          ring: isDark ? '#06b6d4' : '#0891b2',
          glow: '#34d399',
          mouthCurve: 1.0, // Smiling up
        };
      case 'Negative':
        return {
          core: isDark ? '#f43f5e' : '#e11d48',
          ring: isDark ? '#8b5cf6' : '#6d28d9',
          glow: '#fb7185',
          mouthCurve: -1.0, // Frowning down
        };
      case 'Neutral':
      default:
        return {
          core: isDark ? '#f59e0b' : '#d97706',
          ring: isDark ? '#6366f1' : '#4f46e5',
          glow: '#fbbf24',
          mouthCurve: 0.0, // Straight calm line
        };
    }
  }, [sentiment, isDark]);

  // Animated procedural deformation & smooth lerp
  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();

    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(time * 1.5) * 0.12;
      groupRef.current.rotation.y += delta * 0.3;
    }

    if (ringRef1.current) {
      ringRef1.current.rotation.x = time * 0.6;
      ringRef1.current.rotation.y = time * 0.4;
    }

    if (ringRef2.current) {
      ringRef2.current.rotation.x = -time * 0.4;
      ringRef2.current.rotation.z = time * 0.5;
    }

    // Mouth curvature morph based on sentiment (Math.PI = smiling up ∪, 0 = frowning down ∩)
    if (mouthRef.current) {
      const targetCurve = sentimentColors.mouthCurve;
      mouthRef.current.scale.y = THREE.MathUtils.lerp(
        mouthRef.current.scale.y,
        sentiment === 'Neutral' ? 0.3 : 1.0,
        delta * 4
      );
      // Positive (targetCurve > 0) -> Math.PI for happy smile ∪
      // Negative (targetCurve < 0) -> 0 for sad/crying frown ∩
      const targetRotationZ = targetCurve > 0 ? Math.PI : 0;
      mouthRef.current.rotation.z = THREE.MathUtils.lerp(
        mouthRef.current.rotation.z,
        targetRotationZ,
        delta * 6
      );
      // Smoothly adjust Y position so smile dip or frown arch sits naturally
      const targetY = sentiment === 'Positive' ? -0.32 : sentiment === 'Negative' ? -0.18 : -0.24;
      mouthRef.current.position.y = THREE.MathUtils.lerp(
        mouthRef.current.position.y,
        targetY,
        delta * 6
      );
    }
  });

  return (
    <group ref={groupRef}>
      {/* Central Holographic Sphere (Body) */}
      <mesh>
        <sphereGeometry args={[1.3, 64, 64]} />
        <meshPhysicalMaterial
          color={sentimentColors.core}
          emissive={sentimentColors.core}
          emissiveIntensity={isDark ? 0.45 : 0.15}
          roughness={0.2}
          metalness={0.1}
          transmission={isDark ? 0.65 : 0.4}
          thickness={1.2}
          transparent
          opacity={0.88}
          wireframe={false}
        />
      </mesh>

      {/* Inner Neural Core Sphere */}
      <mesh>
        <sphereGeometry args={[0.75, 32, 32]} />
        <meshStandardMaterial
          color={sentimentColors.glow}
          emissive={sentimentColors.glow}
          emissiveIntensity={isDark ? 0.8 : 0.4}
          roughness={0.1}
          wireframe={true}
        />
      </mesh>

      {/* Orbiting Orbital Gyro Ring 1 */}
      <mesh ref={ringRef1}>
        <torusGeometry args={[1.8, 0.03, 16, 100]} />
        <meshStandardMaterial
          color={sentimentColors.ring}
          emissive={sentimentColors.ring}
          emissiveIntensity={isDark ? 0.9 : 0.3}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Orbiting Orbital Gyro Ring 2 */}
      <mesh ref={ringRef2}>
        <torusGeometry args={[2.1, 0.02, 16, 100]} />
        <meshStandardMaterial
          color={sentimentColors.glow}
          emissive={sentimentColors.glow}
          emissiveIntensity={isDark ? 0.7 : 0.25}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>

      {/* Face Features: Left Eye */}
      <mesh ref={leftEyeRef} position={[-0.42, 0.25, 1.18]}>
        <sphereGeometry args={[0.13, 32, 32]} />
        <meshStandardMaterial
          color={isDark ? '#ffffff' : '#0f172a'}
          emissive={isDark ? '#ffffff' : '#475569'}
          emissiveIntensity={isDark ? 0.9 : 0.2}
        />
      </mesh>

      {/* Face Features: Right Eye */}
      <mesh ref={rightEyeRef} position={[0.42, 0.25, 1.18]}>
        <sphereGeometry args={[0.13, 32, 32]} />
        <meshStandardMaterial
          color={isDark ? '#ffffff' : '#0f172a'}
          emissive={isDark ? '#ffffff' : '#475569'}
          emissiveIntensity={isDark ? 0.9 : 0.2}
        />
      </mesh>

      {/* Face Features: Expressive Mouth Arc */}
      <mesh ref={mouthRef} position={[0, -0.24, 1.2]}>
        {sentiment === 'Neutral' ? (
          <boxGeometry args={[0.55, 0.08, 0.05]} />
        ) : (
          <torusGeometry args={[0.32, 0.065, 16, 32, Math.PI]} />
        )}
        <meshStandardMaterial
          color={isDark ? '#ffffff' : '#0f172a'}
          emissive={isDark ? '#ffffff' : '#334155'}
          emissiveIntensity={isDark ? 0.9 : 0.2}
        />
      </mesh>

      {/* Crying Teardrop for Negative Sentiment */}
      {sentiment === 'Negative' && (
        <mesh position={[-0.42, 0.04, 1.23]}>
          <sphereGeometry args={[0.07, 16, 16]} />
          <meshStandardMaterial
            color="#38bdf8"
            emissive="#38bdf8"
            emissiveIntensity={0.9}
            roughness={0.1}
          />
        </mesh>
      )}

      {/* Rosy Cheeks for Positive Sentiment */}
      {sentiment === 'Positive' && (
        <>
          <mesh position={[-0.56, -0.05, 1.15]}>
            <sphereGeometry args={[0.1, 16, 16]} />
            <meshStandardMaterial
              color="#fb7185"
              emissive="#fb7185"
              emissiveIntensity={0.6}
              transparent
              opacity={0.7}
            />
          </mesh>
          <mesh position={[0.56, -0.05, 1.15]}>
            <sphereGeometry args={[0.1, 16, 16]} />
            <meshStandardMaterial
              color="#fb7185"
              emissive="#fb7185"
              emissiveIntensity={0.6}
              transparent
              opacity={0.7}
            />
          </mesh>
        </>
      )}
    </group>
  );
};

// Ambient Particle Dust System
const ParticleField: React.FC<{ isDark: boolean; sentiment: SentimentState }> = ({
  isDark,
  sentiment,
}) => {
  const pointsRef = useRef<THREE.Points>(null);
  const count = 180;

  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    const baseColor = new THREE.Color(
      sentiment === 'Positive' ? '#10b981' : sentiment === 'Negative' ? '#f43f5e' : '#6366f1'
    );
    const secondaryColor = new THREE.Color(isDark ? '#06b6d4' : '#38bdf8');

    for (let i = 0; i < count; i++) {
      // Distribute in a spherical cloud
      const r = 2.4 + Math.random() * 3.5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);

      const mixed = baseColor.clone().lerp(secondaryColor, Math.random());
      col[i * 3] = mixed.r;
      col[i * 3 + 1] = mixed.g;
      col[i * 3 + 2] = mixed.b;
    }
    return [pos, col];
  }, [count, sentiment, isDark]);

  useFrame((state, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.08;
      pointsRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
    }
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
        size={isDark ? 0.065 : 0.05}
        vertexColors
        transparent
        opacity={isDark ? 0.75 : 0.55}
        blending={isDark ? THREE.AdditiveBlending : THREE.NormalBlending}
      />
    </points>
  );
};

export const HeroSentimentCenterpiece: React.FC<HeroCenterpieceProps> = ({
  sentiment = 'Positive',
  interactive = true,
}) => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <div className="w-full h-full relative cursor-grab active:cursor-grabbing">
      <Canvas
        camera={{ position: [0, 0, 5.2], fov: 45 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      >
        <ambientLight intensity={isDark ? 0.7 : 1.2} />
        <directionalLight
          position={[5, 6, 5]}
          intensity={isDark ? 1.4 : 1.8}
          color={isDark ? '#e0e7ff' : '#ffffff'}
        />
        <pointLight
          position={[-4, -3, -2]}
          intensity={isDark ? 1.2 : 0.6}
          color={sentiment === 'Positive' ? '#10b981' : sentiment === 'Negative' ? '#f43f5e' : '#f59e0b'}
        />

        <Float speed={2.5} rotationIntensity={0.3} floatIntensity={0.6}>
          <SentimentAvatar sentiment={sentiment} isDark={isDark} />
        </Float>

        <ParticleField isDark={isDark} sentiment={sentiment} />

        {interactive && (
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            maxPolarAngle={Math.PI / 1.7}
            minPolarAngle={Math.PI / 2.3}
            maxAzimuthAngle={Math.PI / 4}
            minAzimuthAngle={-Math.PI / 4}
            rotateSpeed={0.5}
            dampingFactor={0.05}
          />
        )}
      </Canvas>

      {/* Floating Sentiment Status Pill in Canvas overlay */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-none">
        <div
          className={`px-4 py-1.5 rounded-full text-xs font-semibold backdrop-blur-md border transition-all duration-300 flex items-center gap-2 shadow-lg ${
            sentiment === 'Positive'
              ? 'dark:bg-emerald-500/10 bg-emerald-100/90 dark:text-emerald-400 text-emerald-800 dark:border-emerald-500/30 border-emerald-300'
              : sentiment === 'Negative'
              ? 'dark:bg-rose-500/10 bg-rose-100/90 dark:text-rose-400 text-rose-800 dark:border-rose-500/30 border-rose-300'
              : 'dark:bg-amber-500/10 bg-amber-100/90 dark:text-amber-400 text-amber-800 dark:border-amber-500/30 border-amber-300'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full animate-pulse ${
              sentiment === 'Positive'
                ? 'bg-emerald-500 shadow-[0_0_8px_#34d399]'
                : sentiment === 'Negative'
                ? 'bg-rose-500 shadow-[0_0_8px_#fb7185]'
                : 'bg-amber-500 shadow-[0_0_8px_#fbbf24]'
            }`}
          />
          Interactive 3D Identity: <span className="font-bold">{sentiment} State</span>
        </div>
      </div>
    </div>
  );
};
