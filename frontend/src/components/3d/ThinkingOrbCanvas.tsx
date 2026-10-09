import React from 'react';
import { useTheme } from '../../context/ThemeContext';

export const ThinkingOrbCanvas: React.FC = () => {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <div className="w-24 h-24 relative mx-auto flex items-center justify-center">
      {/* Outer Gyro Ring 1 */}
      <div
        className="absolute inset-0 rounded-full border-2 border-dashed border-sky-400/80 animate-[spin_3s_linear_infinite]"
        style={{ transformStyle: 'preserve-3d' }}
      />
      {/* Mid Gyro Ring 2 */}
      <div
        className="absolute inset-2 rounded-full border-2 border-purple-400/80 animate-[spin_4s_linear_infinite_reverse]"
        style={{ transform: 'rotateX(60deg)' }}
      />
      {/* Inner Gyro Ring 3 */}
      <div
        className="absolute inset-4 rounded-full border-2 border-pink-400/80 animate-[spin_2s_linear_infinite]"
        style={{ transform: 'rotateY(60deg)' }}
      />
      {/* Central Pulsing Neural Core */}
      <div
        className={`w-8 h-8 rounded-full animate-ping opacity-75 ${
          isDark ? 'bg-primary shadow-[0_0_20px_#6366f1]' : 'bg-primary shadow-[0_0_15px_#4f46e5]'
        }`}
      />
      <div
        className={`w-6 h-6 rounded-full absolute flex items-center justify-center font-mono font-bold text-[9px] text-white ${
          isDark ? 'bg-primary' : 'bg-primary'
        }`}
      >
        AI
      </div>
    </div>
  );
};
