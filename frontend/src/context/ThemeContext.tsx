import React, { createContext, useContext, useEffect, useState } from 'react';

export type Theme = 'dark' | 'light';

export interface ThemeColors3D {
  background: string;
  particlePrimary: string;
  particleSecondary: string;
  ambientLight: string;
  directionalLight: string;
  gridColor: string;
  fogColor: string;
}

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  colors3D: ThemeColors3D;
}

const darkColors3D: ThemeColors3D = {
  background: '#070a13',
  particlePrimary: '#6366f1',
  particleSecondary: '#06b6d4',
  ambientLight: '#1e1b4b',
  directionalLight: '#818cf8',
  gridColor: '#1e293b',
  fogColor: '#070a13',
};

const lightColors3D: ThemeColors3D = {
  background: '#f8fafc',
  particlePrimary: '#4f46e5',
  particleSecondary: '#0284c7',
  ambientLight: '#ffffff',
  directionalLight: '#38bdf8',
  gridColor: '#cbd5e1',
  fogColor: '#f8fafc',
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('sentiment_showcase_theme');
    if (saved === 'dark' || saved === 'light') return saved;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'light') {
      root.classList.add('light');
      root.classList.remove('dark');
    } else {
      root.classList.add('dark');
      root.classList.remove('light');
    }
    localStorage.setItem('sentiment_showcase_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  const colors3D = theme === 'dark' ? darkColors3D : lightColors3D;

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, colors3D }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
