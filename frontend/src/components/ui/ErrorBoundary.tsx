import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.warn('3D Canvas encountered a non-fatal WebGL issue, recovering gracefully:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center rounded-2xl glass-panel border border-surface-border">
          <div className="w-10 h-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center font-bold mb-2">
            3D
          </div>
          <p className="text-xs text-foreground-muted">WebGL Interactive View (Optimized Mode)</p>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="mt-3 px-3 py-1 text-xs rounded-lg bg-surface border border-surface-border hover:border-primary text-foreground"
          >
            Reload 3D Scene
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
