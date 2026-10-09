// 100% Safari & WebKit Safe DOM Particle Celebration
// Completely avoids creating 2D <canvas> overlays over WebGL contexts,
// which causes Metal compositor context loss and black screens in Safari.

export interface SafeConfettiOptions {
  particleCount?: number;
  spread?: number;
  origin?: { x?: number; y?: number };
}

export function fireSafeConfetti(options?: SafeConfettiOptions) {
  if (typeof document === 'undefined') return;

  try {
    const count = Math.min(options?.particleCount || 35, 45);
    const startX = (options?.origin?.x !== undefined ? options.origin.x : 0.5) * window.innerWidth;
    const startY = (options?.origin?.y !== undefined ? options.origin.y : 0.65) * window.innerHeight;

    const container = document.createElement('div');
    container.setAttribute('aria-hidden', 'true');
    container.style.cssText = `
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 9999;
      overflow: hidden;
    `;

    const colors = [
      '#6366f1', // Indigo
      '#06b6d4', // Cyan
      '#10b981', // Emerald
      '#f59e0b', // Amber
      '#ec4899', // Pink
      '#8b5cf6', // Purple
      '#38bdf8', // Sky
    ];

    const elements: HTMLDivElement[] = [];

    for (let i = 0; i < count; i++) {
      const el = document.createElement('div');
      const color = colors[i % colors.length];
      const width = Math.random() * 8 + 6;
      const height = Math.random() * 10 + 6;
      const angle = (Math.random() - 0.5) * 110 - 90; // Upward fan spread
      const velocity = Math.random() * 260 + 180;
      const rad = (angle * Math.PI) / 180;
      const destX = Math.cos(rad) * velocity + (Math.random() - 0.5) * 60;
      const destY = Math.sin(rad) * velocity + 100;
      const rotation = Math.random() * 720 - 360;

      el.style.cssText = `
        position: absolute;
        left: ${startX}px;
        top: ${startY}px;
        width: ${width}px;
        height: ${height}px;
        background-color: ${color};
        border-radius: ${Math.random() > 0.4 ? '2px' : '50%'};
        opacity: 1;
        transform: translate3d(0, 0, 0) rotate(0deg);
        transition: transform 1.2s cubic-bezier(0.12, 0.9, 0.32, 1), opacity 1.2s ease-in;
        will-change: transform, opacity;
      `;

      container.appendChild(el);
      elements.push(el);

      // Trigger animation on next frame
      requestAnimationFrame(() => {
        el.style.transform = `translate3d(${destX}px, ${destY}px, 0) rotate(${rotation}deg) scale(0.5)`;
        el.style.opacity = '0';
      });
    }

    document.body.appendChild(container);

    setTimeout(() => {
      if (container.parentNode) {
        container.parentNode.removeChild(container);
      }
    }, 1300);
  } catch (err) {
    console.warn('DOM particle animation skipped:', err);
  }
}
