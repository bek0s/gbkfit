// The icons of the page (strokes in the current colour).
import type { ReactNode } from 'react';

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
      {children}
    </svg>
  );
}

// Data: a cube
export const DataIcon = () => (
  <Icon>
    <path d="M4 8l8-4 8 4-8 4z" />
    <path d="M4 8v8l8 4 8-4V8" />
    <path d="M12 12v8" />
  </Icon>
);

// The model: a galaxy seen inclined
export const ModelIcon = () => (
  <Icon>
    <ellipse cx="12" cy="12" rx="9" ry="4" transform="rotate(-25 12 12)" />
    <circle cx="12" cy="12" r="1.6" />
  </Icon>
);

// The fit: a curve through a point
export const FitIcon = () => (
  <Icon>
    <path d="M3.5 19c3-9 6-12 9-12s4 6 8 6" />
    <circle cx="12.5" cy="7" r="1.4" />
  </Icon>
);

export const Logo = () => (
  <svg className="logo-mark" viewBox="0 0 24 24" aria-hidden="true">
    <ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-30 12 12)" />
    <circle cx="12" cy="12" r="2.2" />
  </svg>
);
