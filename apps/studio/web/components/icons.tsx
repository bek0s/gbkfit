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

// The colour schemes: of the system (a screen), light (the sun), dark
// (the moon)
export const SystemIcon = () => (
  <Icon>
    <rect x="3" y="4.5" width="18" height="12" rx="1.5" />
    <path d="M9 20h6M12 16.5V20" />
  </Icon>
);

export const LightIcon = () => (
  <Icon>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" />
  </Icon>
);

export const DarkIcon = () => (
  <Icon>
    <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
  </Icon>
);

// Zoom to the selection: a magnifier over a dashed box
export const ZoomSelectionIcon = () => (
  <Icon>
    <path d="M3 3h3M9 3h3M3 3v3M3 9v3M3 12h3M12 3v3" />
    <circle cx="14" cy="14" r="4.5" />
    <path d="M17.3 17.3L21 21" />
  </Icon>
);

// The whole (map, spectrum): the four corners of a frame
export const ZoomWholeIcon = () => (
  <Icon>
    <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
  </Icon>
);

// Clear: a cross
export const ClearIcon = () => (
  <Icon>
    <path d="M6 6l12 12M18 6L6 18" />
  </Icon>
);

// The panels at the side: a window with a column at its right
export const PanelsIcon = () => (
  <Icon>
    <rect x="3" y="4.5" width="18" height="15" rx="1.5" />
    <path d="M15 4.5v15" />
  </Icon>
);

export const Logo = () => (
  <svg className="logo-mark" viewBox="0 0 24 24" aria-hidden="true">
    <ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-30 12 12)" />
    <circle cx="12" cy="12" r="2.2" />
  </svg>
);
