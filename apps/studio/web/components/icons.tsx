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

// The settings: a gear
export const SettingsIcon = () => (
  <Icon>
    <path d="M18.88 10.24L21.39 10.58L21.39 13.42L18.88 13.76A7.1 7.1 0 0 1 18.11 15.62L19.65 17.64L17.64 19.65L15.62 18.11A7.1 7.1 0 0 1 13.76 18.88L13.42 21.39L10.58 21.39L10.24 18.88A7.1 7.1 0 0 1 8.38 18.11L6.36 19.65L4.35 17.64L5.89 15.62A7.1 7.1 0 0 1 5.12 13.76L2.61 13.42L2.61 10.58L5.12 10.24A7.1 7.1 0 0 1 5.89 8.38L4.35 6.36L6.36 4.35L8.38 5.89A7.1 7.1 0 0 1 10.24 5.12L10.58 2.61L13.42 2.61L13.76 5.12A7.1 7.1 0 0 1 15.62 5.89L17.64 4.35L19.65 6.36L18.11 8.38A7.1 7.1 0 0 1 18.88 10.24z" />
    <circle cx="12" cy="12" r="3" />
  </Icon>
);

// The select tool: a pointer at the corner of a dashed box
// A check: a choice made (e.g. in a menu)
export const CheckIcon = () => (
  <Icon>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </Icon>
);

export const SelectIcon = () => (
  <Icon>
    <path d="M3 3h3M9 3h3M3 3v3M3 9v3M3 12h3" />
    <path d="M10 10l10 4-4.2 1.8L14 20z" />
  </Icon>
);

// The pan tool: arrows to the four sides
export const PanIcon = () => (
  <Icon>
    <path d="M12 3v18M3 12h18" />
    <path d="M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3" />
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

// Linked (e.g. the x and y of a size, kept one): two links of a chain
export const LinkIcon = () => (
  <Icon>
    <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" />
    <path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" />
  </Icon>
);

// Not linked: the two links of a chain apart
export const UnlinkIcon = () => (
  <Icon>
    <path d="M15.5 12.5l3.2-3.2a4 4 0 0 0-5.7-5.7L9.8 6.8" />
    <path d="M8.5 11.5l-3.2 3.2a4 4 0 0 0 5.7 5.7l3.2-3.2" />
  </Icon>
);

export const Logo = () => (
  <svg className="logo-mark" viewBox="0 0 24 24" aria-hidden="true">
    <ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-30 12 12)" />
    <circle cx="12" cy="12" r="2.2" />
  </svg>
);

// More (the menu of a block): three dots, in a row
export const MoreIcon = () => (
  <Icon>
    <circle cx="5" cy="12" r="1.3" fill="currentColor" />
    <circle cx="12" cy="12" r="1.3" fill="currentColor" />
    <circle cx="19" cy="12" r="1.3" fill="currentColor" />
  </Icon>
);
