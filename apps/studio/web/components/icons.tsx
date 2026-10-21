// The icons of the page (strokes in the current colour).
import type { ReactNode } from 'react';

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
      {children}
    </svg>
  );
}

export const FilesIcon = () => (
  <Icon>
    <path d="M4 5h6l2 2h8v12H4z" />
  </Icon>
);

export const JobsIcon = () => (
  <Icon>
    <path d="M5 6h14M5 12h14M5 18h9" />
  </Icon>
);

export const PreviewIcon = () => (
  <Icon>
    <circle cx="12" cy="12" r="3" />
    <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" />
  </Icon>
);

export const Logo = () => (
  <svg className="logo-mark" viewBox="0 0 24 24" aria-hidden="true">
    <ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-30 12 12)" />
    <circle cx="12" cy="12" r="2.2" />
  </svg>
);
