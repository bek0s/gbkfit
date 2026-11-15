// Small marks of the sheet of a galaxy, drawn in the colour of their text:
// a value per ring (two rings), and the directions of the speeds and the
// dispersions of traits (see the directions of the library's traits).
import type { ReactNode } from 'react';

// A value per ring
export function RingMark() {
  return (
    <svg className="gm-rmark" viewBox="0 0 10 10" aria-hidden="true">
      <circle cx="5" cy="5" r="4" />
      <circle cx="5" cy="5" r="1.5" />
    </svg>
  );
}

// The marks of the directions, by name: around the centre (tan), away from
// it in the plane (rad), up (ver), towards the observer (los), away from
// the centre in space (out), away from the plane on both sides (away), and
// every way (iso)
const DIRECTIONS: Record<string, ReactNode> = {
  tan: (
    <>
      <path d="M10 6a4 4 0 1 1-1.17-2.83" />
      <path d="M6.3 2.9 8.83 3.17 8.6.6" />
    </>
  ),
  rad: (
    <>
      <circle className="gm-mark-dot" cx="2.6" cy="6" r="1.3" />
      <path d="M5.2 6h5.6M8.4 3.6 10.8 6 8.4 8.4" />
    </>
  ),
  ver: <path d="M6 10.6V1.6M3.6 4 6 1.6 8.4 4" />,
  los: (
    <>
      <circle cx="6" cy="6" r="4.4" />
      <circle className="gm-mark-dot" cx="6" cy="6" r="1.2" />
    </>
  ),
  out: (
    <>
      <circle className="gm-mark-dot" cx="2.8" cy="9.2" r="1.3" />
      <path d="M4.8 7.2 10.2 1.8M6.6 1.8h3.6v3.6" />
    </>
  ),
  away: <path d="M1.4 6h9.2M6 4.4V1.4M4.2 3.2 6 1.4 7.8 3.2M6 7.6v3M4.2 8.8 6 10.6 7.8 8.8" />,
  iso: <path d="M6 1.4v9.2M2 3.7l8 4.6M2 8.3l8-4.6" />,
};

// The mark of a direction (none of one it does not know)
export function DirectionMark({ direction }: { direction: string }) {
  const mark = DIRECTIONS[direction];
  if (!mark) return null;
  return (
    <svg className="gm-dmark" viewBox="0 0 12 12" aria-hidden="true">
      {mark}
    </svg>
  );
}
