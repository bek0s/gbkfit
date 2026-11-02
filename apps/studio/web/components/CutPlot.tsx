// A cut through a map (e.g. a PSF): the values of a row or a column of its
// pixels, as steps, each pixel as wide as on the map, so that a cut beside
// the map (under it, or at its side, vertical) lines up with it as it
// resizes. It shows a window of the pixels (first and last, inclusive), as
// the map does, over a range of values, with a line at a pixel (e.g. that
// of the other cut). A vertical cut runs up, as the rows of a map do.
import { useId } from 'react';
import { significant } from './format';

// (the plot is drawn on a square stretched to its box; its lines keep
// their width)
const SIZE = 1000;

export function CutPlot({
  values,
  window: [i0, i1],
  limits: [low, high],
  at,
  vertical = false,
  label,
}: {
  values: (number | null)[];
  window: [number, number];
  limits: [number, number];
  // The pixel marked (none: none)
  at?: number;
  vertical?: boolean;
  label: string;
}) {
  const clip = useId();
  const n = i1 - i0 + 1;
  // (where a pixel and a value are on the plot, along it and across it,
  // from 0 to 1)
  const along = (i: number) => (i - i0) / n;
  const across = (value: number) => Math.min(1, Math.max(0, (value - low) / (high - low || 1)));
  const point = (u: number, v: number) =>
    vertical ? `${(v * SIZE).toFixed(1)},${((1 - u) * SIZE).toFixed(1)}` : `${(u * SIZE).toFixed(1)},${((1 - v) * SIZE).toFixed(1)}`;
  // (each pixel a step from edge to edge; a pixel without a value breaks
  // the line)
  const lines: string[] = [];
  let line: string[] = [];
  for (let i = i0; i <= i1; i++) {
    const value = values[i];
    if (value === null || value === undefined || !Number.isFinite(value)) {
      if (line.length) lines.push(line.join(' '));
      line = [];
      continue;
    }
    line.push(point(along(i), across(value)), point(along(i + 1), across(value)));
  }
  if (line.length) lines.push(line.join(' '));
  const mark = at !== undefined && at >= i0 && at <= i1 ? along(at) + 0.5 / n : null;
  return (
    <figure className={vertical ? 'cut vertical' : 'cut'}>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} preserveAspectRatio="none" role="img" aria-label={label}>
        {mark !== null &&
          (vertical ? (
            <line className="now" x1={0} x2={SIZE} y1={(1 - mark) * SIZE} y2={(1 - mark) * SIZE} />
          ) : (
            <line className="now" x1={mark * SIZE} x2={mark * SIZE} y1={0} y2={SIZE} />
          ))}
        <clipPath id={clip}>
          <rect x={0} y={0} width={SIZE} height={SIZE} />
        </clipPath>
        <g clipPath={`url(#${clip})`}>
          {lines.map((points, i) => (
            <polyline key={i} points={points} />
          ))}
        </g>
      </svg>
      <span className="cut-high">{significant(high, 3)}</span>
    </figure>
  );
}
