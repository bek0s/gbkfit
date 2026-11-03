// A cut through a map (e.g. a PSF): the values of a row or a column of its
// pixels, as steps, each pixel as wide as on the map, so that a cut beside
// the map (under it, or at its side, vertical) lines up with it as it
// resizes. It shows a window of the pixels (first and last, inclusive), as
// the map does, over a range of values: lines of several kinds (e.g.
// through the peak, and through a pixel), and marks at pixels (e.g. where
// the other cut of a pixel crosses it), and its width at a level (e.g. at
// half the peak of the cut through it: its FWHM). A vertical cut runs up,
// as the rows of a map do.
import { useId } from 'react';
import type { Width } from './contour';

// (the plot is drawn on a square stretched to its box; its lines keep
// their width)
const SIZE = 1000;

// (the room left above the top of its range, a fraction of the plot, so
// that a peak does not touch its edge)
const HEADROOM = 0.1;

// A line of a cut, of a kind (e.g. through the peak, or through a pixel
// clicked or under the pointer), which its colour and weight say
export type CutLine = { values: (number | null)[]; kind: string };

export function CutPlot({
  lines,
  window: [i0, i1],
  limits: [low, high],
  marks = [],
  width = null,
  vertical = false,
  label,
}: {
  // (drawn in their order, the last on top)
  lines: CutLine[];
  window: [number, number];
  limits: [number, number];
  // The pixels marked, each of a kind (e.g. where the other cut of a pixel
  // crosses this one)
  marks?: { at: number; kind: string }[];
  width?: Width | null;
  vertical?: boolean;
  label: string;
}) {
  const clip = useId();
  const n = i1 - i0 + 1;
  // (where a pixel and a value are on the plot, along it and across it,
  // from 0 to 1)
  const along = (i: number) => (i - i0) / n;
  const across = (value: number) => (1 - HEADROOM) * Math.min(1, Math.max(0, (value - low) / (high - low || 1)));
  const point = (u: number, v: number) =>
    vertical ? `${(v * SIZE).toFixed(1)},${((1 - u) * SIZE).toFixed(1)}` : `${(u * SIZE).toFixed(1)},${((1 - v) * SIZE).toFixed(1)}`;
  // (each pixel a step from edge to edge; a pixel without a value breaks
  // the line)
  const runs = (values: (number | null)[]) => {
    const out: string[] = [];
    let run: string[] = [];
    for (let i = i0; i <= i1; i++) {
      const value = values[i];
      if (value === null || value === undefined || !Number.isFinite(value)) {
        if (run.length) out.push(run.join(' '));
        run = [];
        continue;
      }
      run.push(point(along(i), across(value)), point(along(i + 1), across(value)));
    }
    if (run.length) out.push(run.join(' '));
    return out;
  };
  return (
    <figure className={vertical ? 'cut vertical' : 'cut'}>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} preserveAspectRatio="none" role="img" aria-label={label}>
        {marks
          .filter(({ at }) => at >= i0 && at <= i1)
          .map(({ at, kind }) => {
            const u = along(at) + 0.5 / n;
            return vertical ? (
              <line key={kind} className={`mark ${kind}`} x1={0} x2={SIZE} y1={(1 - u) * SIZE} y2={(1 - u) * SIZE} />
            ) : (
              <line key={kind} className={`mark ${kind}`} x1={u * SIZE} x2={u * SIZE} y1={0} y2={SIZE} />
            );
          })}
        {width !== null &&
          (() => {
            const [a, b, v] = [along(width.from) + 0.5 / n, along(width.to) + 0.5 / n, across(width.level)];
            return vertical ? (
              <line className="width" x1={v * SIZE} x2={v * SIZE} y1={(1 - a) * SIZE} y2={(1 - b) * SIZE} />
            ) : (
              <line className="width" x1={a * SIZE} x2={b * SIZE} y1={(1 - v) * SIZE} y2={(1 - v) * SIZE} />
            );
          })()}
        <clipPath id={clip}>
          <rect x={0} y={0} width={SIZE} height={SIZE} />
        </clipPath>
        <g clipPath={`url(#${clip})`}>
          {lines.map(({ values, kind }) => runs(values).map((points, i) => <polyline key={`${kind}-${i}`} className={kind} points={points} />))}
        </g>
      </svg>
    </figure>
  );
}
