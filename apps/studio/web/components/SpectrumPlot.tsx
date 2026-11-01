// A spectrum: its values against the coordinates of its channels along its
// spectral axis (or their numbers, if unknown; their range under it), as
// steps (each channel a bin, to its edges halfway to its neighbours), with
// a line at the channel shown. The range of its values (given, or else
// that of the values shown, with 0; see valueRange) shows at its left, in
// their unit. It shows a window of its channels (all of them by default; the
// range of its values is that of the window), which the wheel asks to zoom
// about a channel. A click picks the nearest channel, and a drag a band of
// channels (first and last, inclusive), which is shown with the channels
// out of it dimmed. Another band (e.g. a crop) can be marked by its edges.
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { significant } from './format';

const WIDTH = 800;
const HEIGHT = 130;
const PAD = { left: 8, right: 8, top: 10, bottom: 10 };

// How far the pointer moves (in screen pixels) before a click is a drag
const DRAG = 4;

export function SpectrumPlot({
  values,
  spectral,
  unit,
  valueUnit = null,
  channel,
  band = null,
  marked = null,
  window = null,
  limits = null,
  onPick,
  onBand,
  onZoom,
  children,
}: {
  values: (number | null)[];
  spectral: number[] | null;
  unit: string | null;
  // The unit of the values
  valueUnit?: string | null;
  channel: number;
  band?: [number, number] | null;
  marked?: [number, number] | null;
  // The channels shown (first and last, inclusive; null: all), and how the
  // wheel zooms about a channel (by a factor: above 1 in, below 1 out)
  window?: [number, number] | null;
  // The range of the values shown (low, high; null: that of the values)
  limits?: [number, number] | null;
  onPick: (channel: number) => void;
  onBand?: (band: [number, number]) => void;
  onZoom?: (channel: number, factor: number) => void;
  // What shows over the plot (e.g. its zoom buttons)
  children?: ReactNode;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const clip = useId();
  const press = useRef<{ channel: number; at: number } | null>(null);
  const [dragged, setDragged] = useState<[number, number] | null>(null);
  const xs = spectral ?? values.map((_, i) => i);
  const [i0, i1] = window ?? [0, values.length - 1];
  const inWindow = (i: number) => i >= i0 && i <= i1;
  const shownXs = xs.slice(i0, i1 + 1);
  // (the width of a channel; the axis runs from the outer edges of the
  // channels shown)
  const step = xs.length > 1 ? (xs[xs.length - 1] - xs[0]) / (xs.length - 1) : 1;
  const half = Math.abs(step) / 2;
  const [x0, x1] = [Math.min(...shownXs) - half, Math.max(...shownXs) + half];
  const [y0, y1] = limits !== null && limits[1] > limits[0] ? limits : valueRange(values, window);
  const X = (x: number) => PAD.left + ((x - x0) / (x1 - x0 || 1)) * (WIDTH - PAD.left - PAD.right);
  const Y = (y: number) => HEIGHT - PAD.bottom - ((y - y0) / (y1 - y0 || 1)) * (HEIGHT - PAD.top - PAD.bottom);
  // The edges of a channel, halfway to its neighbours, on the plot
  const edge = (i: number, side: -1 | 1) => Math.min(WIDTH, Math.max(0, X(xs[i] + (side * step) / 2)));
  // (each channel a step from edge to edge; a channel without a value
  // breaks the line)
  const lines: string[] = [];
  let line: string[] = [];
  for (let i = i0; i <= i1; i++) {
    const value = values[i];
    if (value === null) {
      if (line.length) lines.push(line.join(' '));
      line = [];
    } else {
      const y = Y(value).toFixed(1);
      line.push(`${edge(i, -1).toFixed(1)},${y}`, `${edge(i, 1).toFixed(1)},${y}`);
    }
  }
  if (line.length) lines.push(line.join(' '));
  const axis = spectral && unit ? unit : 'channel';
  // The channel of the window nearest a point of the screen
  const channelAt = (clientX: number) => {
    const box = svg.current!.getBoundingClientRect();
    const along = ((clientX - box.left) / box.width) * WIDTH - PAD.left;
    const x = x0 + (along / (WIDTH - PAD.left - PAD.right)) * (x1 - x0);
    let best = i0;
    for (let i = i0; i <= i1; i++) if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i;
    return best;
  };
  // (the wheel zooms, without scrolling the page: a listener that is not
  // passive, which reads the latest window through a ref)
  const zoom = useRef<(event: WheelEvent) => void>(() => {});
  zoom.current = (event) => {
    if (onZoom === undefined) return;
    event.preventDefault();
    onZoom(channelAt(event.clientX), event.deltaY < 0 ? 1.25 : 0.8);
  };
  useEffect(() => {
    const element = svg.current;
    const listener = (event: WheelEvent) => zoom.current(event);
    element?.addEventListener('wheel', listener, { passive: false });
    return () => element?.removeEventListener('wheel', listener);
  }, []);
  const shown = dragged ?? band;
  const [left, right] = shown ? [Math.min(edge(shown[0], -1), edge(shown[1], 1)), Math.max(edge(shown[0], -1), edge(shown[1], 1))] : [0, 0];
  return (
    <figure className="spectrum">
      <svg
        ref={svg}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="The spectrum"
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          press.current = { channel: channelAt(event.clientX), at: event.clientX };
        }}
        onPointerMove={(event) => {
          const start = press.current;
          if (start === null || onBand === undefined) return;
          if (dragged !== null || Math.abs(event.clientX - start.at) > DRAG) {
            const now = channelAt(event.clientX);
            setDragged([Math.min(start.channel, now), Math.max(start.channel, now)]);
          }
        }}
        onPointerUp={(event) => {
          const start = press.current;
          press.current = null;
          if (start === null) return;
          // (the band to where the pointer is let go: the last drawn may
          // lag behind it)
          const now = channelAt(event.clientX);
          if (dragged !== null) onBand?.([Math.min(start.channel, now), Math.max(start.channel, now)]);
          else onPick(now);
          setDragged(null);
        }}
        onPointerCancel={() => {
          press.current = null;
          setDragged(null);
        }}
      >
        {shown && (
          <>
            <rect className="out" x={0} y={0} width={Math.max(0, left)} height={HEIGHT} />
            <rect className="out" x={right} y={0} width={Math.max(0, WIDTH - right)} height={HEIGHT} />
          </>
        )}
        {marked &&
          [edge(marked[0], -1), edge(marked[1], 1)].map((x, i) => (
            <line key={i} className="marked" x1={x} x2={x} y1={0} y2={HEIGHT} />
          ))}
        <line className="zero" x1={PAD.left} x2={WIDTH - PAD.right} y1={Y(0)} y2={Y(0)} />
        {inWindow(channel) && <line className="now" x1={X(xs[channel])} x2={X(xs[channel])} y1={PAD.top} y2={HEIGHT - PAD.bottom} />}
        <clipPath id={clip}>
          <rect x={0} y={PAD.top} width={WIDTH} height={HEIGHT - PAD.top - PAD.bottom} />
        </clipPath>
        <g clipPath={`url(#${clip})`}>
          {lines.map((points, i) => (
            <polyline key={i} points={points} />
          ))}
        </g>
      </svg>
      <span className="spectrum-y top">
        {significant(y1, 3)} {valueUnit ?? ''}
      </span>
      <span className="spectrum-y bottom">{significant(y0, 3)}</span>
      {children}
      <figcaption>
        <span>{significant(x0, 6)}</span>
        <span>{axis}</span>
        <span>{significant(x1, 6)}</span>
      </figcaption>
    </figure>
  );
}

// The range of the values of the channels of a window of a spectrum (all of
// them by default), with 0 in it ((0, 1) if none has a value)
export function valueRange(values: (number | null)[], window: [number, number] | null = null): [number, number] {
  const [i0, i1] = window ?? [0, values.length - 1];
  const finite = values.slice(i0, i1 + 1).filter((v): v is number => v !== null);
  return finite.length ? [Math.min(0, ...finite), Math.max(...finite)] : [0, 1];
}
