// A spectrum: its values (their range at the left, in their unit) against
// the coordinates of its channels along its spectral axis (or their
// numbers, if unknown; their range under it), with a line at the channel
// shown; a click picks the nearest channel, and a drag a band of
// channels (first and last, inclusive), which is shown with the channels
// out of it dimmed.
import { useRef, useState } from 'react';
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
  onPick,
  onBand,
}: {
  values: (number | null)[];
  spectral: number[] | null;
  unit: string | null;
  // The unit of the values
  valueUnit?: string | null;
  channel: number;
  band?: [number, number] | null;
  onPick: (channel: number) => void;
  onBand?: (band: [number, number]) => void;
}) {
  const press = useRef<{ channel: number; at: number } | null>(null);
  const [dragged, setDragged] = useState<[number, number] | null>(null);
  const xs = spectral ?? values.map((_, i) => i);
  const finite = values.filter((v): v is number => v !== null);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = finite.length ? [Math.min(0, ...finite), Math.max(...finite)] : [0, 1];
  const X = (x: number) => PAD.left + ((x - x0) / (x1 - x0 || 1)) * (WIDTH - PAD.left - PAD.right);
  const Y = (y: number) => HEIGHT - PAD.bottom - ((y - y0) / (y1 - y0 || 1)) * (HEIGHT - PAD.top - PAD.bottom);
  // (a channel without a value breaks the line)
  const lines: string[] = [];
  let line: string[] = [];
  values.forEach((value, i) => {
    if (value === null) {
      if (line.length) lines.push(line.join(' '));
      line = [];
    } else line.push(`${X(xs[i]).toFixed(1)},${Y(value).toFixed(1)}`);
  });
  if (line.length) lines.push(line.join(' '));
  const axis = spectral && unit ? unit : 'channel';
  // The channel nearest a point of the screen
  const channelAt = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const along = ((event.clientX - box.left) / box.width) * WIDTH - PAD.left;
    const x = x0 + (along / (WIDTH - PAD.left - PAD.right)) * (x1 - x0);
    return xs.reduce((best, value, i) => (Math.abs(value - x) < Math.abs(xs[best] - x) ? i : best), 0);
  };
  const shown = dragged ?? band;
  // The edges of a channel, halfway to its neighbours
  const step = xs.length > 1 ? (xs[xs.length - 1] - xs[0]) / (xs.length - 1) : 1;
  const edge = (i: number, side: -1 | 1) => X(xs[i] + (side * step) / 2);
  const [left, right] = shown ? [Math.min(edge(shown[0], -1), edge(shown[1], 1)), Math.max(edge(shown[0], -1), edge(shown[1], 1))] : [0, 0];
  return (
    <figure className="spectrum">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="The spectrum of the pixel"
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          press.current = { channel: channelAt(event), at: event.clientX };
        }}
        onPointerMove={(event) => {
          const start = press.current;
          if (start === null || onBand === undefined) return;
          if (dragged !== null || Math.abs(event.clientX - start.at) > DRAG) {
            const now = channelAt(event);
            setDragged([Math.min(start.channel, now), Math.max(start.channel, now)]);
          }
        }}
        onPointerUp={(event) => {
          const start = press.current;
          press.current = null;
          if (start === null) return;
          // (the band to where the pointer is let go: the last drawn may
          // lag behind it)
          const now = channelAt(event);
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
        <line className="zero" x1={PAD.left} x2={WIDTH - PAD.right} y1={Y(0)} y2={Y(0)} />
        <line className="now" x1={X(xs[channel])} x2={X(xs[channel])} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
        {lines.map((points, i) => (
          <polyline key={i} points={points} />
        ))}
      </svg>
      <span className="spectrum-y top">
        {significant(y1, 3)} {valueUnit ?? ''}
      </span>
      <span className="spectrum-y bottom">{significant(y0, 3)}</span>
      <figcaption>
        <span>{significant(x0, 6)}</span>
        <span>{axis}</span>
        <span>{significant(x1, 6)}</span>
      </figcaption>
    </figure>
  );
}
