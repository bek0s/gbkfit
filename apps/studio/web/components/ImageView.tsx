// An image of a 2-D array, its first row at the bottom (as in FITS),
// coloured over the range of its finite values: sequential, or diverging
// about 0 (e.g. residuals). NaN is transparent.
import { useEffect, useMemo, useRef } from 'react';

interface Props {
  data: Float32Array;
  // (rows, columns)
  shape: number[];
  diverging?: boolean;
  // The range of the colours (by default, that of the values)
  limits?: [number, number];
}

export function ImageView({ data, shape, diverging = false, limits }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [rows, columns] = shape;
  const own = useMemo(() => range(data, diverging), [data, diverging]);
  const [low, high] = limits ?? own;

  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const stops = (diverging ? DIVERGING : SEQUENTIAL).map(color);
    const image = context.createImageData(columns, rows);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const value = data[row * columns + column];
        const pixel = 4 * ((rows - 1 - row) * columns + column);
        if (!Number.isFinite(value)) continue;
        const t = high > low ? (value - low) / (high - low) : 0.5;
        const [r, g, b] = interpolate(stops, t);
        image.data.set([r, g, b, 255], pixel);
      }
    }
    context.putImageData(image, 0, 0);
  }, [data, rows, columns, low, high, diverging]);

  return (
    <figure className="image">
      <canvas ref={canvas} width={columns} height={rows} />
      <figcaption>
        <span>{format(low)}</span>
        <span className={diverging ? 'scale diverging' : 'scale'} />
        <span>{format(high)}</span>
      </figcaption>
    </figure>
  );
}

// The colour maps, by the tokens of their stops (see styles.css)
const SEQUENTIAL = ['--seq0', '--seq1', '--seq2', '--seq3'];
const DIVERGING = ['--map-neg2', '--map-neg', '--map-mid', '--map-pos', '--map-pos2'];

export function range(data: Float32Array, diverging: boolean): [number, number] {
  let low = Infinity;
  let high = -Infinity;
  for (const value of data) {
    if (!Number.isFinite(value)) continue;
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  if (low > high) return [0, 0];
  if (!diverging) return [low, high];
  const extent = Math.max(Math.abs(low), Math.abs(high));
  return [-extent, extent];
}

// The colour of a token, in any form of CSS (e.g. #fff, as minified), as
// the browser reads it: an opaque colour reads back as #rrggbb.
function color(token: string): [number, number, number] {
  const probe = document.createElement('canvas').getContext('2d')!;
  probe.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
  const value = parseInt(String(probe.fillStyle).slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function interpolate(stops: [number, number, number][], t: number): number[] {
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  return stops[i].map((a, k) => Math.round(a + f * (stops[i + 1][k] - a)));
}

export function format(value: number): string {
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  return magnitude >= 1e4 || magnitude < 1e-2 ? value.toExponential(2) : value.toPrecision(4);
}
