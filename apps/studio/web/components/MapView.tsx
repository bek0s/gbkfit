// A map of values (ny by nx; the row y = 0 at the bottom, as on the sky)
// on a canvas, coloured over a range: sequentially, or diverging about
// the middle of the range (see colours.ts). A pixel without a value (NaN)
// has the colour of the panel. A click gives the pixel (x, y), which a
// mark can show.
import { useEffect, useRef } from 'react';
import { rampColour, rampStops, token } from './colours';

export function MapView({
  values,
  shape,
  range,
  diverging = false,
  mark,
  onPick,
  label,
}: {
  values: Float32Array;
  shape: number[];
  range: [number, number];
  diverging?: boolean;
  mark?: [number, number];
  onPick?: (x: number, y: number) => void;
  label: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [ny, nx] = shape;
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;
    element.width = nx;
    element.height = ny;
    const stops = rampStops(diverging);
    const empty = token('--panel2');
    const [low, high] = range;
    const image = context.createImageData(nx, ny);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const value = values[y * nx + x];
        const colour = Number.isFinite(value) ? rampColour(stops, (value - low) / (high - low || 1)) : empty;
        // (the row y is drawn from the bottom)
        image.data.set([...colour, 255], ((ny - 1 - y) * nx + x) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, nx, ny, range, diverging]);
  return (
    <div className="map" style={{ aspectRatio: `${nx} / ${ny}` }}>
      <canvas
        ref={canvas}
        aria-label={label}
        onClick={(event) => {
          if (onPick === undefined) return;
          const box = event.currentTarget.getBoundingClientRect();
          const x = Math.floor(((event.clientX - box.left) / box.width) * nx);
          const y = ny - 1 - Math.floor(((event.clientY - box.top) / box.height) * ny);
          onPick(Math.min(nx - 1, Math.max(0, x)), Math.min(ny - 1, Math.max(0, y)));
        }}
      />
      {mark && (
        <svg viewBox={`0 0 ${nx} ${ny}`} preserveAspectRatio="none" aria-hidden="true">
          <rect x={mark[0]} y={ny - 1 - mark[1]} width={1} height={1} />
        </svg>
      )}
    </div>
  );
}
