// A map of values (ny by nx; the row y = 0 at the bottom, as on the sky)
// on a canvas, coloured over a range: sequentially, or diverging about
// the middle of the range (see colours.ts). A pixel without a value (NaN),
// or out of the range of the values kept, has the colour of the panel. It
// shows a window of its pixels (all of them by default), which the wheel
// asks to zoom about a pixel. A click gives the pixel (x, y), which a mark
// can show; a drag gives a region (x0, x1, y0, y1, inclusive), which is
// shown with the pixels out of it dimmed, or pans the window.
import { useEffect, useRef, useState } from 'react';
import { rampColour, rampStops, token } from './colours';

export type Region = [number, number, number, number];

// How far the pointer moves (in screen pixels) before a click is a drag
const DRAG = 4;

export function MapView({
  values,
  shape,
  range,
  diverging = false,
  kept = null,
  window = null,
  drag = 'region',
  mark,
  region = null,
  onPick,
  onRegion,
  onZoom,
  onPan,
  label,
}: {
  values: Float32Array;
  shape: number[];
  range: [number, number];
  diverging?: boolean;
  // The range of the values kept (null: all; an end null: open)
  kept?: [number | null, number | null] | null;
  // The pixels shown (null: all), and how the wheel zooms about a pixel
  // (by a factor: above 1 in, below 1 out)
  window?: Region | null;
  mark?: [number, number];
  region?: Region | null;
  onPick?: (x: number, y: number) => void;
  onRegion?: (region: Region) => void;
  onZoom?: (x: number, y: number, factor: number) => void;
  // What a drag does: give a region, or pan the window (by whole pixels)
  drag?: 'region' | 'pan';
  onPan?: (dx: number, dy: number) => void;
  label: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [ny, nx] = shape;
  const [wx0, wx1, wy0, wy1] = window ?? [0, nx - 1, 0, ny - 1];
  const [w, h] = [wx1 - wx0 + 1, wy1 - wy0 + 1];
  // Where a press started (its pixel and its point on the screen), and the
  // region dragged from it so far
  const press = useRef<{ pixel: [number, number]; at: [number, number] } | null>(null);
  const [dragged, setDragged] = useState<Region | null>(null);
  // (the pan not yet made whole pixels, and where the pointer was)
  const panned = useRef<{ at: [number, number]; rest: [number, number] } | null>(null);
  const [keepLow, keepHigh] = kept ?? [null, null];
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;
    element.width = w;
    element.height = h;
    const stops = rampStops(diverging);
    const empty = token('--panel2');
    const [low, high] = range;
    const image = context.createImageData(w, h);
    for (let y = wy0; y <= wy1; y++) {
      for (let x = wx0; x <= wx1; x++) {
        const value = values[y * nx + x];
        const shown = Number.isFinite(value) && (keepLow === null || value >= keepLow) && (keepHigh === null || value <= keepHigh);
        const colour = shown ? rampColour(stops, (value - low) / (high - low || 1)) : empty;
        // (the row y is drawn from the bottom)
        image.data.set([...colour, 255], ((wy1 - y) * w + (x - wx0)) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, nx, range, diverging, keepLow, keepHigh, wx0, wx1, wy0, wy1, w, h]);
  // The pixel under a point of the screen
  const pixelAt = (clientX: number, clientY: number): [number, number] => {
    const box = canvas.current!.getBoundingClientRect();
    const x = wx0 + Math.floor(((clientX - box.left) / box.width) * w);
    const y = wy1 - Math.floor(((clientY - box.top) / box.height) * h);
    return [Math.min(wx1, Math.max(wx0, x)), Math.min(wy1, Math.max(wy0, y))];
  };
  // (the wheel zooms, without scrolling the page: a listener that is not
  // passive, which reads the latest window through a ref)
  const zoom = useRef<(event: WheelEvent) => void>(() => {});
  zoom.current = (event) => {
    if (onZoom === undefined) return;
    event.preventDefault();
    onZoom(...pixelAt(event.clientX, event.clientY), event.deltaY < 0 ? 1.25 : 0.8);
  };
  useEffect(() => {
    const element = canvas.current;
    const listener = (event: WheelEvent) => zoom.current(event);
    element?.addEventListener('wheel', listener, { passive: false });
    return () => element?.removeEventListener('wheel', listener);
  }, []);
  const between = ([ax, ay]: [number, number], [bx, by]: [number, number]): Region => [
    Math.min(ax, bx),
    Math.max(ax, bx),
    Math.min(ay, by),
    Math.max(ay, by),
  ];
  const shown = dragged ?? region;
  return (
    <div className="map" data-drag={drag}>
      <canvas
        ref={canvas}
        aria-label={label}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          press.current = { pixel: pixelAt(event.clientX, event.clientY), at: [event.clientX, event.clientY] };
          panned.current = null;
        }}
        onPointerMove={(event) => {
          const start = press.current;
          if (start === null) return;
          const moved = Math.hypot(event.clientX - start.at[0], event.clientY - start.at[1]);
          if (drag === 'pan') {
            if (onPan === undefined || (panned.current === null && moved <= DRAG)) return;
            // (the pixels moved since the last move: the map follows the
            // pointer, so the window moves the other way)
            const box = event.currentTarget.getBoundingClientRect();
            const last = panned.current ?? { at: start.at, rest: [0, 0] };
            const dx = last.rest[0] - ((event.clientX - last.at[0]) / box.width) * w;
            const dy = last.rest[1] + ((event.clientY - last.at[1]) / box.height) * h;
            const [mx, my] = [Math.trunc(dx), Math.trunc(dy)];
            panned.current = { at: [event.clientX, event.clientY], rest: [dx - mx, dy - my] };
            if (mx !== 0 || my !== 0) onPan(mx, my);
            return;
          }
          if (onRegion === undefined) return;
          if (dragged !== null || moved > DRAG) setDragged(between(start.pixel, pixelAt(event.clientX, event.clientY)));
        }}
        onPointerUp={(event) => {
          const start = press.current;
          press.current = null;
          if (start === null) return;
          if (panned.current !== null) {
            panned.current = null;
            return;
          }
          // (the region to where the pointer is let go: the last drawn may
          // lag behind it)
          const now = pixelAt(event.clientX, event.clientY);
          if (dragged !== null) onRegion?.(between(start.pixel, now));
          else onPick?.(...now);
          setDragged(null);
        }}
        onPointerCancel={() => {
          press.current = null;
          setDragged(null);
        }}
      />
      {/* (in the pixels of the whole map; the window is its view) */}
      <svg viewBox={`${wx0} ${ny - 1 - wy1} ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
        {shown && (
          <>
            {/* (the pixels out of the region, dimmed: the map less the region) */}
            <path
              className="map-out"
              fillRule="evenodd"
              d={`M0 0H${nx}V${ny}H0Z M${shown[0]} ${ny - 1 - shown[3]}H${shown[1] + 1}V${ny - shown[2]}H${shown[0]}Z`}
            />
            <rect
              className="map-region"
              x={shown[0]}
              y={ny - 1 - shown[3]}
              width={shown[1] - shown[0] + 1}
              height={shown[3] - shown[2] + 1}
            />
          </>
        )}
        {mark && <rect className="map-mark" x={mark[0]} y={ny - 1 - mark[1]} width={1} height={1} />}
      </svg>
    </div>
  );
}
