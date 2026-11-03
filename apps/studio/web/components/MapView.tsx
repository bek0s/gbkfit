// A map of values (ny by nx; the row y = 0 at the bottom, as on the sky)
// on a canvas, coloured over a range: sequentially, or diverging about
// the middle of the range (see colours.ts). A pixel without a value (NaN),
// out of the range of the values kept, or left out (e.g. by the edits of
// moment maps), has the colour of the panel. It
// shows a window of its pixels (all of them by default), which the wheel
// asks to zoom about a pixel. A click gives the pixel (x, y), which a mark
// can show; a drag gives a region (x0, x1, y0, y1, inclusive), which is
// shown with the pixels out of it dimmed, or pans the window. The pixel
// under the pointer is given as it moves (none as it leaves, or drags).
// Another region (e.g. a crop) can be outlined, and a contour drawn. A
// press or a drag with the right button stretches its colours as QFitsView
// does (see stretchAt), by where the pointer is on the map: across, the
// value at the middle of the ramp (to the right, higher), and up and down,
// how much of the range the ramp spans about it (at the top, twice the
// range; at the bottom, none of it: two colours). At the centre of the
// map they are as they were.
import { useEffect, useRef, useState } from 'react';
import { rampColour, rampStops, stretched, token, UNSTRETCHED, useColourMap, useScheme, type Stretch } from './colours';

export type Region = [number, number, number, number];

// How far the pointer moves (in screen pixels) before a click is a drag
const DRAG = 4;

// The stretch of the colours of a point of a map (x and y the fractions of
// the map to its left and above it; none outside it), as QFitsView sets
// it from the pointer (dpuser 4.3: QFitsView/QFitsScroller.cpp,
// processMouseClick, and QFitsBuffers/QFitsBaseBuffer.cpp,
// calculateColortableScaling), there in steps of 0.001 over 256 colours:
//
//   x, y   = the pointer over the width and the height of the view, x 1000
//   middle = (1000 - x) * 255 / 1000      (the colour at the middle)
//   width  = (1000 - y) * 255 * 2 / 1000  (the colours of the ramp)
//   the ramp from middle - width / 2 to middle + width / 2, the values
//   beyond it the colours of its ends (calculateColortableScaling)
//
// One difference: QFitsView draws its colour bar backwards, its high end
// at the left (QFitsWedge.cpp, resizeEvent), so that its middle, at the
// low end as the pointer goes right, is under the pointer on the bar; the
// colour bar here has its low end at the left, so the middle is x, not
// 1 - x, to keep it under the pointer (to the right, the map darker).
function stretchAt(x: number, y: number): Stretch | null {
  if (x < 0 || x >= 1 || y < 0 || y >= 1) return null;
  return { middle: x, width: 2 * (1 - y) };
}

export function MapView({
  values,
  shape,
  range,
  diverging = false,
  kept = null,
  out = null,
  window = null,
  drag = 'region',
  mark,
  region = null,
  outline = null,
  contour = null,
  stretch = UNSTRETCHED,
  onStretch,
  onPick,
  onRegion,
  onZoom,
  onPan,
  onHover,
  label,
}: {
  values: Float32Array;
  shape: number[];
  range: [number, number];
  diverging?: boolean;
  // The range of the values kept (null: all; an end null: open)
  kept?: [number | null, number | null] | null;
  // The pixels left out (1; null: none), as the values
  out?: Uint8Array | null;
  // The pixels shown (null: all), and how the wheel zooms about a pixel
  // (by a factor: above 1 in, below 1 out)
  window?: Region | null;
  mark?: [number, number];
  region?: Region | null;
  outline?: Region | null;
  // (a path in its pixels, as of contourPath)
  contour?: string | null;
  // (what a press or a drag with the right button gives)
  stretch?: Stretch;
  onStretch?: (stretch: Stretch) => void;
  onPick?: (x: number, y: number) => void;
  onRegion?: (region: Region) => void;
  onZoom?: (x: number, y: number, factor: number) => void;
  // What a drag does: give a region, or pan the window (by whole pixels)
  drag?: 'region' | 'pan';
  onPan?: (dx: number, dy: number) => void;
  onHover?: (pixel: [number, number] | null) => void;
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
  // (whether the right button is down, stretching the colours)
  const stretching = useRef(false);
  const [keepLow, keepHigh] = kept ?? [null, null];
  // (drawn again in the colours of another scheme)
  const scheme = useScheme();
  // (and in another colour map)
  const colourMap = useColourMap();
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
        const shown = Number.isFinite(value) && !out?.[y * nx + x] && (keepLow === null || value >= keepLow) && (keepHigh === null || value <= keepHigh);
        const colour = shown ? rampColour(stops, stretched((value - low) / (high - low || 1), stretch)) : empty;
        // (the row y is drawn from the bottom)
        image.data.set([...colour, 255], ((wy1 - y) * w + (x - wx0)) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, nx, range, diverging, stretch, keepLow, keepHigh, out, wx0, wx1, wy0, wy1, w, h, scheme, colourMap]);
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
  // (the stretch of the point of the map under the pointer)
  const stretchTo = (event: { clientX: number; clientY: number }) => {
    const box = canvas.current!.getBoundingClientRect();
    const next = stretchAt((event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height);
    if (next !== null) onStretch?.(next);
  };
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
        onContextMenu={(event) => onStretch && event.preventDefault()}
        onPointerDown={(event) => {
          if (event.button === 2 && onStretch) {
            event.currentTarget.setPointerCapture(event.pointerId);
            stretching.current = true;
            onHover?.(null);
            stretchTo(event);
            return;
          }
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          press.current = { pixel: pixelAt(event.clientX, event.clientY), at: [event.clientX, event.clientY] };
          panned.current = null;
        }}
        onPointerMove={(event) => {
          if (stretching.current) {
            stretchTo(event);
            return;
          }
          const start = press.current;
          if (start === null) {
            onHover?.(pixelAt(event.clientX, event.clientY));
            return;
          }
          const moved = Math.hypot(event.clientX - start.at[0], event.clientY - start.at[1]);
          // (a press that becomes a drag hides the pixel under the pointer;
          // a click keeps it, the pixel it picks)
          if (moved > DRAG) onHover?.(null);
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
          if (stretching.current) {
            stretching.current = false;
            return;
          }
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
          stretching.current = false;
          press.current = null;
          setDragged(null);
        }}
        onPointerLeave={() => press.current === null && onHover?.(null)}
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
        {outline && (
          <rect
            className="map-outline"
            x={outline[0]}
            y={ny - 1 - outline[3]}
            width={outline[1] - outline[0] + 1}
            height={outline[3] - outline[2] + 1}
          />
        )}
        {contour && <path className="map-contour" d={contour} />}
        {mark && <rect className="map-mark" x={mark[0]} y={ny - 1 - mark[1]} width={1} height={1} />}
      </svg>
    </div>
  );
}
