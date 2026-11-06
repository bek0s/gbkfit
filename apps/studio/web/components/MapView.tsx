// A map of values (ny by nx; the row y = 0 at the bottom, as on the sky)
// on a canvas, coloured over a range: sequentially, or diverging about
// the middle of the range (see colours.ts). A pixel without a value (NaN),
// out of the range of the values kept, or left out (e.g. by the edits of
// moment maps), has the colour of the panel. It
// shows a window of its pixels (all of them by default), which the wheel
// asks to zoom about a pixel, and a double click to show whole. Its
// gestures: a click gives the pixel (x, y), which a mark can show; a drag
// pans the window, or, as a viewer asks (see MapPanel: its select tool),
// draws a region (x0, x1, y0, y1, inclusive), shown with the pixels out of
// it dimmed; a drag with the middle button, or with Space held, pans. The pixel under the
// pointer is given as it moves (none as it leaves, or drags).
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

// A box of pixels of a map of one size in those of another (e.g. a render
// made again at another size): the same part of it
export function rescale(box: Region, [fx, fy]: [number, number], [tx, ty]: [number, number]): Region {
  if (fx === tx && fy === ty) return box;
  const [x0, x1, y0, y1] = box;
  const along = (a: number, b: number, from: number, to: number): [number, number] => {
    const first = Math.min(to - 1, Math.max(0, Math.floor((a * to) / from)));
    return [first, Math.min(to - 1, Math.max(first, Math.ceil(((b + 1) * to) / from) - 1))];
  };
  return [...along(x0, x1, fx, tx), ...along(y0, y1, fy, ty)];
}

// The view of a map of nx by ny pixels: a square window of its pixels (as
// wide as its larger side, about its middle, when it is shown whole), so
// that every map, whatever its shape, is shown in the same square, which
// keeps its size as it zooms; a map narrower than it is in its middle,
// the rest clear. How the wheel zooms it about a pixel (by a factor, at
// least LEAST pixels across), how it zooms to a box (the box in its
// middle), and how a drag pans it (its middle kept on the map). It is kept
// as fractions of the map, so that it stays as the map is made again at
// another size (e.g. a render).
export function useZoom(nx: number, ny: number, least = 8) {
  // (its middle, as fractions of the width and the height of the map, and
  // its side, of the larger side of the map; null: the map whole)
  const [view, setView] = useState<{ cx: number; cy: number; side: number } | null>(null);
  const big = Math.max(nx, ny);
  const windowOf = ({ cx, cy, side }: { cx: number; cy: number; side: number }): Region => {
    const n = Math.max(1, Math.round(side * big));
    const x0 = Math.round(cx * nx - n / 2);
    const y0 = Math.round(cy * ny - n / 2);
    return [x0, x0 + n - 1, y0, y0 + n - 1];
  };
  const shown = windowOf(view ?? { cx: 0.5, cy: 0.5, side: 1 });
  const [x0, , y0] = shown;
  const across = shown[1] - shown[0] + 1;
  // (a view n pixels across about a point of the map, in pixels; none if
  // it is as wide as the map, or wider)
  const viewOf = (n: number, mx: number, my: number) => {
    const side = Math.max(n, Math.min(least, big));
    if (side >= big) return null;
    return { cx: Math.min(1, Math.max(0, mx / nx)), cy: Math.min(1, Math.max(0, my / ny)), side: side / big };
  };
  const zoomAt = (x: number, y: number, factor: number) => {
    // (the pixel under the pointer kept where it is on the screen)
    const n = Math.round(across / factor);
    const scale = n / across;
    const [px, py] = [x + 0.5, y + 0.5];
    setView(viewOf(n, px + (x0 + across / 2 - px) * scale, py + (y0 + across / 2 - py) * scale));
  };
  const zoomTo = ([bx0, bx1, by0, by1]: Region) => setView(viewOf(Math.max(bx1 - bx0 + 1, by1 - by0 + 1), (bx0 + bx1 + 1) / 2, (by0 + by1 + 1) / 2));
  const panBy = (dx: number, dy: number) => {
    if (view === null) return;
    setView({ ...view, cx: Math.min(1, Math.max(0, view.cx + dx / nx)), cy: Math.min(1, Math.max(0, view.cy + dy / ny)) });
  };
  return { zoomed: view !== null, whole: () => setView(null), zoomAt, zoomTo, panBy, shown };
}

// Whether Space is held (a drag then pans, as in image editors), not as it
// is typed in a field; the root of the page is marked with it, for the
// cursor (followed once, for every map)
const space = { held: false, followed: false };

function followSpace() {
  if (space.followed) return;
  space.followed = true;
  const typing = (target: EventTarget | null) =>
    target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(target.tagName));
  const set = (held: boolean) => {
    space.held = held;
    if (held) document.documentElement.dataset.space = '';
    else delete document.documentElement.dataset.space;
  };
  window.addEventListener('keydown', (event) => event.code === 'Space' && !typing(event.target) && set(true));
  window.addEventListener('keyup', (event) => event.code === 'Space' && set(false));
  window.addEventListener('blur', () => set(false));
}

// How far the pointer moves (in screen pixels) before a click is a drag
const DRAG = 4;

// The steps of the colours of a ramp a map is drawn in (more than a screen
// can tell apart)
const LEVELS = 1024;

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
  onWhole,
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
  // How a drag pans the window (by whole pixels), and how a double click
  // shows it whole
  onPan?: (dx: number, dy: number) => void;
  onWhole?: () => void;
  onHover?: (pixel: [number, number] | null) => void;
  label: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [ny, nx] = shape;
  const [wx0, wx1, wy0, wy1] = window ?? [0, nx - 1, 0, ny - 1];
  const [w, h] = [wx1 - wx0 + 1, wy1 - wy0 + 1];
  // Where a press started (its pixel and its point on the screen, and
  // whether it pans), and the region dragged from it so far
  const press = useRef<{ pixel: [number, number]; at: [number, number]; pans: boolean } | null>(null);
  useEffect(followSpace, []);
  const [dragged, setDragged] = useState<Region | null>(null);
  // (the pan not yet made whole pixels, and where the pointer was)
  const panned = useRef<{ at: [number, number]; rest: [number, number] } | null>(null);
  // (whether the right button is down, stretching the colours)
  const stretching = useRef(false);
  const [keepLow, keepHigh] = kept ?? [null, null];
  // (drawn again in the colours of another theme or scheme)
  const scheme = useScheme();
  // (and in another colour map)
  const colourMap = useColourMap();
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;
    element.width = w;
    element.height = h;
    // (the colours of the ramp at LEVELS steps, each packed as the four
    // bytes of a pixel, RGBA: a pixel takes the nearest, rather than one
    // mixed of its own, so that a map is drawn fast enough to scrub its
    // channels)
    const stops = rampStops(diverging);
    const pack = ([r, g, b]: number[]) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
    const colours = new Uint32Array(LEVELS);
    for (let k = 0; k < LEVELS; k++) colours[k] = pack(rampColour(stops, k / (LEVELS - 1)));
    const empty = pack(token('--panel2'));
    const [low, high] = range;
    const span = high - low || 1;
    const image = context.createImageData(w, h);
    const pixels = new Uint32Array(image.data.buffer);
    // (a window can reach beyond the map, e.g. a narrow one in a square
    // view: those pixels are left clear)
    for (let y = Math.max(0, wy0); y <= Math.min(ny - 1, wy1); y++) {
      // (the row y is drawn from the bottom)
      const row = (wy1 - y) * w - wx0;
      for (let x = Math.max(0, wx0); x <= Math.min(nx - 1, wx1); x++) {
        const i = y * nx + x;
        const value = values[i];
        const shown = Number.isFinite(value) && !out?.[i] && (keepLow === null || value >= keepLow) && (keepHigh === null || value <= keepHigh);
        if (!shown) {
          pixels[row + x] = empty;
          continue;
        }
        const at = stretched((value - low) / span, stretch);
        pixels[row + x] = colours[at <= 0 ? 0 : at >= 1 ? LEVELS - 1 : Math.round(at * (LEVELS - 1))];
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, nx, ny, range, diverging, stretch, keepLow, keepHigh, out, wx0, wx1, wy0, wy1, w, h, scheme, colourMap]);
  // The pixel under a point of the screen (within the map and the window,
  // or else, if asked, as it is: maybe beyond the map)
  const pixelAt = (clientX: number, clientY: number, within = true): [number, number] => {
    const box = canvas.current!.getBoundingClientRect();
    const x = wx0 + Math.floor(((clientX - box.left) / box.width) * w);
    const y = wy1 - Math.floor(((clientY - box.top) / box.height) * h);
    if (!within) return [x, y];
    return [Math.min(Math.min(wx1, nx - 1), Math.max(Math.max(wx0, 0), x)), Math.min(Math.min(wy1, ny - 1), Math.max(Math.max(wy0, 0), y))];
  };
  // (the pixel of the map under the pointer; none beyond the map)
  const onMap = (clientX: number, clientY: number): [number, number] | null => {
    const [x, y] = pixelAt(clientX, clientY, false);
    return x >= 0 && x < nx && y >= 0 && y < ny ? [x, y] : null;
  };
  // (the wheel zooms, without scrolling the page: a listener that is not
  // passive, which reads the latest window through a ref)
  const zoom = useRef<(event: WheelEvent) => void>(() => {});
  zoom.current = (event) => {
    if (onZoom === undefined) return;
    event.preventDefault();
    onZoom(...pixelAt(event.clientX, event.clientY, false), event.deltaY < 0 ? 1.25 : 0.8);
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
    <div className="map" data-drag={onRegion ? 'region' : 'pan'}>
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
          if (event.button !== 0 && event.button !== 1) return;
          // (the middle button pans, not scrolls)
          if (event.button === 1) event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          const pans = event.button === 1 || space.held || onRegion === undefined;
          press.current = { pixel: pixelAt(event.clientX, event.clientY), at: [event.clientX, event.clientY], pans };
          panned.current = null;
        }}
        onPointerMove={(event) => {
          if (stretching.current) {
            stretchTo(event);
            return;
          }
          const start = press.current;
          if (start === null) {
            onHover?.(onMap(event.clientX, event.clientY));
            return;
          }
          const moved = Math.hypot(event.clientX - start.at[0], event.clientY - start.at[1]);
          // (a press that becomes a drag hides the pixel under the pointer;
          // a click keeps it, the pixel it picks)
          if (moved > DRAG) onHover?.(null);
          if (start.pans) {
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
          else if (event.button === 0) onPick?.(...now);
          setDragged(null);
        }}
        onPointerCancel={() => {
          stretching.current = false;
          press.current = null;
          setDragged(null);
        }}
        onPointerLeave={() => press.current === null && onHover?.(null)}
        onDoubleClick={() => onWhole?.()}
        onAuxClick={(event) => event.button === 1 && event.preventDefault()}
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
            {/* (light over a dark halo, to read on any map, in either scheme) */}
            <rect
              className="map-region-halo"
              x={shown[0]}
              y={ny - 1 - shown[3]}
              width={shown[1] - shown[0] + 1}
              height={shown[3] - shown[2] + 1}
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
