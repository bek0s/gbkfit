// The planes of the picture of an observation (see Picture): a map of the
// sky (a galaxy as the observation sees it, on its pixels, or the galaxy
// itself), drawn in the ramps of the page, over a field (a window of its
// pixels about a centre, as wide as the field of the picture), with the
// PSF to scale at a corner, the lines of its pixels, a scale bar and the
// compass of the sky, and what is under the pointer, or the centre of a
// galaxy that a drag moves (as the source of the lens screen); the
// spectrum of a
// pixel with the LSF to scale; and the checks, chips under the planes.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { rampColour, rampGradient, rampStops, useColourMap, useScheme } from '../../components/colours';
import { significant } from '../../components/format';
import { robustRange } from '../../components/MapPanel';
import { Track } from '../../components/ui/rows';
import { snap } from '../../components/values';
import { barLength, Compass, ScaleBar } from '../lens/Picture';
import { short, type Ellipse } from './model';

// A map of the sky: its values (ny rows of nx, the first row at the bottom),
// the size of its pixels (arcsec), the pixel at the centre of what is
// shown (x, y; fractional, 0-based), whether its colours diverge (e.g. a
// velocity), and its unit
export interface SkyMap {
  values: Float32Array;
  nx: number;
  ny: number;
  step: number;
  centre: [number, number];
  diverging: boolean;
  unit: string;
}

// The pixels across the canvas of a plane, at most (its device pixels)
const MOST_PIXELS = 1200;

export function SkyPlane({
  map,
  range: given,
  width,
  psf,
  pixels,
  picked,
  onPick,
  readout,
  origin = [0, 0],
  centre,
  onCentre,
}: {
  map: SkyMap | undefined;
  // The range of its colours (by default, the robust range of its values,
  // see mapRange), e.g. that of another plane, to compare them
  range?: [number, number] | null;
  // The width of the field shown (arcsec)
  width: number;
  // The PSF to draw to scale (none: none)
  psf: Ellipse | null;
  // Whether its pixels are drawn as they are, with their lines
  pixels: boolean;
  // The pixel picked (its square drawn), and what a click on one does
  picked?: [number, number] | null;
  onPick?: (pixel: [number, number]) => void;
  // What is said of a pixel under the pointer (none: nothing)
  readout?: (pixel: [number, number], value: number) => string;
  // The centre of the field, as the model measures positions (x, y;
  // arcsec); the centre of a galaxy (its knob drawn), and where a drag
  // puts it (none: it is not dragged)
  origin?: [number, number];
  centre?: [number, number] | null;
  onCentre?: (centre: [number, number]) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [side, setSide] = useState(0);
  const [pointer, setPointer] = useState<[number, number] | null>(null);
  const scheme = useScheme();
  const colourMap = useColourMap();
  useLayoutEffect(() => {
    const element = box.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setSide(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  // (its colours over the range given, else the robust range of its
  // values)
  const own = useMemo(() => mapRange(map), [map]);
  const range = given ?? own ?? [0, 1];
  // (the map in colours, a pixel of an image for each of its pixels)
  const image = useMemo(() => {
    if (!map) return null;
    const { values, nx, ny, diverging } = map;
    const stops = rampStops(diverging);
    const offscreen = document.createElement('canvas');
    offscreen.width = nx;
    offscreen.height = ny;
    const context = offscreen.getContext('2d');
    if (!context) return null;
    const data = context.createImageData(nx, ny);
    const [low, high] = range;
    const span = high - low || 1;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const value = values[y * nx + x];
        const at = ((ny - 1 - y) * nx + x) * 4;
        if (!Number.isFinite(value)) continue;
        const [r, g, b] = rampColour(stops, (value - low) / span);
        data.data.set([r, g, b, 255], at);
      }
    }
    context.putImageData(data, 0, 0);
    return offscreen;
    // (drawn again in the colours of the theme and the colour map)
  }, [map, range[0], range[1], scheme, colourMap]); // eslint-disable-line react-hooks/exhaustive-deps
  // (the window of its pixels shown: the field about its centre)
  const window = map ? windowOf(map, width) : null;
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;
    const size = Math.min(MOST_PIXELS, Math.round(side * (globalThis.devicePixelRatio || 1)));
    if (element.width !== size) element.width = element.height = size;
    context.clearRect(0, 0, size, size);
    if (!image || !window || size === 0) return;
    const { x0, y0, cells } = window;
    const scale = size / cells;
    context.imageSmoothingEnabled = !pixels;
    // (its rows from the top: y0 is at the bottom of the window)
    context.drawImage(image, x0, map!.ny - (y0 + cells), cells, cells, 0, 0, size, size);
    if (pixels && scale >= 6) {
      context.strokeStyle = 'rgba(255, 255, 255, 0.07)';
      context.lineWidth = 1;
      context.beginPath();
      for (let k = Math.ceil(x0); k <= x0 + cells; k++) {
        const at = Math.round((k - x0) * scale) + 0.5;
        context.moveTo(at, 0);
        context.lineTo(at, size);
      }
      for (let k = Math.ceil(y0); k <= y0 + cells; k++) {
        const at = Math.round(size - (k - y0) * scale) + 0.5;
        context.moveTo(0, at);
        context.lineTo(size, at);
      }
      context.stroke();
    }
  }, [image, window?.x0, window?.y0, window?.cells, side, pixels, map]); // eslint-disable-line react-hooks/exhaustive-deps
  const pixelAt = (event: React.PointerEvent<HTMLElement> | React.MouseEvent<HTMLElement>): [number, number] | null => {
    if (!map || !window) return null;
    const rect = event.currentTarget.getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = 1 - (event.clientY - rect.top) / rect.height;
    const x = Math.floor(window.x0 + fx * window.cells);
    const y = Math.floor(window.y0 + fy * window.cells);
    return x >= 0 && y >= 0 && x < map.nx && y < map.ny ? [x, y] : null;
  };
  const value = pointer && map ? map.values[pointer[1] * map.nx + pointer[0]] : NaN;
  const half = width / 2;
  // (the centre of the galaxy where the pointer is: x to the right, west,
  // y up, north; see Compass)
  const centreAt = (event: React.PointerEvent<HTMLElement>): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    const span: [number, number] = [0, width];
    return [snap(origin[0] + ((event.clientX - rect.left) / rect.width - 0.5) * width, span), snap(origin[1] + (0.5 - (event.clientY - rect.top) / rect.height) * width, span)];
  };
  return (
    <div
      ref={box}
      className={['lp-plane', 'ob-plane', onCentre && 'lp-drag'].filter(Boolean).join(' ')}
      title={onCentre ? 'Drag to move the galaxy (only for the picture, not saved)' : undefined}
      data-picks={onPick ? true : undefined}
      onPointerDown={
        onCentre &&
        ((event) => {
          // (no text selected by the drag, which would cancel it)
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          onCentre(centreAt(event));
        })
      }
      onPointerMove={(event) => {
        setPointer(pixelAt(event));
        if (onCentre && event.buttons & 1) onCentre(centreAt(event));
      }}
      onPointerLeave={() => setPointer(null)}
      onClick={(event) => {
        const at = pixelAt(event);
        if (at && onPick) onPick(at);
      }}
    >
      <canvas ref={canvas} aria-label="The map" />
      <svg className="lp-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {picked && map && window && (
          <rect
            className="ob-picked"
            x={((picked[0] - window.x0) / window.cells) * 100}
            y={100 - ((picked[1] + 1 - window.y0) / window.cells) * 100}
            width={100 / window.cells}
            height={100 / window.cells}
          />
        )}
        {centre && <ellipse className="lp-h-knob" cx={((centre[0] - origin[0]) / width + 0.5) * 100} cy={(0.5 - (centre[1] - origin[1]) / width) * 100} rx={1.6} ry={1.6} />}
      </svg>
      {psf && <PsfMark psf={psf} width={width} />}
      <ScaleBar length={barLength(half)} half={half} />
      <Compass />
      {onCentre && centre ? (
        <span className="lp-readout lp-readout-corner">
          x {short(centre[0], 4)}″ · y {short(centre[1], 4)}″
        </span>
      ) : (
        readout && pointer && Number.isFinite(value) && <span className="lp-readout lp-readout-corner">{readout(pointer, value)}</span>
      )}
    </div>
  );
}

// The window of the pixels of a map shown: a square of the field's width
// (in pixels) about its centre, from its lower left corner (fractional)
function windowOf(map: SkyMap, width: number): { x0: number; y0: number; cells: number } {
  const cells = Math.max(1, width / map.step);
  return { x0: map.centre[0] + 0.5 - cells / 2, y0: map.centre[1] + 0.5 - cells / 2, cells };
}

// The PSF to scale, at the lower left corner of a plane of a width
// (arcsec), above its scale bar
function PsfMark({ psf, width }: { psf: Ellipse; width: number }) {
  const scale = 100 / width;
  const major = psf.major * scale;
  const minor = psf.minor * scale;
  const margin = 4 + major / 2;
  const [cx, cy] = [margin, 100 - 12 - major / 2];
  return (
    <svg className="lp-overlay ob-psf-mark" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <ellipse cx={cx} cy={cy} rx={minor / 2} ry={major / 2} transform={`rotate(${-psf.angle} ${cx} ${cy})`} />
    </svg>
  );
}

// The range of the colours of a map: the robust range of its values (see
// robustRange; about their median, of one that diverges)
export function mapRange(map: SkyMap | undefined): [number, number] | null {
  return map ? robustRange(map.values, map.diverging ? 'median' : 'sequential') : null;
}

// The colour bar of a map, under it: its low end, its ramp, its high end
// and its unit (a line of a fixed height, also while there is no map)
export function MapBar({ map }: { map: SkyMap | undefined }) {
  useScheme();
  useColourMap();
  const range = useMemo(() => mapRange(map), [map]);
  return (
    <div className="ob-bar" data-off={range === null || undefined}>
      <span>{range ? number(range[0]) : ' '}</span>
      <i style={{ background: rampGradient(map?.diverging ?? false, 'to right') }} />
      <span>{range ? number(range[1]) : ' '}</span>
      <span className="ob-bar-unit">{map?.unit ?? ''}</span>
    </div>
  );
}

function number(value: number): string {
  const text = significant(Math.abs(value), 3);
  return value < 0 ? `−${text}` : text;
}

// The spectrum of a pixel (its values against the velocities of its
// channels, km/s), with the LSF to scale (its values on channels of its
// own width, km/s), centred on the light of the spectrum, as tall as half
// its peak; ticks at the channels, longer every fifth
export function SpectrumStrip({ spectrum, lsf }: { spectrum: { values: (number | null)[]; velocities: number[] } | undefined; lsf: { values: number[]; step: number } | null }) {
  const [w, h] = [600, 110];
  if (!spectrum || spectrum.velocities.length < 2) return <svg className="ob-strip" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true" />;
  const { values, velocities } = spectrum;
  const first = velocities[0];
  const last = velocities[velocities.length - 1];
  const step = (last - first) / (velocities.length - 1);
  const [lo, hi] = [Math.min(first, last) - Math.abs(step) / 2, Math.max(first, last) + Math.abs(step) / 2];
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * w;
  const peak = Math.max(1e-30, ...values.map((v) => (v === null ? 0 : v)));
  const y = (f: number) => h - 14 - (f / peak) * (h - 24);
  const line = values.map((v, i) => `${i ? 'L' : 'M'}${x(velocities[i]).toFixed(1)},${y(v ?? 0).toFixed(1)}`).join('');
  // (the LSF about the mean velocity of the light of the spectrum)
  let total = 0;
  let moment = 0;
  values.forEach((v, i) => {
    if (v !== null && v > 0) {
      total += v;
      moment += v * velocities[i];
    }
  });
  const centre = total > 0 ? moment / total : (lo + hi) / 2;
  let profile = '';
  if (lsf && lsf.values.length > 1) {
    const top = Math.max(...lsf.values) || 1;
    const middle = (lsf.values.length - 1) / 2;
    profile = lsf.values.map((f, i) => `${i ? 'L' : 'M'}${x(centre + (i - middle) * lsf.step).toFixed(1)},${y((0.45 * peak * f) / top).toFixed(1)}`).join('');
  }
  return (
    <svg className="ob-strip" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-label="The spectrum of the pixel picked">
      {velocities.map((v, i) => (
        <line key={i} className="ob-tick" x1={x(v)} x2={x(v)} y1={h - 12} y2={h - (i % 5 ? 9 : 5)} />
      ))}
      <line className="ob-axis" x1={0} x2={w} y1={h - 13} y2={h - 13} />
      <path className="ob-spectrum" d={line} />
      {profile && <path className="ob-lsf" d={profile} />}
    </svg>
  );
}

// A check of the picture, a chip: passed, a problem, or a note
export interface Chip {
  kind: 'ok' | 'warn' | 'note';
  text: string;
  title?: string;
}

export function Chips({ chips }: { chips: Chip[] }) {
  return (
    <ul className="ob-chips" aria-label="Checks">
      {chips.map((chip, i) => (
        <li key={i} className="ob-chip" data-kind={chip.kind} title={chip.title ?? chip.text}>
          <b>{chip.kind === 'ok' ? '✓' : chip.kind === 'warn' ? '!' : '·'}</b>
          <span>{chip.text}</span>
        </li>
      ))}
    </ul>
  );
}

// A knob of the test galaxy: its name, its track (its knob at a fraction
// of it; a drag or a click moves it), its value
export function Knob({ label, at, onAt, value }: { label: string; at: number; onAt: (at: number) => void; value: string }) {
  return (
    <div className="ob-knob">
      <span>{label}</span>
      <Track at={at} onAt={onAt} />
      <span className="lp-src-v">{value}</span>
    </div>
  );
}

// The FWHM of a PSF in pixels of a size (arcsec), as a check says it
export function psfPixels(psf: Ellipse, step: number): string {
  return short(psf.major / step, 2);
}
