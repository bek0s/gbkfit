// The view of a galaxy as it is built (see the server's /views/galaxy): a
// map of its light, velocity or dispersion, of all its components or of
// one, with its colour bar; face-on, its motions in its plane as arrows
// over its light; and its curves (the rotation and dispersion of its
// disks, the circular velocity of its mass and of each mass component).
import { useEffect, useMemo, useRef } from 'react';
import type { GalaxyRender } from '../api/queries';
import { rampColour, rampStops, token } from './colours';
import { significant } from './format';

export type Quantity = 'light' | 'velocity' | 'dispersion';

// A value of a colour bar or an axis, to 3 significant digits
const fmt = (value: number) => significant(value, 3);

// The pixels of the map on the page, and of the plot of the curves
export const VIEW_SIZE = 440;
const CURVE_HEIGHT = 170;

// The faintest light whose velocity and dispersion show, of the peak of
// the light (an exponential disk has light everywhere)
const FAINTEST = 0.01;

// The arrows of the motions in the plane: one every this many pixels of
// the map, as long as this many pixels at the largest speed
const ARROW_EVERY = 10;
const ARROW_LENGTH = 16;

// The range of the values of a map that are finite, or null if none are
function rangeOf(values: Float32Array): [number, number] | null {
  let low = Infinity;
  let high = -Infinity;
  for (const v of values) {
    if (Number.isFinite(v)) {
      low = Math.min(low, v);
      high = Math.max(high, v);
    }
  }
  return low <= high ? [low, high] : null;
}

// The colours of a map: light from none to its peak (its square root, so
// that a disk shows beyond its centre), velocities about the middle of
// their range (diverging), dispersions over their range; with the range
// of the colour bar
function colouring(quantity: Quantity, values: Float32Array) {
  const range = rangeOf(values);
  if (range === null) return null;
  if (quantity === 'light') {
    const peak = Math.max(Math.abs(range[0]), Math.abs(range[1])) || 1;
    return { range: [0, peak] as [number, number], diverging: false, at: (v: number) => Math.sqrt(Math.max(0, v) / peak) };
  }
  if (quantity === 'velocity') {
    const middle = (range[0] + range[1]) / 2;
    const half = Math.max(range[1] - middle, 1e-6);
    return { range: [middle - half, middle + half] as [number, number], diverging: true, at: (v: number) => 0.5 + (v - middle) / (2 * half) };
  }
  const width = Math.max(range[1] - range[0], 1e-6);
  return { range, diverging: false, at: (v: number) => (v - range[0]) / width };
}

// The map of a quantity of a render, of all its components or of one (by
// name), with its colour bar; the arrows of the motions in the plane
// over the light, if asked for
export function GalaxyMap({
  render,
  quantity,
  component,
  arrows,
  unit,
}: {
  render: GalaxyRender;
  quantity: Quantity;
  component: string | null;
  arrows: boolean;
  unit: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const prefix = component === null ? '' : `${component}/`;
  const shown: Quantity = arrows ? 'light' : quantity;
  const light = render.maps[`${prefix}light`];
  const values = useMemo(() => {
    const map = render.maps[`${prefix}${shown}`];
    if (!map || shown === 'light' || !light) return map;
    const peak = rangeOf(light)?.[1] ?? 0;
    return map.map((v, i) => (light[i] > FAINTEST * peak ? v : NaN));
  }, [render, prefix, shown, light]);
  const colours = useMemo(() => (values ? colouring(shown, values) : null), [values, shown]);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const n = render.size;
    const image = context.createImageData(n, n);
    const empty = token('--panel');
    const stops = rampStops(colours?.diverging ?? false);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        // (the rows of the maps go up the sky; those of the canvas down)
        const v = values ? values[(n - 1 - y) * n + x] : NaN;
        const colour = colours && Number.isFinite(v) && (shown !== 'light' || v > 0) ? rampColour(stops, colours.at(v)) : empty;
        image.data.set([colour[0], colour[1], colour[2], 255], (y * n + x) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }, [render, values, colours, shown]);
  return (
    <div className="galaxy-map">
      <div className="galaxy-map-frame" style={{ width: VIEW_SIZE, height: VIEW_SIZE }}>
        <canvas ref={canvas} width={render.size} height={render.size} style={{ width: VIEW_SIZE, height: VIEW_SIZE }} />
        {arrows && <Arrows render={render} prefix={prefix} />}
        {!values && <p className="note galaxy-map-empty">Nothing of this here</p>}
      </div>
      {colours && !arrows && (
        <div className="galaxy-bar">
          <span>{fmt(colours.range[0])}</span>
          <span className="galaxy-ramp" style={{ background: gradient(colours.diverging) }} />
          <span>
            {fmt(colours.range[1])} {unit}
          </span>
        </div>
      )}
      {arrows && <p className="note galaxy-bar">Motions in the plane, over the light</p>}
    </div>
  );
}

// A ramp as a CSS gradient, to the right
function gradient(diverging: boolean): string {
  return `linear-gradient(to right, ${rampStops(diverging)
    .map((c) => `rgb(${c.join(', ')})`)
    .join(', ')})`;
}

// The motions in the plane of the sky (vx, vy) as arrows over the map,
// where there is light
function Arrows({ render, prefix }: { render: GalaxyRender; prefix: string }) {
  const vx = render.maps[`${prefix}vx`];
  const vy = render.maps[`${prefix}vy`];
  const light = render.maps[`${prefix}light`];
  if (!vx || !vy || !light) return null;
  const n = render.size;
  const faintest = FAINTEST * (rangeOf(light)?.[1] ?? 0);
  const scale = VIEW_SIZE / n;
  let fastest = 0;
  for (let i = 0; i < n * n; i++) {
    const speed = Math.hypot(vx[i], vy[i]);
    if (Number.isFinite(speed)) fastest = Math.max(fastest, speed);
  }
  if (fastest === 0) return null;
  const lines = [];
  for (let y = ARROW_EVERY / 2; y < n; y += ARROW_EVERY) {
    for (let x = ARROW_EVERY / 2; x < n; x += ARROW_EVERY) {
      const i = Math.floor(y) * n + Math.floor(x);
      if (!Number.isFinite(vx[i]) || !Number.isFinite(vy[i]) || !(light[i] > faintest)) continue;
      const length = ARROW_LENGTH / fastest;
      // (y up the sky; down the page)
      const x0 = x * scale;
      const y0 = (n - y) * scale;
      const x1 = x0 + vx[i] * length;
      const y1 = y0 - vy[i] * length;
      lines.push(<line key={i} x1={x0} y1={y0} x2={x1} y2={y1} markerEnd="url(#galaxy-arrow)" />);
    }
  }
  return (
    <svg className="galaxy-arrows" width={VIEW_SIZE} height={VIEW_SIZE}>
      <defs>
        <marker id="galaxy-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" />
        </marker>
      </defs>
      {lines}
    </svg>
  );
}

// What a curve is, for its legend, its colour (a token) and its dash
function curveStyle(name: string): { label: string; colour: string; dash?: string } {
  const [family, ...rest] = name.split('_');
  const own = rest.join('_');
  if (family === 'rotation') return { label: own ? `rotation of ${own}` : 'rotation', colour: 'accent' };
  if (family === 'dispersion') return { label: own ? `dispersion of ${own}` : 'dispersion', colour: 'fg2', dash: '5 3' };
  if (family === 'vcirc') return own ? { label: `circular, ${own}`, colour: 'map-pos', dash: '5 3' } : { label: 'circular velocity', colour: 'ok' };
  return { label: name, colour: 'faint' };
}

// The curves of the galaxy of a render shown: of its motions, or of its
// mass (its circular velocity, in total and of each mass component, with
// the rotation of its disks over them)
export function GalaxyCurves({ render, mass }: { render: GalaxyRender; mass: boolean }) {
  const radius = render.curves.radius ?? [];
  const names = Object.keys(render.curves).filter((name) => {
    const family = name.split('_')[0];
    return mass ? family === 'vcirc' || family === 'rotation' : family === 'rotation' || family === 'dispersion';
  });
  const curves = names.map((name) => ({ name, values: render.curves[name], ...curveStyle(name) }));
  let high = 0;
  let low = 0;
  for (const { values } of curves) {
    for (const v of values) {
      if (v !== null) {
        high = Math.max(high, v);
        low = Math.min(low, v);
      }
    }
  }
  if (curves.length === 0 || radius.length === 0 || high === low) {
    return <p className="note">No curves: add motions{mass ? ', or mass,' : ''} to see them.</p>;
  }
  const width = VIEW_SIZE;
  const height = CURVE_HEIGHT;
  const pad = { left: 36, right: 8, top: 8, bottom: 18 };
  const rmax = (radius[radius.length - 1] as number | null) ?? 1;
  const x = (r: number) => pad.left + (r / rmax) * (width - pad.left - pad.right);
  const y = (v: number) => pad.top + ((high - v) / (high - low)) * (height - pad.top - pad.bottom);
  const path = (values: (number | null)[]) =>
    values
      .map((v, i) => (v === null || radius[i] === null ? null : `${x(radius[i] as number)},${y(v)}`))
      .filter((p) => p !== null)
      .join(' ');
  return (
    <div className="galaxy-curves">
      <svg width={width} height={height}>
        <line className="galaxy-axis" x1={pad.left} x2={width - pad.right} y1={y(0)} y2={y(0)} />
        <text className="galaxy-tick" x={pad.left - 4} y={y(high) + 4} textAnchor="end">
          {fmt(high)}
        </text>
        <text className="galaxy-tick" x={pad.left - 4} y={y(0) + 4} textAnchor="end">
          0
        </text>
        <text className="galaxy-tick" x={width - pad.right} y={height - 4} textAnchor="end">
          {fmt(rmax)}″
        </text>
        {curves.map((c) => (
          <polyline key={c.name} points={path(c.values)} fill="none" stroke={`var(--${c.colour})`} strokeWidth={2} strokeDasharray={c.dash} />
        ))}
      </svg>
      <div className="galaxy-legend">
        {curves.map((c) => (
          <span key={c.name}>
            <i style={{ borderColor: `var(--${c.colour})`, borderStyle: c.dash ? 'dashed' : 'solid' }} />
            {c.label}
          </span>
        ))}
        <span className="note">km/s</span>
      </div>
    </div>
  );
}
