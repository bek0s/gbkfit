// Curves of a galaxy along its radius (see the server's /views/galaxy),
// drawn on one plot that fills its place: each its label, colour and dash,
// a legend over the plot's top right, and a readout of every curve at the
// radius under the pointer; the rings of a component marked on the axis
// of radii. What the curves are is up to the views (see Profiles): the
// speeds, dispersions and densities of the components, their orientation
// along their rings, the circular speeds of the mass.
import { useElementSize } from '@mantine/hooks';
import { useState } from 'react';
import type { GalaxyRender } from '../../api/queries';
import { significant } from '../../components/format';

const PAD = { left: 44, right: 14, top: 10, bottom: 22 };

// The most decades of a logarithmic axis, below its highest value
const LOG_DECADES = 5;

// The colours of the mass components, in turn (tokens)
const MASS_COLOURS = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat1'];

export interface Curve {
  name: string;
  label: string;
  // A token of its colour (e.g. 'cat1'), its dash and width
  colour: string;
  values: (number | null)[];
  dash?: string;
  width?: number;
  // Its own radii (arcsec), if not those of the plot (e.g. of rings)
  radius?: (number | null)[];
}

// The circular speeds of the mass of a render: in all (and in Newtonian
// gravity, if it differs), of each mass component, with the rotation of
// its components over them (each in the colour of its component)
export function massCurves(render: GalaxyRender, colours: Map<string, string>): Curve[] {
  const curves: Curve[] = [];
  const names = Object.keys(render.curves);
  const own = (name: string, family: string) => (name === family ? '' : name.slice(family.length + 1));
  const masses = names.filter((n) => n.startsWith('vcirc_'));
  // (of one mass component, its curve is that of all)
  for (const [i, name] of (masses.length > 1 ? masses : []).entries()) {
    curves.push({ name, label: own(name, 'vcirc'), colour: MASS_COLOURS[i % MASS_COLOURS.length], values: render.curves[name], dash: '5 4' });
  }
  if (render.curves.vnewton && differs(render.curves.vnewton, render.curves.vcirc)) {
    curves.push({ name: 'vnewton', label: 'Newtonian', colour: 'muted', values: render.curves.vnewton, dash: '2 3' });
  }
  if (render.curves.vcirc) curves.push({ name: 'vcirc', label: 'circular speed', colour: 'fg', values: render.curves.vcirc, width: 2.4 });
  // (of the components that rotate: not e.g. a cone that flows out only)
  for (const name of names.filter((n) => (n === 'rotation' || n.startsWith('rotation_')) && render.curves[n].some((v) => v !== null && v !== 0))) {
    const component = own(name, 'rotation');
    curves.push({ name, label: `rotation · ${component || 'component'}`, colour: colours.get(component) ?? 'fg2', values: render.curves[name], width: 1.4 });
  }
  return curves;
}

function differs(a: (number | null)[], b: (number | null)[] | undefined): boolean {
  if (!b) return true;
  return a.some((v, i) => v !== null && b[i] !== null && Math.abs(v - (b[i] as number)) > 1e-6 * Math.max(1, Math.abs(v)));
}

// Round ticks from 0 (or a low below it) to a high: about count of them
function ticks(low: number, high: number, count: number): number[] {
  const span = high - low || 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? 10 * power;
  const result = [];
  for (let t = Math.ceil(low / step) * step; t <= high + 1e-9 * span; t += step) result.push(Math.abs(t) < 1e-12 ? 0 : t);
  return result;
}

export function CurvePlot({
  curves,
  radius,
  rings = null,
  unit,
  empty: emptyText,
  log = false,
}: {
  curves: Curve[];
  // The radii of the curves (arcsec), and those of the rings to mark
  radius: (number | null)[];
  rings?: number[] | null;
  // The unit of the values, under the readout
  unit: string;
  // What shows without curves
  empty: string;
  // Whether the values are on a logarithmic axis (of decades, from at
  // most LOG_DECADES below the highest value; values at or below 0 not
  // drawn)
  log?: boolean;
}) {
  const { ref, width, height } = useElementSize();
  const [at, setAt] = useState<number | null>(null);
  const radiusOf = (c: Curve) => c.radius ?? radius;
  let high = 0;
  let low = 0;
  let rmax = 0;
  for (const c of curves) {
    for (const v of c.values) if (v !== null && Number.isFinite(v)) [high, low] = [Math.max(high, v), Math.min(low, v)];
    for (const r of radiusOf(c)) if (r !== null) rmax = Math.max(rmax, r);
  }
  // (on a log axis: the decades from the lowest positive value, at most
  // LOG_DECADES below the highest, to the highest)
  const positive = log ? curves.flatMap((c) => c.values.filter((v): v is number => v !== null && Number.isFinite(v) && v > 0)) : [];
  const decades = positive.length ? [Math.max(Math.floor(Math.log10(Math.min(...positive))), Math.ceil(Math.log10(Math.max(...positive))) - LOG_DECADES), Math.ceil(Math.log10(Math.max(...positive)))] : [0, 1];
  if (decades[1] === decades[0]) decades[0] -= 1;
  const empty = curves.length === 0 || rmax === 0 || (log ? positive.length === 0 : high === low);
  const yTicks = empty ? [] : log ? Array.from({ length: decades[1] - decades[0] + 1 }, (_, i) => 10 ** (decades[0] + i)) : ticks(low, high * 1.06, 4);
  const top = yTicks.length ? Math.max(high * 1.06, yTicks[yTicks.length - 1]) : 1;
  const plotW = Math.max(10, width - PAD.left - PAD.right);
  const plotH = Math.max(10, height - PAD.top - PAD.bottom);
  const x = (r: number) => PAD.left + (r / (rmax || 1)) * plotW;
  const y = log
    ? (v: number) => PAD.top + ((decades[1] - Math.log10(Math.max(v, 10 ** decades[0]))) / (decades[1] - decades[0])) * plotH
    : (v: number) => PAD.top + ((top - v) / (top - low || 1)) * plotH;
  const path = (c: Curve) => {
    const radii = radiusOf(c);
    let d = '';
    let pen = false;
    c.values.forEach((v, i) => {
      const r = radii[i];
      if (v === null || r === null || r === undefined || !Number.isFinite(v) || (log && v <= 0)) {
        pen = false;
        return;
      }
      d += `${pen ? 'L' : 'M'}${x(r).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };
  // (each curve read at the radius of its nearest point)
  const read = at === null ? [] : curves.map((c) => ({ c, i: nearest(radiusOf(c), at) })).filter(({ c, i }) => i !== null && c.values[i] !== null && (!log || (c.values[i] as number) > 0));
  return (
    <div className="gx-plot" ref={ref}>
      {empty ? (
        <p className="note gx-curves-empty">{emptyText}</p>
      ) : (
        width > 0 &&
        height > 0 && (
          <svg
            width={width}
            height={height}
            onPointerMove={(event) => {
              const box = event.currentTarget.getBoundingClientRect();
              const r = ((event.clientX - box.left - PAD.left) / plotW) * rmax;
              setAt(r >= 0 && r <= rmax ? r : null);
            }}
            onPointerLeave={() => setAt(null)}
          >
            {yTicks.map((t) => (
              <g key={t}>
                <line className={t === 0 || (log && t === yTicks[0]) ? 'gx-axis' : 'gx-grid'} x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} />
                <text className="gx-tick" x={PAD.left - 6} y={y(t) + 3.5} textAnchor="end">
                  {significant(t, 3)}
                </text>
              </g>
            ))}
            {ticks(0, rmax, Math.max(2, Math.floor(plotW / 70))).map((t) => (
              <text key={t} className="gx-tick" x={x(t)} y={height - 6} textAnchor={x(t) > width - PAD.right - 12 ? 'end' : 'middle'}>
                {significant(t, 3)}″
              </text>
            ))}
            {rings?.map((r, i) => (r <= rmax ? <line key={i} className="gx-ring" x1={x(r)} x2={x(r)} y1={PAD.top + plotH} y2={PAD.top + plotH - 5} /> : null))}
            {curves.map((c) => (
              <path key={c.name} d={path(c)} fill="none" stroke={`var(--${c.colour})`} strokeWidth={c.width ?? 1.8} strokeDasharray={c.dash} strokeLinejoin="round" />
            ))}
            {at !== null && (
              <g className="gx-cursor">
                <line x1={x(at)} x2={x(at)} y1={PAD.top} y2={PAD.top + plotH} />
                {read.map(({ c, i }) => (
                  <circle key={c.name} cx={x(radiusOf(c)[i!] as number)} cy={y(c.values[i!] as number)} r={3} style={{ fill: `var(--${c.colour})` }} />
                ))}
              </g>
            )}
          </svg>
        )
      )}
      {!empty && (
        <div className="gx-legend-box">
          {curves.map((c) => (
            <span key={c.name}>
              <svg viewBox="0 0 22 8" aria-hidden="true">
                <line x1="1" x2="21" y1="4" y2="4" stroke={`var(--${c.colour})`} strokeWidth={Math.min(c.width ?? 1.8, 2.4)} strokeDasharray={c.dash} />
              </svg>
              {c.label}
            </span>
          ))}
        </div>
      )}
      {at !== null && read.length > 0 && (
        <div className="gx-curves-read" style={{ left: Math.min(width - 190, Math.max(0, x(at) + 10)) }}>
          <b>{significant(at, 3)}″</b>
          {read.map(({ c, i }) => (
            <span key={c.name}>
              <i style={{ background: `var(--${c.colour})` }} />
              {c.label}
              <em>{significant(c.values[i!] as number, 4)}</em>
            </span>
          ))}
          <small>{unit}</small>
        </div>
      )}
    </div>
  );
}

function nearest(radius: (number | null)[], r: number): number | null {
  let best: number | null = null;
  for (let i = 0; i < radius.length; i++) {
    const v = radius[i];
    if (v === null) continue;
    if (best === null || Math.abs(v - r) < Math.abs((radius[best] as number) - r)) best = i;
  }
  return best;
}
