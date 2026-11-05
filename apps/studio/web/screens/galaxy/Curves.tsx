// The curves of a galaxy rendered (see the server's /views/galaxy), from
// its centre to its largest radius: the rotation and the dispersion of
// each of its components (in its colour); or, of its mass, the circular
// velocity in all and of each mass component (with the rotation of its
// components over them). The pointer reads every curve at a radius; the
// rings of the component chosen are marked on the axis of radii.
import { useElementSize } from '@mantine/hooks';
import { useState } from 'react';
import type { GalaxyRender } from '../../api/queries';
import { significant } from '../../components/format';

export interface CurvesFocus {
  mode: 'motions' | 'mass';
  // The radii of the rings of the component chosen (arcsec), if any
  rings: number[] | null;
}

const PAD = { left: 40, right: 18, top: 10, bottom: 22 };

// The colours of the mass components, in turn (tokens)
const MASS_COLOURS = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat1'];

interface Curve {
  name: string;
  label: string;
  colour: string;
  values: (number | null)[];
  dash?: string;
  width?: number;
}

// The curves shown, of a mode
function curvesOf(render: GalaxyRender, focus: CurvesFocus, colours: Map<string, string>): Curve[] {
  const curves: Curve[] = [];
  const names = Object.keys(render.curves);
  const own = (name: string, family: string) => (name === family ? '' : name.slice(family.length + 1));
  const colourOf = (component: string) => colours.get(component) ?? (colours.size === 1 ? [...colours.values()][0] : 'fg2');
  if (focus.mode === 'mass') {
    const masses = names.filter((n) => n.startsWith('vcirc_'));
    // (of one mass component, its curve is that of all)
    for (const [i, name] of (masses.length > 1 ? masses : []).entries()) {
      curves.push({ name, label: own(name, 'vcirc'), colour: MASS_COLOURS[i % MASS_COLOURS.length], values: render.curves[name], dash: '5 4' });
    }
    if (render.curves.vnewton && differs(render.curves.vnewton, render.curves.vcirc)) {
      curves.push({ name: 'vnewton', label: 'Newtonian', colour: 'muted', values: render.curves.vnewton, dash: '2 3' });
    }
    if (render.curves.vcirc) curves.push({ name: 'vcirc', label: 'circular velocity', colour: 'fg', values: render.curves.vcirc, width: 2.4 });
    for (const name of names.filter((n) => n === 'rotation' || n.startsWith('rotation_'))) {
      curves.push({ name, label: `rotation · ${own(name, 'rotation') || 'disk'}`, colour: colourOf(own(name, 'rotation')), values: render.curves[name], width: 1.4 });
    }
    return curves;
  }
  for (const name of names) {
    const family = name.startsWith('rotation') ? 'rotation' : name.startsWith('dispersion') ? 'dispersion' : null;
    if (family === null) continue;
    const component = own(name, family);
    curves.push({
      name,
      label: `${family}${component ? ` · ${component}` : ''}`,
      colour: colourOf(component),
      values: render.curves[name],
      dash: family === 'dispersion' ? '5 4' : undefined,
    });
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

export function Curves({
  render,
  focus,
  components,
  height: HEIGHT,
}: {
  render: GalaxyRender;
  focus: CurvesFocus;
  components: { name: string; colour: string }[];
  height: number;
}) {
  const { ref, width } = useElementSize();
  const [at, setAt] = useState<number | null>(null);
  const colours = new Map(components.map((c) => [c.name, c.colour]));
  const curves = curvesOf(render, focus, colours);
  const radius = render.curves.radius ?? [];
  let high = 0;
  let low = 0;
  for (const c of curves) for (const v of c.values) if (v !== null && Number.isFinite(v)) [high, low] = [Math.max(high, v), Math.min(low, v)];
  const empty = curves.length === 0 || radius.length === 0 || high === low;
  const rmax = (radius[radius.length - 1] as number | null) ?? 1;
  const yTicks = empty ? [] : ticks(low, high * 1.06, 4);
  const top = yTicks.length ? Math.max(high * 1.06, yTicks[yTicks.length - 1]) : 1;
  const plotW = Math.max(10, width - PAD.left - PAD.right);
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = (r: number) => PAD.left + (r / rmax) * plotW;
  const y = (v: number) => PAD.top + ((top - v) / (top - low || 1)) * plotH;
  const path = (values: (number | null)[]) => {
    let d = '';
    let pen = false;
    values.forEach((v, i) => {
      const r = radius[i];
      if (v === null || r === null || !Number.isFinite(v)) {
        pen = false;
        return;
      }
      d += `${pen ? 'L' : 'M'}${x(r as number).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };
  const index = at === null ? null : nearest(radius, at);
  return (
    <div className="gx-curves" ref={ref}>
      <div className="gx-curves-head">
        <span className="gx-curves-title">{focus.mode === 'mass' ? 'Circular velocity' : 'Rotation and dispersion'}</span>
        <div className="gx-legend">
          {curves.map((c) => (
            <span key={c.name}>
              <i style={{ borderColor: `var(--${c.colour})`, borderTopStyle: c.dash ? 'dashed' : 'solid' }} />
              {c.label}
            </span>
          ))}
        </div>
      </div>
      {empty ? (
        <p className="note gx-curves-empty">{focus.mode === 'mass' ? 'Add mass to the galaxy to see its circular velocity.' : 'No curves: the rotation and dispersion are drawn of disks with motions.'}</p>
      ) : (
        width > 0 && (
          <svg
            width={width}
            height={HEIGHT}
            onPointerMove={(event) => {
              const box = event.currentTarget.getBoundingClientRect();
              const r = ((event.clientX - box.left - PAD.left) / plotW) * rmax;
              setAt(r >= 0 && r <= rmax ? r : null);
            }}
            onPointerLeave={() => setAt(null)}
          >
            {yTicks.map((t) => (
              <g key={t}>
                <line className={t === 0 ? 'gx-axis' : 'gx-grid'} x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} />
                <text className="gx-tick" x={PAD.left - 6} y={y(t) + 3.5} textAnchor="end">
                  {significant(t, 3)}
                </text>
              </g>
            ))}
            {ticks(0, rmax, Math.max(2, Math.floor(plotW / 70))).map((t) => (
              <text key={t} className="gx-tick" x={x(t)} y={HEIGHT - 6} textAnchor={x(t) > width - PAD.right - 12 ? 'end' : 'middle'}>
                {significant(t, 3)}″
              </text>
            ))}
            {focus.rings?.map((r, i) =>
              r <= rmax ? <line key={i} className="gx-ring" x1={x(r)} x2={x(r)} y1={PAD.top + plotH} y2={PAD.top + plotH - 5} /> : null,
            )}
            {curves.map((c) => (
              <path key={c.name} d={path(c.values)} fill="none" stroke={`var(--${c.colour})`} strokeWidth={c.width ?? 1.8} strokeDasharray={c.dash} strokeLinejoin="round" />
            ))}
            {index !== null && radius[index] !== null && (
              <g className="gx-cursor">
                <line x1={x(radius[index] as number)} x2={x(radius[index] as number)} y1={PAD.top} y2={PAD.top + plotH} />
                {curves.map((c) =>
                  c.values[index] === null ? null : <circle key={c.name} cx={x(radius[index] as number)} cy={y(c.values[index] as number)} r={3} style={{ fill: `var(--${c.colour})` }} />,
                )}
              </g>
            )}
          </svg>
        )
      )}
      {index !== null && !empty && (
        <div className="gx-curves-read" style={{ left: Math.min(width - 190, Math.max(0, x(radius[index] as number) + 10)) }}>
          <b>{significant(radius[index] as number, 3)}″</b>
          {curves.map((c) =>
            c.values[index] === null ? null : (
              <span key={c.name}>
                <i style={{ background: `var(--${c.colour})` }} />
                {c.label}
                <em>{significant(c.values[index] as number, 4)}</em>
              </span>
            ),
          )}
          <small>km/s</small>
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
