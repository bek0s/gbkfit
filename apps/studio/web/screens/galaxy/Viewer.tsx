// The view of a galaxy as it is built (see the server's /views/galaxy and
// /views/galaxy-cube): a map of its light, velocity or dispersion, of all
// its components or of one, as seen, face-on (its motions in its plane as
// arrows) or edge-on; or seen by an observation of the project (see Seen:
// its cube, or its maps, as the observation records them). The map is in a
// map panel (see MapPanel: as every map of the Studio), rendered at about
// its size on the screen; its place on the sky is in arcsec from its
// centre. Light can be shown on a linear, square-root or logarithmic
// scale.
import { useDebouncedValue, useElementSize } from '@mantine/hooks';
import { useMemo, useRef, useState } from 'react';
import { useGalaxyRender, type GalaxyRender } from '../../api/queries';
import { MapPanel } from '../../components/MapPanel';
import type { Region } from '../../components/MapView';
import { Choice, Segments } from '../../components/ui/inputs';
import { Curves, type CurvesFocus } from './Curves';
import { SeenView } from './Seen';
import type { Options } from './options';

export type Orientation = 'sky' | 'face-on' | 'edge-on';
export type Quantity = 'light' | 'velocity' | 'dispersion';
type Scale = 'linear' | 'sqrt' | 'log';

// The faintest light whose velocity and dispersion show, of the peak of
// the light (an exponential disk has light everywhere)
const FAINTEST = 0.005;

// The arrows of the motions in the plane: about this many across the
// window, each at most as long as the space between them
const ARROWS_ACROSS = 18;

// The pixels across a render: about half those of the map on the screen,
// in steps (so that a small change of the window does not render again).
// A galaxy with depth (a thick disk, a cone, a spheroid) is rendered on a
// grid of voxels as deep as the field is wide, whose cost grows as the
// cube of its size: fewer pixels. While it changes (e.g. a slider is
// moved), fewer still; as it settles, sharp again.
function renderSize(shown: number, deep: boolean, moving: boolean): number {
  const full = Math.min(deep ? 160 : 384, Math.max(128, Math.round(shown / 2 / 32) * 32));
  return moving ? Math.min(full, deep ? 96 : 192) : full;
}

// Whether a galaxy has depth (see renderSize)
function hasDepth(galaxy: Options): boolean {
  const components = Array.isArray(galaxy.components) ? galaxy.components : [galaxy.components];
  return components.some((c) => {
    const component = c as Options | undefined;
    if (!component) return false;
    return component.type === 'thick_disk' || component.type === 'cone' || component.type === 'spheroid';
  });
}

// Whether a galaxy has height (a thick disk, a cone, a spheroid,
// particles): one of thin disks and points alone lies in a plane, and is
// a line seen edge-on
function hasHeight(galaxy: Options): boolean {
  const components = Array.isArray(galaxy.components) ? galaxy.components : [galaxy.components];
  return components.some((c) => {
    const component = c as Options | undefined;
    return component !== undefined && component.type !== 'thin_disk' && component.type !== 'point';
  });
}

// (what "Seen by" lists first: nothing, the galaxy as it is)
const IDEAL = '@ideal';

// (a change within this long of the last is one of a gesture: see
// renderSize)
const SETTLE_MS = 350;

// The heights of the plot of the curves (its least and most), and of the
// rest of their card (pixels)
const CURVES_LEAST = 170;
const CURVES_MOST = 300;
const CURVES_FRAME = 44;

export interface ViewerProps {
  project: string;
  // What sees it: none (it as it is), or an observation of the project, by
  // name; and the observations there are
  seenBy: string | null;
  onSeenBy: (observation: string | null) => void;
  observations: string[];
  // The options and the properties (for the render; see params.forRender)
  // of the galaxy, as they settle
  galaxy: Options | undefined;
  properties: Record<string, unknown>;
  orientation: Orientation;
  onOrientation: (orientation: Orientation) => void;
  quantity: Quantity;
  onQuantity: (quantity: Quantity) => void;
  // The components (their names and colours), and the one shown alone
  components: { name: string; colour: string }[];
  isolated: string | null;
  onIsolated: (name: string | null) => void;
  // What the curves under the map are of: the motions, or the mass; and
  // the radii of the rings of the component chosen, marked on them
  curves: CurvesFocus;
}

export function Viewer(props: ViewerProps) {
  const { galaxy, onOrientation, quantity, onQuantity, components, isolated, onIsolated, seenBy, onSeenBy, observations } = props;
  // (one no longer there: none)
  const seen = seenBy !== null && observations.includes(seenBy) ? seenBy : null;
  // (edge-on, as seen, of a galaxy with no height: see hasHeight)
  const flat = useMemo(() => galaxy !== undefined && !hasHeight(galaxy), [galaxy]);
  const orientation = flat && props.orientation === 'edge-on' ? 'sky' : props.orientation;
  return (
    <div className="gx-view">
      <div className="gx-toolbar">
        <label className="gx-seen-by" title="As it is (fine pixels, no instrument), or as an observation of the project records it: its grid, PSF, LSF and line of sight">
          <span>Seen by</span>
          <Choice
            items={[{ value: IDEAL, label: 'Nothing: as it is' }, ...observations.map((name) => ({ value: name, label: name }))]}
            value={seen ?? IDEAL}
            onChange={(next) => onSeenBy(next === IDEAL ? null : next)}
            label="What sees the galaxy"
          />
        </label>
        <Segments
          items={[
            { value: 'sky', label: 'As seen' },
            { value: 'face-on', label: 'Face-on', disabled: seen !== null, title: seen ? 'An observation sees it on the sky' : undefined },
            {
              value: 'edge-on',
              label: 'Edge-on',
              disabled: flat || seen !== null,
              title: seen
                ? 'An observation sees it on the sky'
                : flat
                  ? 'Its disks are thin, with no height: edge-on, a line. Make a disk thick to see it edge-on.'
                  : undefined,
            },
          ]}
          value={seen ? 'sky' : orientation}
          onChange={onOrientation}
          label="How the galaxy is seen"
        />
        {seen === null && <Segments
          items={[
            { value: 'light', label: 'Light' },
            { value: 'velocity', label: 'Velocity' },
            { value: 'dispersion', label: 'Dispersion' },
          ]}
          value={quantity}
          onChange={onQuantity}
          label="What is shown"
        />}
        {components.length > 1 && seen === null && (
          <div className="gx-isolate" role="group" aria-label="The components shown">
            <button className="gx-chip" data-on={isolated === null || undefined} onClick={() => onIsolated(null)}>
              All
            </button>
            {components.map((c) => (
              <button key={c.name} className="gx-chip" data-on={isolated === c.name || undefined} onClick={() => onIsolated(isolated === c.name ? null : c.name)}>
                <i style={{ background: `var(--${c.colour})` }} />
                {c.name}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="gx-stage">
        {galaxy === undefined ? (
          <div className="gx-empty">
            <p>An empty galaxy.</p>
            <p className="note">Add its first component in the outline to see it.</p>
          </div>
        ) : seen !== null ? (
          <SeenView project={props.project} galaxy={galaxy} properties={props.properties} observation={seen} curves={props.curves} components={components} />
        ) : (
          <RenderPane {...props} galaxy={galaxy} quantity={quantity} />
        )}
      </div>
    </div>
  );
}

// A render of the galaxy: its map, in a map panel, and its curves under it
function RenderPane({
  project,
  galaxy,
  properties,
  orientation,
  quantity,
  isolated,
  components,
  curves,
}: ViewerProps & { galaxy: Options; quantity: Quantity }) {
  // (the map as large as the column allows, over curves of their least
  // height; the curves taller where the map leaves room)
  const { ref: column, width, height } = useElementSize();
  const side = Math.max(160, Math.min(width - 76, height - CURVES_LEAST - CURVES_FRAME - 40));
  const curvesHeight = Math.max(CURVES_LEAST, Math.min(CURVES_MOST, height - side - CURVES_FRAME - 40));
  const key = JSON.stringify([galaxy, properties, orientation]);
  const [settledKey] = useDebouncedValue(key, SETTLE_MS);
  const size = renderSize(side || 400, useMemo(() => hasDepth(galaxy), [galaxy]), settledKey !== key);
  const render = useGalaxyRender(project, galaxy, properties, orientation, size);
  // (the last render that worked stays while the galaxy has a problem)
  const last = useRef<GalaxyRender | undefined>(undefined);
  if (render.data) last.current = render.data;
  const shown = render.data ?? last.current;
  const [scale, setScale] = useState<Scale>('sqrt');
  const n = shown?.size ?? size;
  // (face-on, the velocity on the line of sight is none: the speed in the
  // plane, under its arrows)
  const arrows = orientation === 'face-on' && quantity === 'velocity';
  const what: Shown = arrows ? 'speed' : quantity;
  const map = useMemo(() => (shown ? mapOf(shown, what, isolated, scale) : null), [shown, what, isolated, scale]);
  const unit = quantity === 'light' ? '' : 'km/s';
  // (its pixels on the sky: arcsec from its centre, where the model
  // measures positions from)
  const frame = useMemo(() => (shown ? { step: [shown.step, shown.step] as [number, number], rpix: [(n - 1) / 2, (n - 1) / 2] as [number, number] } : null), [shown, n]);
  return (
    <div className="gx-render" ref={column}>
      <MapPanel
        values={map?.shown ?? null}
        read={map?.values}
        shape={[n, n]}
        range={map?.range ?? [0, 1]}
        bar={map?.bar}
        diverging={what === 'velocity'}
        unit={unit}
        stretchKey={`galaxy/${quantity}`}
        frame={frame}
        pixels={false}
        smooth
        label={`The ${quantity} of the galaxy`}
        note={render.error ? <p className="gx-error">{render.error.message}</p> : <div className="gx-map-wait" />}
        tags={
          <>
            {render.isFetching && <span className="gx-busy" title="Rendering" />}
            {arrows && map?.any && <span className="map-source">speed in the plane</span>}
          </>
        }
        over={(_, window) => (
          <>
            {arrows && shown && <Arrows render={shown} component={isolated} window={window} />}
            {map && !map.any && <p className="gx-map-note">{quantity === 'light' ? 'No light here' : `No ${quantity} here: it needs motions, and light to weigh them`}</p>}
            {render.error && shown && <p className="gx-error">{render.error.message}</p>}
          </>
        )}
        side={
          quantity === 'light' && (
            <div className="gx-scale" role="group" aria-label="The scale of the light">
              {(['linear', 'sqrt', 'log'] as Scale[]).map((s) => (
                <button key={s} data-on={scale === s || undefined} onClick={() => setScale(s)} title={{ linear: 'Linear', sqrt: 'Square root', log: 'Logarithmic' }[s]}>
                  {{ linear: 'lin', sqrt: '√', log: 'log' }[s]}
                </button>
              ))}
            </div>
          )
        }
      />
      {shown && <Curves render={shown} focus={curves} components={components} height={curvesHeight} />}
    </div>
  );
}

// What a map shows: a quantity, or (face-on) the speed in the plane
type Shown = Quantity | 'speed';

function speedOf(vx: Float32Array | undefined, vy: Float32Array | undefined): Float32Array | undefined {
  if (!vx || !vy) return undefined;
  return vx.map((x, i) => Math.hypot(x, vy[i]));
}

// The light of the components of a render that have a map (of one, if
// given), summed
function lightOf(render: GalaxyRender, component: string | null, map: string): Float32Array | undefined {
  if (component !== null) return render.maps[`${component}/light`];
  const names = Object.keys(render.maps)
    .filter((key) => key.endsWith(`/${map}`))
    .map((key) => key.slice(0, -map.length - 1));
  if (names.length === 0) return undefined;
  const total = new Float32Array(render.size * render.size);
  for (const name of names) {
    const light = render.maps[`${name}/light`];
    if (light) for (let i = 0; i < total.length; i++) total[i] += Math.abs(light[i]) || 0;
  }
  return total;
}

interface ShownMap {
  // The values (of the light, on its scale) and their range for colours
  shown: Float32Array;
  range: [number, number];
  // The values as they are, and their range for the colour bar
  values: Float32Array;
  bar: [number, number];
  any: boolean;
}

// The map of a quantity of a render (of all its components or of one), its
// velocity and dispersion where there is light (above FAINTEST of its
// peak), its light on a scale
function mapOf(render: GalaxyRender, quantity: Shown, component: string | null, scale: Scale): ShownMap {
  const prefix = component === null ? '' : `${component}/`;
  const n = render.size * render.size;
  // (the speed in the plane of the sky, of its parts along x and y)
  const raw = quantity === 'speed' ? speedOf(render.maps[`${prefix}vx`], render.maps[`${prefix}vy`]) : render.maps[`${prefix}${quantity}`];
  // (the light that weighs a velocity or a dispersion: of the components
  // that have it, as the server weighs them)
  const light = quantity === 'light' ? render.maps[`${prefix}light`] : lightOf(render, component, quantity === 'speed' ? 'vx' : quantity);
  if (!raw) {
    const empty = new Float32Array(n).fill(NaN);
    return { shown: empty, range: [0, 1], values: empty, bar: [0, 1], any: false };
  }
  let values = raw;
  if (quantity !== 'light' && light) {
    const peak = finiteRange(light)?.[1] ?? 0;
    values = raw.map((v, i) => (light[i] > FAINTEST * peak ? v : NaN));
  }
  const range = finiteRange(values);
  if (range === null) return { shown: values, range: [0, 1], values, bar: [0, 1], any: false };
  if (quantity === 'velocity') {
    // (diverging about the middle of the velocities: their systemic one)
    const middle = (range[0] + range[1]) / 2;
    // (at least 1 km/s either way: a galaxy seen face-on, or not moving)
    const half = Math.max(range[1] - middle, 1);
    const bar: [number, number] = [middle - half, middle + half];
    return { shown: values, range: bar, values, bar, any: true };
  }
  if (quantity === 'dispersion' || quantity === 'speed') {
    const bar: [number, number] = range[0] === range[1] ? [range[0] - 1, range[1] + 1] : range;
    return { shown: values, range: bar, values, bar, any: true };
  }
  // (light: from none to its peak, on its scale; the log over 4 decades)
  const peak = Math.max(range[1], 1e-30);
  const floor = peak * 1e-4;
  const on = (v: number) => (scale === 'linear' ? v : scale === 'sqrt' ? Math.sqrt(Math.max(0, v)) : Math.log10(Math.max(v, floor)));
  const shown = values.map((v) => (v > 0 ? on(v) : NaN));
  return { shown, range: [on(scale === 'log' ? floor : 0), on(peak)], values, bar: [scale === 'log' ? floor : 0, peak], any: true };
}

function finiteRange(values: Float32Array): [number, number] | null {
  let low = Infinity;
  let high = -Infinity;
  for (const v of values) {
    if (Number.isFinite(v)) {
      if (v < low) low = v;
      if (v > high) high = v;
    }
  }
  return low <= high ? [low, high] : null;
}

// The motions in the plane of the sky (vx, vy) as arrows over the map,
// where there is light; in the pixels of the window
function Arrows({ render, component, window }: { render: GalaxyRender; component: string | null; window: Region }) {
  const prefix = component === null ? '' : `${component}/`;
  const vx = render.maps[`${prefix}vx`];
  const vy = render.maps[`${prefix}vy`];
  const light = render.maps[`${prefix}light`];
  const arrows = useMemo(() => {
    if (!vx || !vy || !light) return [];
    const n = render.size;
    const [x0, x1, y0, y1] = window;
    const every = Math.max(2, (x1 - x0 + 1) / ARROWS_ACROSS);
    const faintest = FAINTEST * 4 * (finiteRange(light)?.[1] ?? 0);
    let fastest = 0;
    for (let i = 0; i < n * n; i++) {
      const speed = Math.hypot(vx[i], vy[i]);
      if (Number.isFinite(speed) && light[i] > faintest) fastest = Math.max(fastest, speed);
    }
    if (fastest === 0) return [];
    const length = (every * 0.9) / fastest;
    const result: { x: number; y: number; dx: number; dy: number; key: number }[] = [];
    for (let y = y0 + every / 2; y <= y1; y += every) {
      for (let x = x0 + every / 2; x <= x1; x += every) {
        const i = Math.floor(y) * n + Math.floor(x);
        if (!Number.isFinite(vx[i]) || !Number.isFinite(vy[i]) || !(light[i] > faintest)) continue;
        result.push({ x: x + 0.5, y: n - y - 0.5, dx: vx[i] * length, dy: -vy[i] * length, key: i });
      }
    }
    return result;
  }, [vx, vy, light, render.size, window]);
  const n = render.size;
  const [x0, x1, y0, y1] = window;
  const w = x1 - x0 + 1;
  return (
    <svg className="gx-arrows" viewBox={`${x0} ${n - 1 - y1} ${w} ${y1 - y0 + 1}`} preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <marker id="gx-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" />
        </marker>
        <marker id="gx-arrow-halo" viewBox="-1 -1 8 8" refX="5" refY="3" markerWidth="3" markerHeight="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" />
        </marker>
      </defs>
      {/* (each over a dark halo, to read on light and dark alike) */}
      <g className="gx-arrow-halo">
        {arrows.map((a) => (
          <line key={a.key} x1={a.x - a.dx / 2} y1={a.y - a.dy / 2} x2={a.x + a.dx / 2} y2={a.y + a.dy / 2} markerEnd="url(#gx-arrow-halo)" />
        ))}
      </g>
      {arrows.map((a) => (
        <line key={a.key} x1={a.x - a.dx / 2} y1={a.y - a.dy / 2} x2={a.x + a.dx / 2} y2={a.y + a.dy / 2} markerEnd="url(#gx-arrow)" />
      ))}
    </svg>
  );
}
