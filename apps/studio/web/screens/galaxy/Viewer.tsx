// The view of a galaxy as it is built (see the server's /views/galaxy and
// /views/galaxy-cube): a map of its light, velocity or dispersion, of all
// its components or of one, as seen, face-on (its motions in its plane as
// arrows) or edge-on; or seen by an observation of the project (see Seen:
// its cube, or its maps, as the observation records them). The map is as
// large as its place, rendered at about its size on the screen; the wheel
// zooms it, a drag pans it, and the pointer reads it (its place on the
// sky, in arcsec from the centre, and its value). Light can be shown on a
// linear, square-root or logarithmic scale.
import { useDebouncedValue, useElementSize } from '@mantine/hooks';
import { useMemo, useRef, useState } from 'react';
import { useGalaxyRender, type GalaxyRender } from '../../api/queries';
import { ColourBar } from '../../components/ColourBar';
import { useStretches } from '../../components/colours';
import { significant } from '../../components/format';
import { GESTURES } from '../../components/MapTools';
import { MapView, type Region } from '../../components/MapView';
import { RegionsBar, useSkyRegions } from '../../components/RegionsBar';
import { IconButton } from '../../components/ui/basics';
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

// A render of the galaxy: its map, as large as its place, with its colour
// bar, and its curves under it
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
  const side = Math.max(160, Math.min(width - 64, height - CURVES_LEAST - CURVES_FRAME - 12));
  const curvesHeight = Math.max(CURVES_LEAST, Math.min(CURVES_MOST, height - side - CURVES_FRAME - 12));
  const key = JSON.stringify([galaxy, properties, orientation]);
  const [settledKey] = useDebouncedValue(key, SETTLE_MS);
  const size = renderSize(side || 400, useMemo(() => hasDepth(galaxy), [galaxy]), settledKey !== key);
  const render = useGalaxyRender(project, galaxy, properties, orientation, size);
  // (the last render that worked stays while the galaxy has a problem)
  const last = useRef<GalaxyRender | undefined>(undefined);
  if (render.data) last.current = render.data;
  const shown = render.data ?? last.current;
  const [scale, setScale] = useState<Scale>('sqrt');
  const [zoom, setZoom] = useState<{ window: Region; size: number } | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [stretch, setStretch] = useStretches(`galaxy/${quantity}`);
  const n = shown?.size ?? size;
  // (a zoom of a render of another size, scaled to this one)
  const window = zoom && shown ? scaleWindow(zoom.window, zoom.size, n) : null;
  // (face-on, the velocity on the line of sight is none: the speed in the
  // plane, under its arrows)
  const arrows = orientation === 'face-on' && quantity === 'velocity';
  const what: Shown = arrows ? 'speed' : quantity;
  const map = useMemo(() => (shown ? mapOf(shown, what, isolated, scale) : null), [shown, what, isolated, scale]);
  const [x0, x1, y0, y1] = window ?? [0, n - 1, 0, n - 1];
  const zoomAt = (x: number, y: number, factor: number) => {
    const w = Math.min(n, Math.max(8, Math.round((x1 - x0 + 1) / factor)));
    if (w >= n) return setZoom(null);
    const left = Math.min(n - w, Math.max(0, Math.round(x - ((x - x0) * w) / (x1 - x0 + 1))));
    const bottom = Math.min(n - w, Math.max(0, Math.round(y - ((y - y0) * w) / (y1 - y0 + 1))));
    setZoom({ window: [left, left + w - 1, bottom, bottom + w - 1], size: n });
  };
  const panBy = (dx: number, dy: number) => {
    if (window === null) return;
    const mx = Math.min(n - 1 - x1, Math.max(-x0, dx));
    const my = Math.min(n - 1 - y1, Math.max(-y0, dy));
    setZoom({ window: [x0 + mx, x1 + mx, y0 + my, y1 + my], size: n });
  };
  const unit = quantity === 'light' ? '' : 'km/s';
  const readout = hover && shown && map ? readAt(shown, map, hover) : null;
  // (the regions of the sky of the project, on the render: its pixels from
  // its middle, where the model measures positions from; not of the galaxy
  // turned face-on or edge-on, where a drag pans)
  const sky = orientation === 'sky';
  const frame = useMemo(() => (sky && shown ? { step: [shown.step, shown.step] as [number, number], rpix: [(n - 1) / 2, (n - 1) / 2] as [number, number] } : null), [sky, shown, n]);
  const regions = useSkyRegions(sky ? project : null, frame, [n, n]);
  const region = sky ? (regions.active?.pixels ?? null) : null;
  return (
    <div className="gx-render" ref={column}>
    <div className="gx-map-pane" style={{ ['--side' as string]: `${side}px` }}>
      <div className="gx-map">
        {shown && map ? (
          <>
            <MapView
              values={map.shown}
              shape={[n, n]}
              range={map.range}
              diverging={what === 'velocity'}
              window={window}
              stretch={stretch}
              onStretch={setStretch}
              onZoom={zoomAt}
              onPan={panBy}
              onWhole={() => setZoom(null)}
              region={region}
              regions={sky ? regions.placed.filter((r) => r.pixels && r.id !== regions.active?.id).map((r) => r.pixels!) : []}
              onRegion={sky ? regions.add : undefined}
              onHover={setHover}
              label={`The ${quantity} of the galaxy`}
            />
            {arrows && <Arrows render={shown} component={isolated} window={[x0, x1, y0, y1]} />}
            {sky && (
              <div className="map-regions-over">
                <RegionsBar regions={regions} values={map.values} nx={n} unit={unit} />
              </div>
            )}
            <div className="gx-map-corner">
              {render.isFetching && <span className="gx-busy" title="Rendering" />}
              {window !== null && (
                <IconButton label="See all of it" onClick={() => setZoom(null)}>
                  <svg className="icon" viewBox="0 0 24 24">
                    <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
                  </svg>
                </IconButton>
              )}
            </div>
            <div className="gx-readout" aria-live="off" title={GESTURES}>
              {readout ? (
                <>
                  <span>x {signed(readout.x)}″</span>
                  <span>y {signed(readout.y)}″</span>
                  <b>{readout.value === null ? '—' : `${significant(readout.value, 4)}${unit ? ` ${unit}` : ''}`}</b>
                </>
              ) : (
                <span className="gx-readout-hint">{scaleBar(shown, x1 - x0 + 1)}</span>
              )}
            </div>
            {!map.any && <p className="gx-map-note">{quantity === 'light' ? 'No light here' : `No ${quantity} here: it needs motions, and light to weigh them`}</p>}
            {arrows && map.any && <span className="gx-map-tag">speed in the plane</span>}
          </>
        ) : render.error ? null : (
          <div className="gx-map-wait" />
        )}
      </div>
      <div className="map-side" style={{ height: 'var(--side)' }}>
        {map && map.any && <ColourBar range={map.bar} diverging={what === 'velocity'} unit={unit} stretch={stretch} onReset={() => setStretch({ middle: 0.5, width: 1 })} />}
        {quantity === 'light' && (
          <div className="gx-scale" role="group" aria-label="The scale of the light">
            {(['linear', 'sqrt', 'log'] as Scale[]).map((s) => (
              <button key={s} data-on={scale === s || undefined} onClick={() => setScale(s)} title={{ linear: 'Linear', sqrt: 'Square root', log: 'Logarithmic' }[s]}>
                {{ linear: 'lin', sqrt: '√', log: 'log' }[s]}
              </button>
            ))}
          </div>
        )}
      </div>
      {render.error && <p className="gx-error">{render.error.message}</p>}
    </div>
    {shown && <Curves render={shown} focus={curves} components={components} height={curvesHeight} />}
    </div>
  );
}

// A window of pixels of a render of one size, in those of another
function scaleWindow(window: Region, from: number, to: number): Region {
  if (from === to) return window;
  const f = to / from;
  const [x0, x1, y0] = window;
  const w = Math.max(8, Math.round((x1 - x0 + 1) * f));
  const left = Math.min(to - w, Math.max(0, Math.round(x0 * f)));
  const bottom = Math.min(to - w, Math.max(0, Math.round(y0 * f)));
  return [left, left + w - 1, bottom, bottom + w - 1];
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

// The place on the sky (arcsec from the centre of the render) and the value
// of a pixel
function readAt(render: GalaxyRender, map: ShownMap, [x, y]: [number, number]) {
  const middle = (render.size - 1) / 2;
  const value = map.values[y * render.size + x];
  return { x: (x - middle) * render.step, y: (y - middle) * render.step, value: Number.isFinite(value) ? value : null };
}

function signed(value: number): string {
  const text = significant(Math.abs(value), 3);
  return value < -1e-9 ? `−${text}` : text;
}

// The width of the window, as a hint where the pointer is not
function scaleBar(render: GalaxyRender | undefined, pixels: number): string {
  if (!render) return '';
  return `${significant(pixels * render.step, 3)}″ across`;
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
