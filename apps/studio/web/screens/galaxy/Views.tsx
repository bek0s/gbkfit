// The picture of a galaxy: as many views as wanted, arranged freely (see
// Dockview: each a panel that can be dragged beside, under or into
// another, resized and maximised), or by a preset of 1, 2 or 4; the
// arrangement is kept for the next time. Each view picks what it shows
// from a menu: the galaxy through an observation of the project (as it
// records it), the galaxy itself (as seen, face-on or edge-on: its light,
// velocity or dispersion; its profiles along the radius; its mass; its
// light in 3d, as slices or a volume to turn), and the extra outputs of
// the library's evaluation (see the library's Extras), each with its title
// and what it is, as the server lists them. The arrangement is a preference
// of the user (see preferences: the same in every browser).
import { useIsFetching } from '@tanstack/react-query';
import { DockviewReact, type DockviewApi, type DockviewReadyEvent, type IDockviewPanelHeaderProps, type IDockviewPanelProps } from 'dockview-react';
import 'dockview-react/dist/styles/dockview.css';
import { createContext, useContext, useMemo, useRef, type ReactNode } from 'react';
import { useGalaxyCube, useGalaxyExtra, useGalaxyRender, useGalaxyVolume, usePV, type GalaxyExtraInfo, type GalaxyRender } from '../../api/queries';
import { MapPanel } from '../../components/MapPanel';
import { Choice } from '../../components/ui/inputs';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import { CurvePlot, massCurves } from './Curves';
import { componentsOf, type Options } from './options';
import { settingOf, type Param } from './params';
import { PVPlot } from './PV';
import { Profiles, PROFILE_QUANTITIES, type ProfileQuantity } from './Profiles';
import { SeenView } from './Seen';
import { hasHeight, RenderMap, type Orientation, type Quantity } from './Viewer';
import { Slices, VolumeView } from './Volume';
import { preference, setPreference } from '../../shell/preferences';

// What a view shows
export type ViewKind =
  | { kind: 'seen' }
  | { kind: 'map'; orientation: Orientation; quantity: Quantity }
  | { kind: 'profiles'; quantity: ProfileQuantity }
  | { kind: 'mass' }
  | { kind: 'slices' }
  | { kind: '3d' }
  | { kind: 'pv'; axis: 'major' | 'minor' }
  | { kind: 'extra'; name: string; ids: Record<string, string>; title: string };

// What the views share: the galaxy as it settles, what sees it, its
// components and their colours, its parameters
interface ViewsData {
  project: string;
  galaxy: Options | undefined;
  properties: Record<string, unknown>;
  why: 'empty' | 'problems' | 'waiting';
  components: { name: string; colour: string }[];
  isolated: string | null;
  observations: string[];
  seenBy: string | null;
  options: Options;
  // The parameters of the galaxy (e.g. for the axes of its main component)
  params: Param[];
  kpcPerArcsec: number | null;
  rings: number[] | null;
  // The render of the galaxy as seen, small: its curves, and the list of
  // the extra outputs of its evaluation
  render: GalaxyRender | undefined;
}

const ViewsContext = createContext<ViewsData | null>(null);

function useViews(): ViewsData {
  const data = useContext(ViewsContext);
  if (data === null) throw new Error('a view is used out of Views');
  return data;
}

const LAYOUT_KEY = 'galaxy-views';
const isLayout = (value: unknown): value is object => typeof value === 'object' && value !== null;

// The views of a preset of a number of them: what sees the galaxy (or its
// velocity as seen), its profiles, edge-on, slices of its light in 3d
function presetKinds(seen: boolean): ViewKind[] {
  return [
    seen ? { kind: 'seen' } : { kind: 'map', orientation: 'sky', quantity: 'velocity' },
    { kind: 'profiles', quantity: 'speed' },
    { kind: 'map', orientation: 'edge-on', quantity: 'light' },
    { kind: 'slices' },
  ];
}

let panelCount = 0;
function panelId(): string {
  panelCount += 1;
  return `view-${Date.now().toString(36)}-${panelCount}`;
}

// The views as a preset: 1, 2 side by side, or 4 in a square
function applyPreset(api: DockviewApi, count: 1 | 2 | 4, seen: boolean) {
  const kinds = presetKinds(seen);
  api.clear();
  const add = (kind: ViewKind, position?: { referencePanel: string; direction: 'right' | 'below' }) => {
    const id = panelId();
    api.addPanel({ id, component: 'view', params: { view: kind }, title: titleOf(kind).title, ...(position ? { position } : {}) });
    return id;
  };
  const first = add(kinds[0]);
  if (count === 1) return;
  const second = add(kinds[1], { referencePanel: first, direction: 'right' });
  if (count === 2) return;
  add(kinds[2], { referencePanel: first, direction: 'below' });
  add(kinds[3], { referencePanel: second, direction: 'below' });
}

export interface ViewsProps extends Omit<ViewsData, 'render'> {
  onSeenBy: (observation: string | null) => void;
}

export function Views(props: ViewsProps) {
  const { project, galaxy, properties, seenBy, observations, onSeenBy } = props;
  const seen = seenBy !== null && observations.includes(seenBy) ? seenBy : null;
  const render = useGalaxyRender(project, galaxy, properties, 'sky', 64).data;
  const data = useMemo<ViewsData>(() => ({ ...props, seenBy: seen, render }), [props, seen, render]);
  const api = useRef<DockviewApi | null>(null);
  const fetching = useIsFetching({ queryKey: ['view'] });
  const onReady = (event: DockviewReadyEvent) => {
    api.current = event.api;
    const saved = preference<object | null>(LAYOUT_KEY, isLayout, null);
    try {
      if (saved) event.api.fromJSON(saved as Parameters<DockviewApi['fromJSON']>[0]);
      else applyPreset(event.api, 4, seen !== null);
    } catch {
      applyPreset(event.api, 4, seen !== null);
    }
    if (event.api.panels.length === 0) applyPreset(event.api, 4, seen !== null);
    // (kept as it settles: a drag changes it many times)
    let timer: ReturnType<typeof setTimeout> | undefined;
    event.api.onDidLayoutChange(() => {
      clearTimeout(timer);
      timer = setTimeout(() => setPreference(LAYOUT_KEY, event.api.toJSON()), 500);
    });
  };
  return (
    <ViewsContext.Provider value={data}>
      <div className="gv">
        <div className="gv-tool">
          <span className="gv-lab">Views</span>
          <div className="gv-mini" role="group" aria-label="A preset of views">
            {([1, 2, 4] as const).map((n) => (
              <button key={n} onClick={() => api.current && applyPreset(api.current, n, seen !== null)} title={`${n} ${n === 1 ? 'view' : 'views'}`}>
                {n}
              </button>
            ))}
          </div>
          <button
            className="gv-add"
            onClick={() => {
              const dock = api.current;
              if (!dock) return;
              const kind: ViewKind = { kind: 'map', orientation: 'face-on', quantity: 'light' };
              const last = dock.panels.at(-1);
              dock.addPanel({ id: panelId(), component: 'view', params: { view: kind }, title: titleOf(kind).title, ...(last ? { position: { referencePanel: last.id, direction: 'right' as const } } : {}) });
            }}
            title="Add a view (drag its head to place it)"
          >
            + view
          </button>
          <span className="gv-lab">seen by</span>
          <div className="gv-seen">
            <Choice
              items={[{ value: '', label: 'nothing: the galaxy itself' }, ...observations.map((name) => ({ value: name, label: name }))]}
              value={seen ?? ''}
              onChange={(next) => onSeenBy(next === '' ? null : next)}
              label="The observation that the views of what it sees take"
            />
          </div>
          <span className="gv-status">{fetching > 0 ? 'rendering…' : galaxy ? 'rendered' : ''}</span>
        </div>
        {galaxy === undefined ? (
          <div className="gv-empty">{props.why === 'waiting' ? <div className="gx-map-wait" /> : <EmptyNote why={props.why} />}</div>
        ) : (
          <DockviewReact
            className="gv-dock"
            theme={{ name: 'studio', className: 'dockview-theme-studio', gap: 8 }}
            components={{ view: ViewBody }}
            defaultTabComponent={ViewHead}
            singleTabMode="fullwidth"
            onReady={onReady}
          />
        )}
      </div>
    </ViewsContext.Provider>
  );
}

function EmptyNote({ why }: { why: 'empty' | 'problems' }) {
  return (
    <div className="gx-empty">
      <p>{why === 'empty' ? 'An empty galaxy.' : 'Nothing to show.'}</p>
      <p className="note">{why === 'empty' ? 'Add its first component in the model to see it.' : 'The library cannot make the galaxy: its problems are in the model.'}</p>
    </div>
  );
}

// The title of a view, and what it is under it
function titleOf(kind: ViewKind, seenBy: string | null = null): { title: string; small: string } {
  switch (kind.kind) {
    case 'seen':
      return { title: 'As observed', small: seenBy ? `by ${seenBy}` : 'by no observation' };
    case 'map':
      return {
        title: kind.orientation === 'sky' ? 'As seen' : kind.orientation === 'face-on' ? 'Face-on' : 'Edge-on',
        small: `${kind.quantity}, the galaxy itself`,
      };
    case 'profiles':
      return { title: 'Profiles', small: 'along the radius' };
    case 'mass':
      return { title: 'Mass model', small: 'circular speeds of each part' };
    case 'slices':
      return { title: 'Slices of the light', small: '3d, along z' };
    case '3d':
      return { title: 'In 3d', small: 'the light, to turn' };
    case 'pv':
      return { title: 'Position–velocity', small: `along its ${kind.axis} axis` };
    case 'extra':
      return { title: kind.title, small: 'an output of the library' };
  }
}

// The menu of what a view shows: through the observation, the galaxy
// itself, the extra outputs of the library
function viewEntries(current: ViewKind, data: ViewsData, choose: (kind: ViewKind) => void): MenuEntry[] {
  const same = (kind: ViewKind) => JSON.stringify(kind) === JSON.stringify(current);
  const entry = (label: string, description: string, kind: ViewKind, disabled = false): MenuEntry => ({
    label,
    description,
    current: same(kind),
    disabled,
    onClick: () => choose(kind),
  });
  const flat = data.galaxy !== undefined && !hasHeight(data.galaxy);
  const extras = (data.render?.extras ?? []).filter((x) => x.ndim >= 1 && x.ndim <= 3);
  return [
    { section: data.seenBy ? `Through ${data.seenBy}` : 'Through an observation' },
    entry('As observed', data.seenBy ? 'Its cube or maps, as the observation records them: channels, spectra, moments' : 'Choose what sees the galaxy, in the bar above', { kind: 'seen' }, !data.seenBy),
    { section: 'The galaxy itself' },
    entry('As seen', 'Light, velocity and dispersion, at the orientation of its parameters', { kind: 'map', orientation: 'sky', quantity: 'light' }),
    entry('Face-on', 'Light, velocity (its motions in its plane) and dispersion', { kind: 'map', orientation: 'face-on', quantity: 'light' }),
    entry('Edge-on', flat ? 'Its disks are thin: edge-on, a line' : 'Light, velocity and dispersion', { kind: 'map', orientation: 'edge-on', quantity: 'light' }, flat),
    entry('Profiles', 'Every trait along the radius: speeds, dispersion, density, orientation', { kind: 'profiles', quantity: 'speed' }),
    entry('Mass model', 'The circular speeds of its mass and of each part', { kind: 'mass' }),
    entry('Slices of the light', 'Its light in 3d, at heights above its plane', { kind: 'slices' }),
    entry('In 3d', 'Its light as a volume to turn', { kind: '3d' }),
    entry('Position–velocity', 'Its cube along a slit through its centre, on the major or the minor axis of its main component, with its rotation over it', { kind: 'pv', axis: 'major' }),
    ...(extras.length ? [{ section: 'Extra outputs of the library' } as MenuEntry] : []),
    ...extras.map((x) => entry(extraLabel(x), x.description, { kind: 'extra', name: x.name, ids: x.ids, title: extraLabel(x) })),
  ];
}

// An extra output by its title and what it is of (e.g. its component)
function extraLabel(extra: GalaxyExtraInfo): string {
  const of = extra.ids.component ?? (extra.name.startsWith('total_') ? 'all' : null);
  return of ? `${extra.title} · ${of}` : extra.title;
}

// The head of a view: what it shows (a menu), the switches of its kind
// (e.g. light, velocity, dispersion) and what maximises it
function ViewHead(props: IDockviewPanelHeaderProps<{ view: ViewKind }>) {
  const data = useViews();
  const kind = props.params.view;
  const choose = (next: ViewKind) => {
    props.api.updateParameters({ view: next });
    props.api.setTitle(titleOf(next).title);
    // (what a view shows is of the arrangement kept, but not a change of
    // its layout: kept here)
    setPreference(LAYOUT_KEY, props.containerApi.toJSON());
  };
  const { title, small } = titleOf(kind, data.seenBy);
  let switches: ReactNode = null;
  if (kind.kind === 'map') {
    switches = (['light', 'velocity', 'dispersion'] as Quantity[]).map((q) => (
      <button key={q} data-on={kind.quantity === q || undefined} onClick={() => choose({ ...kind, quantity: q })}>
        {q[0].toUpperCase() + q.slice(1)}
      </button>
    ));
  } else if (kind.kind === 'pv') {
    switches = (['major', 'minor'] as const).map((axis) => (
      <button key={axis} data-on={kind.axis === axis || undefined} onClick={() => choose({ ...kind, axis })}>
        {axis === 'major' ? 'Major axis' : 'Minor axis'}
      </button>
    ));
  } else if (kind.kind === 'profiles') {
    switches = PROFILE_QUANTITIES.map((q) => (
      <button key={q.value} data-on={kind.quantity === q.value || undefined} onClick={() => choose({ ...kind, quantity: q.value })}>
        {q.label}
      </button>
    ));
  }
  return (
    <div className="gv-head" onPointerDown={(event) => (event.target as HTMLElement).closest('button') && event.stopPropagation()}>
      <DropMenu
        label="What the view shows"
        width={340}
        entries={viewEntries(kind, data, choose)}
        target={
          <button className="gv-pick">
            {title} <small>{small}</small>
          </button>
        }
      />
      <span className="gv-switches">{switches}</span>
      <button className="gv-x" title={props.api.isMaximized() ? 'Back to the views' : 'Maximise it'} onClick={() => (props.api.isMaximized() ? props.api.exitMaximized() : props.api.maximize())}>
        ⤢
      </button>
      <button className="gv-x" title="Close it" onClick={() => props.api.close()}>
        ×
      </button>
    </div>
  );
}

// The body of a view: what it shows
function ViewBody(props: IDockviewPanelProps<{ view: ViewKind }>) {
  const data = useViews();
  const kind = props.params.view;
  return <div className="gv-body">{data.galaxy && <ViewContent kind={kind} data={data} galaxy={data.galaxy} />}</div>;
}

function ViewContent({ kind, data, galaxy }: { kind: ViewKind; data: ViewsData; galaxy: Options }) {
  const colours = new Map(data.components.map((c) => [c.name, c.colour]));
  switch (kind.kind) {
    case 'seen':
      return data.seenBy ? (
        <SeenView project={data.project} galaxy={galaxy} properties={data.properties} observation={data.seenBy} />
      ) : (
        <p className="note gv-note">Choose what sees the galaxy, in the bar above.</p>
      );
    case 'map':
      return <RenderMap project={data.project} galaxy={galaxy} properties={data.properties} orientation={kind.orientation} quantity={kind.quantity} isolated={data.isolated} />;
    case 'profiles':
      return <Profiles render={data.render} quantity={kind.quantity} options={data.options} colours={colours} rings={data.rings} />;
    case 'mass':
      return data.render ? (
        <CurvePlot curves={massCurves(data.render, colours)} radius={data.render.curves.radius ?? []} rings={data.rings} unit="km/s" empty="Add mass to the galaxy to see its circular speeds." />
      ) : (
        <div className="gx-map-wait" />
      );
    case 'slices':
      return <SlicesView data={data} galaxy={galaxy} />;
    case '3d':
      return <Volume3DView data={data} galaxy={galaxy} />;
    case 'pv':
      return <PVView data={data} galaxy={galaxy} axis={kind.axis} />;
    case 'extra':
      return <ExtraView data={data} galaxy={galaxy} extra={kind} />;
  }
}

// The position-velocity diagram of the galaxy (see PVPlot): of its cube,
// as seen, along a slit through the centre of its main component (its
// first disk, else its first component with a position angle), on its
// major axis (its position angle; their mean, of one per ring) or its
// minor axis; on the major axis, with the line of sight velocity of its
// rotation over it
function PVView({ data, galaxy, axis }: { data: ViewsData; galaxy: Options; axis: 'major' | 'minor' }) {
  const cube = useGalaxyCube(data.project, galaxy, data.properties, 'sky', 128);
  const main = mainAxes(data, galaxy);
  const step = cube.data?.step;
  const [nx, ny] = cube.data?.shape ?? [0, 0];
  const centre: [number, number] | null = main && step ? [(nx - 1) / 2 + main.xpos / step, (ny - 1) / 2 + main.ypos / step] : null;
  const pa = (main?.posa ?? 0) + (axis === 'minor' ? 90 : 0);
  const pv = usePV(cube.data?.path, 0, pa, centre);
  const error = cube.error ?? pv.error;
  if (error) return <p className="gx-error gv-note">{error.message}</p>;
  if (!pv.data) return <div className="gx-map-wait" />;
  // (the rotation of the main component along the major axis, on the line
  // of sight, positive positions on its receding side; at the velocities
  // of the channels where it lands, see onAxis)
  const curves = data.render?.curves;
  const rotation = main && curves ? curves[main.suffix ? `rotation_${main.suffix}` : 'rotation'] : undefined;
  const sine = Math.sin(((main?.incl ?? 0) * Math.PI) / 180);
  const cubeAxis = cube.data;
  const model =
    axis === 'major' && main && rotation && curves?.radius && cubeAxis
      ? curves.radius
          .flatMap((r, i) => {
            const v = rotation[i];
            return r === null || v === null ? [] : [-1, 1].map((side) => ({ position: side * r, velocity: onAxis(main.vsys + side * v * sine, cubeAxis) }));
          })
          .sort((a, b) => a.position - b.position)
      : null;
  return <PVPlot pv={pv.data} model={model} />;
}

// The velocity of the channels of a cube of a galaxy where a velocity of
// the galaxy lands (see the server's CubeAxis): of the Doppler factor
// shift D(v), in the convention of the channels
function onAxis(velocity: number, axis: { convention: 'optical' | 'radio' | 'relativistic'; shift: number }): number {
  const c = 299792.458;
  const factor = axis.shift * Math.sqrt((1 + velocity / c) / (1 - velocity / c));
  if (axis.convention === 'optical') return c * (factor - 1);
  if (axis.convention === 'radio') return c * (1 - 1 / factor);
  return (c * (factor ** 2 - 1)) / (factor ** 2 + 1);
}

// The axes of the main component of a galaxy (its first disk, else its
// first component with a position angle): its centre, position angle,
// inclination and systemic velocity (the mean of the values of each, of
// one per ring), and the suffix of its functions (its name, if named)
function mainAxes(data: ViewsData, galaxy: Options): { xpos: number; ypos: number; posa: number; incl: number; vsys: number; suffix: string } | null {
  const components = componentsOf(galaxy);
  const own = (index: number, key: string) => data.params.find((p) => p.owner.length === 2 && p.owner[0] === 'components' && p.owner[1] === index && (p.name === key || p.name.endsWith(`_${key}`)));
  const disks = components.map((_, i) => i).filter((i) => String(components[i].type).endsWith('disk'));
  const index = [...disks, ...components.map((_, i) => i)].find((i) => own(i, 'posa') !== undefined);
  if (index === undefined) return null;
  const value = (key: string) => {
    const param = own(index, key);
    if (!param) return 0;
    const setting = settingOf(data.properties, param);
    const values = setting.kind === 'fixed' || setting.kind === 'free' ? setting.values.filter(Number.isFinite) : [];
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : (param.default ?? 0);
  };
  const name = components[index].name;
  return { xpos: value('xpos'), ypos: value('ypos'), posa: value('posa'), incl: value('incl'), vsys: value('vsys'), suffix: typeof name === 'string' ? name : '' };
}

function SlicesView({ data, galaxy }: { data: ViewsData; galaxy: Options }) {
  const volume = useGalaxyVolume(data.project, galaxy, data.properties, 96);
  return <Slices volume={volume.data} error={volume.error} kpcPerArcsec={data.kpcPerArcsec} />;
}

function Volume3DView({ data, galaxy }: { data: ViewsData; galaxy: Options }) {
  const volume = useGalaxyVolume(data.project, galaxy, data.properties, 96);
  return <VolumeView volume={volume.data} error={volume.error} />;
}

// An extra output of the library: a map (of an image, or of a cube along
// its depth, summed), or a curve
function ExtraView({ data, galaxy, extra }: { data: ViewsData; galaxy: Options; extra: { name: string; ids: Record<string, string> } }) {
  const rendered = useGalaxyExtra(data.project, galaxy, data.properties, extra, 'sky', 128);
  const shown = useMemo(() => {
    const e = rendered.data;
    if (!e) return null;
    if (e.shape.length === 1) return { kind: 'curve' as const, values: Array.from(e.values) };
    const [ny, nx] = e.shape.slice(-2);
    let values = e.values;
    if (e.shape.length === 3) {
      // (a cube with a depth axis: summed along it, as seen)
      const nz = e.shape[0];
      values = new Float32Array(nx * ny);
      for (let z = 0; z < nz; z++) for (let i = 0; i < nx * ny; i++) values[i] += Number.isFinite(e.values[z * nx * ny + i]) ? e.values[z * nx * ny + i] : 0;
    }
    let low = Infinity;
    let high = -Infinity;
    for (const v of values) if (Number.isFinite(v)) [low, high] = [Math.min(low, v), Math.max(high, v)];
    return { kind: 'map' as const, values, nx, ny, range: (low <= high ? [low, high === low ? low + 1 : high] : [0, 1]) as [number, number], unit: e.unit, step: e.step };
  }, [rendered.data]);
  if (rendered.error) return <p className="gx-error gv-note">{rendered.error.message}</p>;
  if (!shown) return <div className="gx-map-wait" />;
  if (shown.kind === 'curve') {
    return <CurvePlot curves={[{ name: 'extra', label: rendered.data?.title ?? '', colour: 'accent', values: shown.values }]} radius={shown.values.map((_, i) => i)} unit={rendered.data?.unit ?? ''} empty="No values." />;
  }
  return (
    <div className="gx-render">
      <MapPanel
        values={shown.values}
        shape={[shown.ny, shown.nx]}
        range={shown.range}
        diverging={shown.unit.replaceAll(' ', '') === 'km/s' && extra.name.startsWith('v')}
        unit={shown.unit}
        stretchKey={`extra/${extra.name}`}
        frame={{ step: [shown.step, shown.step], rpix: [(shown.nx - 1) / 2, (shown.ny - 1) / 2] }}
        pixels={false}
        smooth
        label={rendered.data?.title ?? extra.name}
        note={<div className="gx-map-wait" />}
      />
    </div>
  );
}
