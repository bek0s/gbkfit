// The screen of a galaxy, built as it is seen: a short list of what it is
// made of (the galaxy as a whole: its distance, mass and gravity; and its
// components, each of its parts: geometry, density, motions, light, ...), the view
// of the galaxy (its maps, face-on, edge-on or as seen, and its curves),
// and the part chosen in the list with its few parameters as sliders.
// Only what has been added shows; the rest is added from the list. The
// values of the sliders are the starting values of the parameters, kept
// in the params block of the same name as the galaxy.
import { useDebouncedValue } from '@mantine/hooks';
import { useMemo, useRef, useState } from 'react';
import { useCreateBlock, useGalaxyCube, useGalaxyParameters, useGalaxyRender, useLines, usePiece, useUpdateBlock, type GalaxyRender } from '../api/queries';
import { CubeViewer } from '../components/CubeViewer';
import type { Region } from '../components/MapView';
import { joinPath } from '../components/FilePicker';
import { GalaxyCurves, GalaxyMap, type Quantity } from '../components/GalaxyView';
import { significant } from '../components/format';
import { membersOf, resolve, typeOf, type Defs, type Options, type Schema } from '../components/SchemaForm';
import { Button, Close, TextField } from '../components/ui/basics';
import { Check, Choice, NumberField, Segments } from '../components/ui/inputs';
import { DropMenu } from '../components/ui/overlays';
import { StepSlider } from '../components/ui/sliders';
import { useBlockHead } from '../shell/BlockMenu';
import { useStudio } from '../studio';
import { useEditedBlock, useUnsaved } from './editing';

type Path = (string | number)[];

// A parameter as the library describes it (see the server's /parameters):
// its name, size (node-wise: one value for each ring), default, unit,
// valid range (e.g. '[0, inf)'), period, and the option that makes it
// (e.g. ['components', 0, 'kinematics', 'vptraits', 1])
interface Param {
  name: string;
  size: number;
  default: number | null;
  unit: string | null;
  range: string;
  period: number | null;
  owner: Path;
}

type View = 'sky' | 'face-on' | 'edge-on';

// --- Options --------------------------------------------------------------

// A value as a list: a list, one item, or none
function asList(value: unknown): Options[] {
  if (Array.isArray(value)) return value as Options[];
  return value && typeof value === 'object' ? [value as Options] : [];
}

// The value at a path of options (undefined if none)
function getAt(options: unknown, path: Path): unknown {
  return path.reduce<unknown>((value, key) => (value && typeof value === 'object' ? (value as Record<string, unknown>)[key as string] : undefined), options);
}

// Options with the value at a path replaced (undefined: removed), the
// rest as it is
function setAt(options: Options, path: Path, value: unknown): Options {
  const [key, ...rest] = path;
  if (Array.isArray(options)) {
    const copy = [...(options as unknown[])];
    copy[key as number] = rest.length ? setAt((copy[key as number] ?? {}) as Options, rest, value) : value;
    return copy.filter((item) => item !== undefined) as unknown as Options;
  }
  const copy: Options = { ...options };
  const next = rest.length ? setAt((copy[key as string] ?? (typeof rest[0] === 'number' ? [] : {})) as Options, rest, value) : value;
  if (next === undefined) delete copy[key as string];
  else copy[key as string] = next;
  return copy;
}

// The list of a part that is one item or a list (e.g. ptraits), with an
// item replaced, added or removed: one item stays one item, several a list
function withItems(items: Options[]): Options | Options[] | undefined {
  return items.length === 0 ? undefined : items.length === 1 ? items[0] : items;
}

// --- What a galaxy is made of ---------------------------------------------

// What a new component of each type starts with: the least that shows
// (each needs light; a disk of a line moves)
const STARTS: Record<string, Options> = {
  disk: {
    rnodes: { max: 16, sep: 2 },
    density: { ptraits: { type: 'exponential' } },
    kinematics: { type: 'traits', vptraits: { type: 'arctan', direction: 'tan' }, dptraits: { type: 'uniform' } },
    emission: { type: 'lines' },
  },
  cone: {
    rnodes: { max: 6, sep: 1 },
    density: { ptraits: { type: 'uniform' }, htraits: { type: 'gauss', zmax: 2 } },
    kinematics: { type: 'traits', vptraits: { type: 'uniform', direction: 'out' }, dptraits: { type: 'uniform' } },
    emission: { type: 'lines' },
  },
  spheroid: { rmax: 10, profile: { type: 'sersic' }, emission: { type: 'band' } },
  point: { emission: { type: 'band' } },
};

// A part of a component (or of the galaxy as a whole): its title, the
// option that holds it (its parameters are those it makes), the family of
// its items in the schema if it is a list of traits (e.g. 'vpt'), and
// whether it can be added or removed
interface PartKind {
  key: string;
  title: string;
  option: Path;
  family?: string;
  // The family of its one typed option (e.g. the emission)
  typed?: string;
  optional?: boolean;
  // What an added one starts with
  start?: unknown;
}

const DISK_PARTS: PartKind[] = [
  { key: 'position', title: 'Geometry', option: [] },
  { key: 'density', title: 'Density', option: ['density', 'ptraits'], family: 'rpt' },
  { key: 'thickness', title: 'Thickness', option: ['density', 'htraits'], family: 'rht', optional: true, start: { type: 'sech2', zmax: 6 } },
  { key: 'motions', title: 'Motions', option: ['kinematics', 'vptraits'], family: 'vpt' },
  { key: 'dispersion', title: 'Dispersion', option: ['kinematics', 'dptraits'], family: 'dpt' },
  { key: 'light', title: 'Light', option: ['emission'], typed: 'emission' },
  { key: 'opacity', title: 'Opacity', option: ['opacity'], typed: 'opacity', optional: true, start: { type: 'dust' } },
];

const PARTS: Record<string, PartKind[]> = {
  disk: DISK_PARTS,
  cone: DISK_PARTS.map((part) => (part.key === 'thickness' ? { ...part, title: 'Walls', optional: false } : part)),
  spheroid: [
    { key: 'position', title: 'Geometry', option: [] },
    { key: 'profile', title: 'Profile', option: ['profile'], typed: 'light_profile' },
    { key: 'shape', title: 'Shape', option: ['shape'], typed: 'spheroid_shape', optional: true, start: { type: 'flattened' } },
    { key: 'motions', title: 'Motions', option: ['kinematics'], typed: 'spheroid_kinematics', optional: true, start: { type: 'jeans' } },
    { key: 'light', title: 'Light', option: ['emission'], typed: 'emission' },
  ],
  point: [
    { key: 'position', title: 'Position', option: [] },
    { key: 'light', title: 'Light', option: ['emission'], typed: 'emission' },
  ],
  particles: [
    { key: 'position', title: 'Position', option: [] },
    { key: 'light', title: 'Light', option: ['emission'], typed: 'emission' },
  ],
};

const GALAXY_PARTS: PartKind[] = [
  { key: 'distance', title: 'Distance', option: ['distance'], typed: 'distance', optional: true, start: { type: 'direct' } },
  { key: 'mass', title: 'Mass', option: ['mass'], family: 'mass_component', optional: true },
  { key: 'gravity', title: 'Gravity', option: ['gravity'], typed: 'gravity', optional: true, start: { type: 'newton' } },
];

// The place of a part: of a component (by index) or of the galaxy (null)
interface Selected {
  component: number | null;
  part: string;
}

// What a part is at a glance: the types of its items, or of its option
function glance(value: unknown): string {
  const items = asList(value);
  if (items.length === 0) return '';
  return items.map((item) => (typeof item.name === 'string' && typeof item.type === 'string' ? `${item.name}` : String(item.type ?? ''))).join(', ');
}

// The components of a galaxy, and the galaxy with others
function componentsOf(options: Options): Options[] {
  return asList(options.components);
}

// A name for a new component of a type, not taken
function freshName(components: Options[], type: string): string {
  const taken = new Set(components.map((c) => c.name));
  for (let i = 1; ; i++) {
    const name = i === 1 ? type : `${type}${i}`;
    if (!taken.has(name)) return name;
  }
}

// --- Parameters -------------------------------------------------------------

// Whether a path starts with another
function startsWith(path: Path, start: Path): boolean {
  return start.every((key, i) => path[i] === key);
}

// The parameters of a part (those its option makes; of the position of a
// component, those of the component itself), in the library's order
function paramsOf(params: Param[], component: number | null, part: PartKind): Param[] {
  const base = component === null ? [] : ['components', component];
  const option = [...base, ...part.option];
  return params.filter((p) => (part.key === 'position' ? p.owner.length === option.length && startsWith(p.owner, option) : startsWith(p.owner, option)));
}

// The range of a parameter, of its text (e.g. '[0, 90]', '(0, inf)')
function rangeOf(param: Param): [number, number] {
  const match = /^[[(]\s*([^,]+),\s*([^\])]+)[\])]$/.exec(param.range);
  const read = (text: string | undefined, infinite: number) => {
    const value = Number(text?.replace('inf', 'Infinity'));
    return Number.isFinite(value) ? value : infinite;
  };
  return [read(match?.[1], -Infinity), read(match?.[2], Infinity)];
}

// The slider of a parameter: its valid range where it has one; else a
// reach of its unit (an angle, a velocity, a radius within the field), or
// about its default. It never narrows the valid range: its own reach
// stops at the bounds the library gives (see studio-library-ranges)
function sliderOf(param: Param, field: number): { min: number; max: number; step: number } {
  const [low, high] = rangeOf(param);
  const value = Math.abs(param.default ?? 1) || 1;
  const reach: [number, number] =
    param.unit === 'deg'
      ? [0, param.period ?? 90]
      : param.unit === 'km / s'
        ? [param.name.endsWith('vsys') ? -500 : low >= 0 ? 0 : -400, param.name.endsWith('vsys') ? 500 : 400]
        : param.unit === 'arcsec'
          ? [low >= 0 ? 0 : -field, field]
          : [low >= 0 ? 0 : -10 * value, 10 * value];
  const min = Number.isFinite(low) ? low : reach[0];
  const max = Number.isFinite(high) ? high : Math.max(reach[1], min + 1);
  const span = max - min;
  const step = 10 ** Math.floor(Math.log10(span / 200));
  return { min, max, step };
}

// The name of a parameter in its part: without the name of its component
// and the prefix of its trait (e.g. 'vt' of 'disk_vpt1_vt'), or of its mass
// component (e.g. 'log_m' of 'mass_halo_log_m')
function shortName(param: Param, options: Options): string {
  const [section, index] = param.owner;
  let rest = param.name;
  if (section === 'components' && typeof index === 'number') {
    const name = getAt(options, ['components', index, 'name']);
    if (typeof name === 'string' && rest.startsWith(`${name}_`)) rest = rest.slice(name.length + 1);
    // (of a trait, its prefix: its kind and position, or its name; of a
    // line, its name stays: 'nii6583 ratio')
    if (param.owner[2] === 'emission') rest = rest.replace(/_ratio$/, ' ratio');
    else if (typeof param.owner[param.owner.length - 1] === 'number' && param.owner.length > 3) rest = rest.slice(rest.indexOf('_') + 1);
  } else if (section === 'mass' && typeof index === 'number') {
    const name = getAt(options, ['mass', index, 'name']) ?? getAt(options, ['mass', 'name']);
    if (typeof name === 'string' && rest.startsWith(`mass_${name}_`)) rest = rest.slice(`mass_${name}_`.length);
  }
  return rest;
}

// The value of a parameter in the properties of a params block: a number,
// the first of a list (node-wise ones move together here), or the value
// of a free one; none if not given (or given as an expression)
function valueOf(property: unknown): number | null {
  if (typeof property === 'number') return property;
  if (Array.isArray(property)) return valueOf(property[0]);
  if (property && typeof property === 'object') return valueOf((property as Options).value);
  return null;
}

// The property of a parameter at a value: a number, or a list of it for
// a node-wise one; of a free one, its value (its range stays)
function withValue(property: unknown, param: Param, value: number): unknown {
  if (property && typeof property === 'object' && !Array.isArray(property)) return { ...(property as Options), value };
  return param.size > 1 ? Array(param.size).fill(value) : value;
}

// --- The screen -----------------------------------------------------------

export function GalaxyScreen({ name }: { name: string }) {
  const { project } = useStudio();
  const edited = useEditedBlock(project ?? '.', 'galaxies', name);
  const head = useBlockHead(project ?? '', 'galaxies', name);
  const values = useParamsBlock(project ?? '.', name);
  const options = edited.options ?? {};
  const [settled] = useDebouncedValue(edited.options, 250);
  const description = useGalaxyParameters(project ?? '.', settled);
  const params = (description.data?.parameters ?? []) as unknown as Param[];
  const [selected, setSelected] = useState<Selected | null>(null);
  const [view, setView] = useState<View>('sky');
  const [quantity, setQuantity] = useState<Quantity | 'cube'>('light');
  const [cubeSelection, setCubeSelection] = useState<Region | null>(null);
  const [cubeBand, setCubeBand] = useState<[number, number] | null>(null);
  const [isolated, setIsolated] = useState<string | null>(null);
  const [settledValues] = useDebouncedValue(values.properties, 120);
  const components = componentsOf(options);
  const render = useGalaxyRender(project ?? '.', components.length ? settled : undefined, settledValues, view);
  // (the cube, while it is shown)
  const cube = useGalaxyCube(project ?? '.', components.length && quantity === 'cube' ? settled : undefined, settledValues, view);
  // The last render that worked stays while the galaxy has a problem
  const last = useRef<GalaxyRender | undefined>(undefined);
  if (render.data) last.current = render.data;
  const shown = render.data ?? last.current;
  useUnsaved(edited.dirty || values.dirty);
  if (project === null) return null;
  const defs = edited.defs;
  const field = shown ? (shown.size * shown.step) / 2 : 20;
  const save = () => {
    if (edited.dirty) edited.save();
    if (values.dirty) values.save();
  };
  const revert = () => {
    edited.revert();
    values.revert();
  };
  const setOptions = (next: Options) => edited.set(withOpacitiesInStep(next));
  const component = selected?.component ?? null;
  const kind = component === null ? null : String(components[component]?.type ?? '');
  const part = selected === null ? null : (component === null ? GALAXY_PARTS : (PARTS[kind ?? ''] ?? [])).find((p) => p.key === selected.part) ?? null;
  const problems = (description.data?.problems ?? []) as { path: Path; message: string }[];
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Galaxy</div>
          {head.title}
        </div>
        <div className="head-actions">
          {(edited.dirty || values.dirty) && <span className="why">Not saved</span>}
          {(edited.saveError ?? values.error) && <span className="error">{(edited.saveError ?? values.error)?.message}</span>}
          <Button disabled={!edited.dirty && !values.dirty} onClick={revert}>
            Revert
          </Button>
          <Button kind="primary" disabled={(!edited.dirty && !values.dirty) || edited.saving} onClick={save}>
            Save
          </Button>
          {head.menu}
        </div>
      </div>
      {edited.error && <p className="error screen-error">{edited.error.message}</p>}
      <div className="galaxy-screen">
        <GalaxyList options={options} defs={defs} selected={selected} onSelect={setSelected} onChange={setOptions} />
        <div className="galaxy-view" data-cube={quantity === 'cube' || undefined}>
          {components.length === 0 ? (
            <div className="galaxy-start">
              <p className="note">An empty galaxy. Add its first component to see it.</p>
              {defs && <AddComponent defs={defs} components={components} onAdd={(c) => setOptions({ ...options, components: [...components, c] })} />}
            </div>
          ) : (
            <>
              <div className="galaxy-view-bar">
                <Segments
                  items={[
                    { value: 'face-on', label: 'Face-on' },
                    { value: 'edge-on', label: 'Edge-on' },
                    { value: 'sky', label: 'As seen' },
                  ]}
                  value={view}
                  onChange={setView}
                  label="How the galaxy is seen"
                />
                <Segments
                  items={[
                    { value: 'light', label: 'Light' },
                    { value: 'velocity', label: 'Velocity' },
                    { value: 'dispersion', label: 'Dispersion' },
                    { value: 'cube', label: 'Cube' },
                  ]}
                  value={quantity}
                  onChange={setQuantity}
                  label="What the map shows"
                />
              </div>
              {quantity === 'cube' ? (
                <div className="galaxy-cube">
                  {cube.data ? (
                    <CubeViewer
                      layers={[{ key: 'data', title: 'Cube', path: cube.data.path, hdu: 0, unit: null }]}
                      shape={cube.data.shape}
                      selection={cubeSelection}
                      band={cubeBand !== null && cubeBand[1] < cube.data.shape[2] ? cubeBand : null}
                      onSelection={setCubeSelection}
                      onBand={setCubeBand}
                      noiseless
                    />
                  ) : cube.error ? (
                    <p className="error">{cube.error.message}</p>
                  ) : (
                    <div className="galaxy-map-frame galaxy-map-wait" />
                  )}
                  {cube.data && cube.error && <p className="error">{cube.error.message}</p>}
                </div>
              ) : shown ? (
                <GalaxyMap
                  render={shown}
                  quantity={quantity}
                  component={isolated}
                  arrows={view === 'face-on' && quantity === 'velocity'}
                  unit={quantity === 'light' ? '' : 'km/s'}
                />
              ) : (
                <div className="galaxy-map-frame galaxy-map-wait" />
              )}
              {render.error && <p className="error">{render.error.message}</p>}
              {components.length > 1 && quantity !== 'cube' && (
                <Segments
                  items={[{ value: '', label: 'All' }, ...components.map((c, i) => ({ value: String(c.name ?? i), label: String(c.name ?? `component ${i + 1}`) }))]}
                  value={isolated ?? ''}
                  onChange={(next) => setIsolated(next === '' ? null : next)}
                  label="The components shown"
                />
              )}
              {shown && <GalaxyCurves render={shown} mass={component === null && (part?.key === 'mass' || part?.key === 'gravity')} />}
            </>
          )}
        </div>
        <div className="galaxy-side">
          {part && selected ? (
            <PartPanel
              options={options}
              defs={defs}
              component={component}
              part={part}
              params={paramsOf(params, component, part)}
              field={field}
              properties={values.properties}
              onOptions={setOptions}
              onValue={(param, value) => values.set(param.name, withValue(values.properties[param.name], param, value))}
              onClose={() => setSelected(null)}
            />
          ) : (
            <p className="note galaxy-hint">{components.length ? 'Choose a part in the list to set it.' : ''}</p>
          )}
          {problems.length > 0 && (
            <div className="galaxy-problems">
              {problems.map((p, i) => (
                <p key={i} className="error">
                  {p.message}
                </p>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

// The params block of a galaxy (of its name): its properties as edited,
// and how they are saved (the block is made the first time)
function useParamsBlock(project: string, name: string) {
  const path = joinPath(project, `params/${name}.yaml`);
  const piece = usePiece(path);
  const update = useUpdateBlock();
  const create = useCreateBlock(project);
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);
  const saved = ((piece.data as Options | undefined)?.properties ?? {}) as Record<string, unknown>;
  const properties = draft ?? saved;
  const exists = piece.data !== undefined;
  return {
    properties,
    dirty: draft !== null && JSON.stringify(draft) !== JSON.stringify(saved),
    error: (update.error ?? create.error) as Error | null,
    set: (param: string, property: unknown) => setDraft({ ...properties, [param]: property }),
    revert: () => setDraft(null),
    save: () => {
      if (draft === null) return;
      const content = { ...((piece.data as Options | undefined) ?? {}), properties: draft };
      if (exists) update.mutate({ path, content }, { onSuccess: () => setDraft(null) });
      else create.mutate({ folder: 'params', name, content }, { onSuccess: () => setDraft(null) });
    },
  };
}

// The menu that adds a component of a type, named after it
function AddComponent({ defs, components, onAdd, small }: { defs: Defs; components: Options[]; onAdd: (component: Options) => void; small?: boolean }) {
  return (
    <DropMenu
      label="Add a component"
      width={MENU_WIDTH}
      entries={typesOf(defs, 'component').map(({ value: type, description }) => ({
        label: STARTS[type] ? type : `${type} (needs a file: in the configuration)`,
        description,
        onClick: () => STARTS[type] && onAdd({ type, name: freshName(components, type), ...STARTS[type] }),
      }))}
      target={
        small ? (
          <button className="galaxy-add-icon" aria-label="Add a component" title="Add a component">
            +
          </button>
        ) : (
          <Button kind="small">+ Component</Button>
        )
      }
    />
  );
}

// The list of what the galaxy is made of: the galaxy as a whole (its parts
// that it has, and those to add), and each component with its parts
function GalaxyList({
  options,
  defs,
  selected,
  onSelect,
  onChange,
}: {
  options: Options;
  defs: Defs | undefined;
  selected: Selected | null;
  onSelect: (selected: Selected) => void;
  onChange: (options: Options) => void;
}) {
  const components = componentsOf(options);
  const isOn = (component: number | null, part: string) => selected?.component === component && selected.part === part;
  const missing = GALAXY_PARTS.filter((p) => getAt(options, p.option) === undefined);
  return (
    <div className="galaxy-list">
      <div className="galaxy-node galaxy-group">Galaxy</div>
      {GALAXY_PARTS.filter((p) => getAt(options, p.option) !== undefined).map((p) => (
        <button key={p.key} className="galaxy-node galaxy-part" data-on={isOn(null, p.key) || undefined} onClick={() => onSelect({ component: null, part: p.key })}>
          <span>{p.title}</span>
          <span className="galaxy-what">{glance(getAt(options, p.option))}</span>
        </button>
      ))}
      {missing.length > 0 && (
        <DropMenu
          label="Add to the galaxy"
          entries={missing.map((p) => ({
            label: p.title,
            // (gravity acts on the mass: none without it)
            ...(p.key === 'gravity' && asList(options.mass).length === 0 ? { disabled: true, description: 'Add mass first: gravity is that of the mass.' } : {}),
            // (mass needs the distance of the galaxy: a direct one comes with it)
            ...(p.key === 'mass' && options.distance === undefined ? { description: 'With a distance, which mass needs.' } : {}),
            onClick: () => {
              const withDistance = p.key === 'mass' && options.distance === undefined ? setAt(options, ['distance'], { type: 'direct' }) : options;
              onChange(setAt(withDistance, p.option, p.start ?? []));
              onSelect({ component: null, part: p.key });
            },
          }))}
          target={<button className="galaxy-add">+ {missing.map((p) => p.title.toLowerCase()).join(', ')}</button>}
        />
      )}
      {components.length > 0 && defs && (
        <div className="galaxy-node galaxy-group galaxy-components-head">
          <span>Components</span>
          <AddComponent defs={defs} components={components} onAdd={(c) => onChange({ ...options, components: [...components, c] })} small />
        </div>
      )}
      {components.map((c, i) => {
        const kinds = PARTS[String(c.type)] ?? [];
        const present = kinds.filter((p) => p.key === 'position' || getAt(c, p.option) !== undefined);
        const absent = kinds.filter((p) => p.optional && getAt(c, p.option) === undefined);
        return (
          <div key={i} className="galaxy-component">
            <div className="galaxy-node galaxy-group">
              <span>{String(c.name ?? `component ${i + 1}`)}</span>
              <span className="galaxy-what">{String(c.type)}</span>
              <Close
                label={`Remove ${String(c.name ?? 'the component')}`}
                onClick={() => onChange({ ...options, components: components.filter((_, j) => j !== i) })}
              />
            </div>
            {present.map((p) => (
              <button key={p.key} className="galaxy-node galaxy-part" data-on={isOn(i, p.key) || undefined} onClick={() => onSelect({ component: i, part: p.key })}>
                <span>{p.title}</span>
                <span className="galaxy-what">{p.key === 'position' ? '' : glance(getAt(c, p.option))}</span>
              </button>
            ))}
            {absent.length > 0 && (
              <DropMenu
                label={`Add to ${String(c.name ?? 'the component')}`}
                entries={absent.map((p) => ({
                  label: p.title,
                  onClick: () => {
                    onChange(setAt(options, ['components', i, ...p.option], p.start));
                    onSelect({ component: i, part: p.key });
                  },
                }))}
                target={<button className="galaxy-add">+ {absent.map((p) => p.title.toLowerCase()).join(', ')}</button>}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// The types of a family of the schema (e.g. 'vpt': the velocity polar
// traits), with what each is
function typesOf(defs: Defs | undefined, family: string): { value: string; label: string; description?: string }[] {
  if (!defs?.[family]) return [];
  // (each type once: a family can have two members of one type, e.g. the
  // mass tables of disks and of spheroids)
  const types = new Map<string, string | undefined>();
  for (const m of membersOf(defs[family], defs)) {
    const type = typeOf(m)!;
    if (!types.has(type)) types.set(type, typeof m.description === 'string' ? firstSentence(m.description) : undefined);
  }
  return [...types].map(([type, description]) => ({ value: type, label: type, description }));
}

// The first sentence of a description (not cut at e.g. or i.e.)
function firstSentence(text: string): string {
  const match = /^.*?\.(?=\s+[A-Z(]|$)/s.exec(text);
  return (match ? match[0] : text).trim();
}

// The width of the menus that add an item of a type, with what each is
const MENU_WIDTH = 320;

// What the numbers that an item requires start at, by name, if the schema
// gives them no default (e.g. the zmax of a density height trait); others
// start at their least
const NUMBER_STARTS: Record<string, number> = { zmax: 6, order: 2, nblobs: 1, ngauss: 1 };

// What some new items start with, by family and type, before the options
// they require (see filled): a halo of an NFW profile, a mass disk of an
// exponential surface, thin
const ITEM_STARTS: Record<string, Record<string, Options>> = {
  // (self-absorption the same everywhere; see withOpacitiesInStep)
  opacity: { line: { coefficients: { type: 'uniform' } } },
  mass_component: {
    spheroid: { profile: { type: 'nfw' } },
    disk: { surface: { type: 'exponential' }, vertical: { type: 'thin' } },
  },
};

// The options of an object of the schema with those it requires: at their
// first choice (e.g. the direction of a velocity trait), of the first type
// of a family (with what it requires), or a name not taken (names)
function filled(defs: Defs, node: Schema, given: Options, names: string[]): Options {
  const resolved = resolve(node, defs);
  const properties = (resolved.properties ?? {}) as Record<string, Schema>;
  const result: Options = { ...given };
  for (const key of (resolved.required as string[] | undefined) ?? []) {
    if (key in result) continue;
    if (key === 'name') {
      result.name = freshName(names.map((name) => ({ name })), String(result.type ?? 'item'));
      continue;
    }
    const property = resolve(properties[key] ?? {}, defs);
    const branches = ((property.anyOf as Schema[] | undefined) ?? [property]).map((b) => resolve(b, defs)).filter((b) => b.type !== 'null');
    const first = branches[0];
    if (!first) continue;
    if (Array.isArray(first.enum)) result[key] = first.enum[0];
    else if (first.type === 'number' || first.type === 'integer') {
      const start = NUMBER_STARTS[key] ?? property.default ?? first.default ?? first.minimum;
      if (typeof start === 'number') result[key] = start;
    }
    else if (membersOf(first, defs).length) {
      const member = membersOf(first, defs)[0];
      result[key] = filled(defs, member, { type: typeOf(member) }, []);
    } else if (first.type === 'object') result[key] = filled(defs, first, {}, []);
  }
  return result;
}

// A new item of a type of a family, with what it requires (see filled),
// named apart from those of its list (names)
function startOf(defs: Defs | undefined, family: string, type: string, names: string[] = []): Options {
  const member = defs?.[family] ? membersOf(defs[family], defs).find((m) => typeOf(m) === type) : undefined;
  const start = { type, ...(ITEM_STARTS[family]?.[type] ?? {}) };
  return defs && member ? filled(defs, member, start, names) : start;
}

// An item of a family given another type: it keeps the options of the old
// that the new type has too (e.g. the zmax and name of a height trait), and
// starts the others (see startOf)
function retyped(defs: Defs | undefined, family: string, item: Options | undefined, type: string): Options {
  const member = defs?.[family] ? membersOf(defs[family], defs).find((m) => typeOf(m) === type) : undefined;
  const has = (member && defs ? (resolve(member, defs).properties ?? {}) : {}) as Record<string, unknown>;
  const kept = Object.entries(item ?? {}).filter(([key]) => key !== 'type' && key in has);
  return { ...startOf(defs, family, type), ...Object.fromEntries(kept) };
}

// The labels of some types where their names say too little
const TYPE_LABELS: Record<string, Record<string, string>> = {
  opacity: { dust: 'dust', line: 'self-absorption (of its lines)' },
};

// The geometry of a component beyond its sliders: the radii of the rings
// of a disk or cone (evenly spaced to a radius, every sep or len of them,
// or given), whether the rings of a disk tilt (posa and incl per ring) or
// wander (centre and vsys per ring); how far the light of a spheroid is
// drawn
function GeometryFields({ component, onChange }: { component: Options; onChange: (component: Options) => void }) {
  const type = component.type;
  if (type === 'spheroid') {
    return (
      <div className="galaxy-fields">
        <label className="galaxy-field">
          <span>Light drawn out to</span>
          <NumberOption value={Number(component.rmax)} onCommit={(rmax) => onChange({ ...component, rmax })} label="rmax" unit="″" />
        </label>
      </div>
    );
  }
  if (type !== 'disk' && type !== 'cone') return null;
  return (
    <div className="galaxy-fields">
      <RingsField value={component.rnodes} onChange={(rnodes) => onChange({ ...component, rnodes })} title={type === 'cone' ? 'Shells' : 'Rings'} />
      {type === 'disk' && (
        <>
          <Check
            checked={component.tilted === true}
            onChange={(tilted) => onChange(withFlag(component, 'tilted', tilted))}
            label="Tilted rings: position angle and inclination per ring (a warp)"
          />
          <Check
            checked={component.loose === true}
            onChange={(loose) => onChange(withFlag(component, 'loose', loose))}
            label="Loose rings: centre and systemic velocity per ring"
          />
        </>
      )}
    </div>
  );
}

// Options with a flag set (true) or left to its default (false: removed)
function withFlag(options: Options, key: string, on: boolean): Options {
  const copy = { ...options };
  if (on) copy[key] = true;
  else delete copy[key];
  return copy;
}

type RingsForm = 'sep' | 'len' | 'list';

// The radii of the rings: {max, sep}, {max, len} or a list of them; a
// change of form keeps the reach of the rings and about their number
function RingsField({ value, onChange, title }: { value: unknown; onChange: (value: unknown) => void; title: string }) {
  const form: RingsForm = Array.isArray(value) ? 'list' : value && typeof value === 'object' && 'len' in value ? 'len' : 'sep';
  const radii = ringRadii(value);
  const max = radii.length ? radii[radii.length - 1] : 16;
  const options = (value && typeof value === 'object' && !Array.isArray(value) ? value : {}) as Options;
  const toForm = (next: RingsForm) => {
    const count = Math.max(radii.length, 2);
    if (next === 'sep') onChange({ max, sep: significantStep(max / (count - 1)) });
    else if (next === 'len') onChange({ max, len: count });
    else onChange(radii);
  };
  return (
    <div className="galaxy-rings">
      <div className="galaxy-rings-head">
        <span>{title}</span>
        <Segments
          items={[
            { value: 'sep', label: 'Spacing' },
            { value: 'len', label: 'Count' },
            { value: 'list', label: 'Radii' },
          ]}
          value={form}
          onChange={toForm}
          label={`How the ${title.toLowerCase()} are given`}
        />
      </div>
      {form === 'list' ? (
        <RadiiOption radii={radii} onCommit={onChange} />
      ) : (
        <div className="galaxy-rings-row">
          <span>out to</span>
          <NumberOption value={Number(options.max)} onCommit={(m) => onChange({ ...options, max: m })} label="max" unit="″" />
          {form === 'sep' ? (
            <>
              <span>every</span>
              <NumberOption value={Number(options.sep)} onCommit={(sep) => onChange({ ...options, sep })} label="sep" unit="″" />
            </>
          ) : (
            <>
              <span>in</span>
              <NumberOption value={Number(options.len)} onCommit={(len) => onChange({ ...options, len: Math.round(len) })} label="len" unit="radii" />
            </>
          )}
        </div>
      )}
      <div className="note">
        {radii.length} radii, 0 to {significant(max, 4)}″
      </div>
    </div>
  );
}

// The radii of rings of any form (see Nodes): every sep from 0 to the first
// at or beyond max, len of them from 0 to max, or those given
function ringRadii(value: unknown): number[] {
  if (Array.isArray(value)) return value.map(Number).filter(Number.isFinite);
  const { max, sep, len } = (value ?? {}) as { max?: number; sep?: number; len?: number };
  if (!(Number(max) > 0)) return [];
  if (Number(sep) > 0 && Number(sep) <= Number(max)) {
    const n = Math.ceil(Number(max) / Number(sep) - 1e-9);
    return Array.from({ length: n + 1 }, (_, i) => i * Number(sep));
  }
  if (Number(len) >= 2) return Array.from({ length: Number(len) }, (_, i) => (i * Number(max)) / (Number(len) - 1));
  return [];
}

// A round step near a value (1, 2 or 5 times a power of 10)
function significantStep(value: number): number {
  const power = 10 ** Math.floor(Math.log10(value));
  const lead = value / power;
  return (lead < 1.5 ? 1 : lead < 3.5 ? 2 : lead < 7.5 ? 5 : 10) * power;
}

// A positive number, as typed: it is set once it reads as one
function NumberOption({ value, onCommit, label, unit }: { value: number; onCommit: (value: number) => void; label: string; unit: string }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <span className="galaxy-number">
      <NumberField
        value={text ?? (Number.isFinite(value) ? String(value) : '')}
        onChange={(next) => {
          setText(next);
          const number = Number(next);
          if (next.trim() !== '' && Number.isFinite(number) && number > 0) onCommit(number);
        }}
        label={label}
      />
      <span className="note">{unit}</span>
    </span>
  );
}

// Radii given as a list, as typed (by commas or spaces): set once each
// reads as a number
function RadiiOption({ radii, onCommit }: { radii: number[]; onCommit: (radii: number[]) => void }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <TextField
      value={text ?? radii.join(', ')}
      onChange={(next) => {
        setText(next);
        const parts = next.split(/[\s,]+/).filter(Boolean);
        const numbers = parts.map(Number);
        if (numbers.length >= 2 && numbers.every(Number.isFinite)) onCommit(numbers);
      }}
      onBlur={() => setText(null)}
      label="The radii (arcsec)"
      mono
    />
  );
}

// The lines of an emission of lines: by default one line at the velocity
// of the spectral axis, whatever it is; else lines of the catalogue (or of
// the configuration, kept as they are), the first the one whose flux the
// density gives, the others of a flux ratio to it
function LinesField({ emission, onChange }: { emission: Options; onChange: (emission: Options) => void }) {
  const catalogue = (useLines().data ?? []) as { name: string; rest: string }[];
  const lines = lineList(emission.lines);
  const rests = new Map(catalogue.map((l) => [l.name, l.rest]));
  const set = (next: unknown[]) => {
    const copy = { ...emission };
    if (next.length === 0) delete copy.lines;
    else copy.lines = next.length === 1 ? next[0] : next;
    onChange(copy);
  };
  const named = lines.map(lineName);
  return (
    <div className="galaxy-lines">
      {lines.length === 0 ? (
        <p className="note">One line, at the velocity of the spectral axis, whatever line it is.</p>
      ) : (
        lines.map((line, i) => (
          <div key={i} className="galaxy-item-head">
            {typeof line === 'string' ? (
              <Choice
                items={catalogue
                  .filter((l) => l.name === line || !named.includes(l.name))
                  .map((l) => ({ value: l.name, label: l.name, description: l.rest }))}
                value={line}
                onChange={(name) => set(lines.map((l, j) => (j === i ? name : l)))}
                label={`Line ${i + 1}`}
              />
            ) : (
              <span className="galaxy-line-own">{lineName(line)} (of the configuration)</span>
            )}
            <span className="note galaxy-line-role">{i === 0 ? 'flux' : 'ratio'}</span>
            <Close label={`Remove the line ${lineName(line)}`} onClick={() => set(lines.filter((_, j) => j !== i))} />
          </div>
        ))
      )}
      <DropMenu
        label="Add a line"
        width={MENU_WIDTH}
        entries={catalogue
          .filter((l) => !named.includes(l.name))
          .map((l) => ({ label: l.name, description: rests.get(l.name), onClick: () => set([...lines, l.name]) }))}
        target={<button className="galaxy-add">+ line</button>}
      />
    </div>
  );
}

// The lines of an emission as a list: names, or lines of the configuration
function lineList(value: unknown): unknown[] {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}

// The name of a line: its name in the catalogue, or that of its options
function lineName(line: unknown): string {
  return typeof line === 'string' ? line : String((line as Options | null)?.name ?? '?');
}

// The galaxy with the coefficients of the opacity of the lines of each
// component (self-absorption) in step with its lines: of one unnamed line,
// its traits; of named lines, the traits of each by its name (kept where
// it had them; else uniform)
function withOpacitiesInStep(options: Options): Options {
  const components = componentsOf(options);
  let changed = false;
  const next = components.map((c) => {
    const opacity = c.opacity as Options | undefined;
    if (opacity?.type !== 'line') return c;
    const names = lineList((c.emission as Options | undefined)?.lines).map(lineName);
    const given = opacity.coefficients;
    const isMapping = given !== null && typeof given === 'object' && !Array.isArray(given) && !('type' in given);
    const single = isMapping ? Object.values(given as Options)[0] : given;
    const fallback = single ?? { type: 'uniform' };
    const coefficients =
      names.length === 0
        ? fallback
        : Object.fromEntries(names.map((name) => [name, (isMapping ? (given as Options)[name] : undefined) ?? fallback]));
    if (JSON.stringify(coefficients) === JSON.stringify(given)) return c;
    changed = true;
    return { ...c, opacity: { ...opacity, coefficients } };
  });
  return changed ? { ...options, components: Array.isArray(options.components) ? next : next[0] } : options;
}

// The panel of a part: its type (or the types of its items, each with its
// own options of a choice, e.g. a direction), its parameters as sliders,
// and the items to add; or to remove it, if it can be
function PartPanel({
  options,
  defs,
  component,
  part,
  params,
  field,
  properties,
  onOptions,
  onValue,
  onClose,
}: {
  options: Options;
  defs: Defs | undefined;
  component: number | null;
  part: PartKind;
  params: Param[];
  field: number;
  properties: Record<string, unknown>;
  onOptions: (options: Options) => void;
  onValue: (param: Param, value: number) => void;
  onClose: () => void;
}) {
  const base: Path = component === null ? [] : ['components', component];
  const option = [...base, ...part.option];
  const value = getAt(options, option);
  const name = component === null ? 'the galaxy' : String(getAt(options, [...base, 'name']) ?? 'the component');
  const sliders = (owned: Param[]) =>
    owned.map((p) => (
      <ParamSlider key={p.name} param={p} label={shortName(p, options)} field={field} value={valueOf(properties[p.name]) ?? p.default} onChange={(v) => onValue(p, v)} />
    ));
  const remove = () => {
    onOptions(setAt(options, option, undefined));
    onClose();
  };
  const spheroids = componentsOf(options)
    .filter((c) => c.type === 'spheroid')
    .map((c) => String(c.name ?? ''));
  return (
    <div className="card galaxy-part-panel">
      <div className="card-head galaxy-panel-head">
        <div>
          <h2>{part.title}</h2>
          <div className="note">of {name}</div>
        </div>
        {part.optional && <Close label={`Remove the ${part.title.toLowerCase()}`} onClick={remove} />}
      </div>
      <div className="card-body">
        {part.key === 'position' && component !== null && (
          <GeometryFields component={getAt(options, base) as Options} onChange={(next) => onOptions(setAt(options, base, next))} />
        )}
        {part.typed && (
          <Choice
            items={typesOf(defs, part.typed).map((t) => ({ ...t, label: TYPE_LABELS[part.typed!]?.[t.value] ?? t.label }))}
            value={String((value as Options | undefined)?.type ?? '')}
            onChange={(type) => onOptions(setAt(options, option, retyped(defs, part.typed!, value as Options | undefined, type)))}
            label={`The ${part.title.toLowerCase()}`}
          />
        )}
        {part.key === 'light' && (value as Options | undefined)?.type === 'lines' && (
          <LinesField emission={value as Options} onChange={(next) => onOptions(setAt(options, option, next))} />
        )}
        {part.typed && sliders(params)}
        {part.family &&
          asList(value).map((item, i) => {
            const owned = params.filter((p) => startsWith(p.owner, [...option, i]));
            const items = asList(value);
            const replace = (next: Options | undefined) => {
              const changed = items.map((it, j) => (j === i ? next : it)).filter((it): it is Options => it !== undefined);
              onOptions(setAt(options, option, withItems(changed)));
            };
            const directions = (defs?.[part.family!] ? membersOf(defs[part.family!], defs).find((m) => typeOf(m) === item.type)?.properties : undefined) as
              | Record<string, { enum?: string[]; type?: string }>
              | undefined;
            return (
              <div key={i} className="galaxy-item">
                <div className="galaxy-item-head">
                  <Choice
                    items={typesOf(defs, part.family!)}
                    value={String(item.type ?? '')}
                    onChange={(type) => replace(retyped(defs, part.family!, item, type))}
                    label={`The type of the ${part.title.toLowerCase()} ${i + 1}`}
                  />
                  {directions?.component && (
                    <Choice
                      items={spheroids.map((s) => ({ value: s, label: s }))}
                      value={String(item.component ?? '')}
                      onChange={(followed) => replace({ ...item, component: followed })}
                      label="The spheroid whose light it follows"
                    />
                  )}
                  {directions?.direction?.enum && (
                    <Choice
                      items={directions.direction.enum.map((d) => ({ value: d, label: d }))}
                      value={String(item.direction ?? '')}
                      onChange={(direction) => replace({ ...item, direction })}
                      label="Its direction"
                    />
                  )}
                  {(items.length > 1 || part.optional) && <Close label="Remove it" onClick={() => replace(undefined)} />}
                </div>
                {sliders(owned)}
              </div>
            );
          })}
        {part.family && (
          <DropMenu
            label={`Add to the ${part.title.toLowerCase()}`}
            width={MENU_WIDTH}
            entries={typesOf(defs, part.family).map((t) => {
              // (mass that follows the light of a spheroid needs one)
              const follows = part.family === 'mass_component' && t.value === 'light_spheroid';
              return {
                label: t.label,
                description: follows && spheroids.length === 0 ? 'Add a spheroid component first: this mass follows its light.' : t.description,
                disabled: follows && spheroids.length === 0,
                onClick: () => {
                  const items = asList(value);
                  const names = items.map((item) => String(item.name ?? ''));
                  const start = startOf(defs, part.family!, t.value, names);
                  onOptions(setAt(options, option, withItems([...items, follows ? { ...start, component: spheroids[0] } : start])));
                },
              };
            })}
            target={<button className="galaxy-add">+ {part.title.toLowerCase()}</button>}
          />
        )}
        {part.key === 'position' && sliders(params)}
        {params.length === 0 && !part.family && !part.typed && <p className="note">Nothing to set.</p>}
      </div>
    </div>
  );
}

// A parameter as a slider: its name (that in its part), its value and
// unit; node-wise ones are set to one value for every ring here
function ParamSlider({
  param,
  label,
  field,
  value,
  onChange,
}: {
  param: Param;
  label: string;
  field: number;
  value: number | null;
  onChange: (value: number) => void;
}) {
  const { min, max, step } = useMemo(() => sliderOf(param, field), [param, field]);
  const shown = value ?? param.default ?? min;
  const unit = param.unit === 'km / s' ? 'km/s' : param.unit === 'deg' ? '°' : param.unit === 'arcsec' ? '″' : (param.unit ?? '');
  return (
    <div className="galaxy-param">
      <div className="galaxy-param-row">
        <span title={param.name}>
          {label}
          {param.size > 1 && <span className="note"> · every ring</span>}
        </span>
        <span className="galaxy-param-value">
          {significant(shown, 4)} {unit}
        </span>
      </div>
      <StepSlider min={min} max={max} step={step} value={shown} onChange={onChange} format={(v) => significant(v, 4)} label={param.name} />
    </div>
  );
}
