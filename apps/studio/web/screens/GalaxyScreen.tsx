// The screen of a galaxy, built as it is seen: a short list of what it is
// made of (the galaxy as a whole: its distance, mass and gravity; and its
// components, each of its parts: density, motions, light, ...), the view
// of the galaxy (its maps, face-on, edge-on or as seen, and its curves),
// and the part chosen in the list with its few parameters as sliders.
// Only what has been added shows; the rest is added from the list. The
// values of the sliders are the starting values of the parameters, kept
// in the params block of the same name as the galaxy.
import { useDebouncedValue } from '@mantine/hooks';
import { useMemo, useRef, useState } from 'react';
import { useCreateBlock, useGalaxyParameters, useGalaxyRender, usePiece, useUpdateBlock, type GalaxyRender } from '../api/queries';
import { joinPath } from '../components/FilePicker';
import { GalaxyCurves, GalaxyMap, type Quantity } from '../components/GalaxyView';
import { significant } from '../components/format';
import { membersOf, resolve, typeOf, type Defs, type Options, type Schema } from '../components/SchemaForm';
import { Button, Close } from '../components/ui/basics';
import { Choice, Segments } from '../components/ui/inputs';
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
  { key: 'position', title: 'Position', option: [] },
  { key: 'density', title: 'Density', option: ['density', 'ptraits'], family: 'rpt' },
  { key: 'thickness', title: 'Thickness', option: ['density', 'htraits'], family: 'rht', optional: true, start: { type: 'sech2', zmax: 6 } },
  { key: 'motions', title: 'Motions', option: ['kinematics', 'vptraits'], family: 'vpt' },
  { key: 'dispersion', title: 'Dispersion', option: ['kinematics', 'dptraits'], family: 'dpt' },
  { key: 'light', title: 'Light', option: ['emission'], typed: 'emission' },
];

const PARTS: Record<string, PartKind[]> = {
  disk: DISK_PARTS,
  cone: DISK_PARTS.map((part) => (part.key === 'thickness' ? { ...part, title: 'Walls', optional: false } : part)),
  spheroid: [
    { key: 'position', title: 'Position', option: [] },
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
    // (of a trait, its prefix: its kind and position, or its name)
    if (typeof param.owner[param.owner.length - 1] === 'number' && param.owner.length > 3) rest = rest.slice(rest.indexOf('_') + 1);
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
  const [quantity, setQuantity] = useState<Quantity>('light');
  const [isolated, setIsolated] = useState<string | null>(null);
  const [settledValues] = useDebouncedValue(values.properties, 120);
  const components = componentsOf(options);
  const render = useGalaxyRender(project ?? '.', components.length ? settled : undefined, settledValues, view);
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
  const setOptions = (next: Options) => edited.set(next);
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
        <div className="galaxy-view">
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
                  ]}
                  value={quantity}
                  onChange={setQuantity}
                  label="What the map shows"
                />
              </div>
              {shown ? (
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
              {components.length > 1 && (
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
function AddComponent({ defs, components, onAdd }: { defs: Defs; components: Options[]; onAdd: (component: Options) => void }) {
  const types = [...new Set(membersOf(defs.component ?? {}, defs).map((m) => typeOf(m)!))];
  return (
    <DropMenu
      label="Add a component"
      entries={types.map((type) => ({
        label: STARTS[type] ? type : `${type} (needs a file: in the configuration)`,
        onClick: () => STARTS[type] && onAdd({ type, name: freshName(components, type), ...STARTS[type] }),
      }))}
      target={<Button kind="small">+ Component</Button>}
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
            onClick: () => {
              onChange(setAt(options, p.option, p.start ?? []));
              onSelect({ component: null, part: p.key });
            },
          }))}
          target={<button className="galaxy-add">+ {missing.map((p) => p.title.toLowerCase()).join(', ')}</button>}
        />
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
      {components.length > 0 && defs && (
        <div className="galaxy-add-component">
          <AddComponent defs={defs} components={components} onAdd={(c) => onChange({ ...options, components: [...components, c] })} />
        </div>
      )}
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
    if (!types.has(type)) types.set(type, typeof m.description === 'string' ? m.description.split('.')[0] : undefined);
  }
  return [...types].map(([type, description]) => ({ value: type, label: type, description }));
}

// What some new items start with, by family and type, before the options
// they require (see filled): a halo of an NFW profile, a mass disk of an
// exponential surface, thin
const ITEM_STARTS: Record<string, Record<string, Options>> = {
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
        {part.typed && (
          <Choice
            items={typesOf(defs, part.typed)}
            value={String((value as Options | undefined)?.type ?? '')}
            onChange={(type) => onOptions(setAt(options, option, startOf(defs, part.typed!, type)))}
            label={`The ${part.title.toLowerCase()}`}
          />
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
              | Record<string, { enum?: string[] }>
              | undefined;
            return (
              <div key={i} className="galaxy-item">
                <div className="galaxy-item-head">
                  <Choice
                    items={typesOf(defs, part.family!)}
                    value={String(item.type ?? '')}
                    onChange={(type) => replace({ ...startOf(defs, part.family!, type), ...(item.name ? { name: item.name } : {}) })}
                    label={`The type of the ${part.title.toLowerCase()} ${i + 1}`}
                  />
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
            entries={typesOf(defs, part.family).map((t) => ({
              label: t.label,
              onClick: () => {
                const items = asList(value);
                const names = items.map((item) => String(item.name ?? ''));
                onOptions(setAt(options, option, withItems([...items, startOf(defs, part.family!, t.value, names)])));
              },
            }))}
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
