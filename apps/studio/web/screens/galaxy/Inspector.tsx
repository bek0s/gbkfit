// The inspector of a part of a galaxy chosen in the outline: what it is
// (its type, or its items, each of a type), its parameters (see
// ParamField), and its other options (a form of the schema). A list of
// traits has a card for each, with the height trait paired with it on a
// thick disk or a cone; items are added and removed here, and a part that
// can be.
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useComponentTraits, useLines } from '../../api/queries';
import { choicesOf, SchemaForm, type Defs, type Schema } from '../../components/SchemaForm';
import { Close } from '../../components/ui/basics';
import { Check, Choice, Segments } from '../../components/ui/inputs';
import { DropMenu } from '../../components/ui/overlays';
import { HEAD_OPTIONS, HEIGHT_START, isDisk, ITEM_STARTS, MASS_HEAD_OPTIONS, type Part } from './catalogue';
import type { EditHow, GalaxyDraft, Problem } from './draft';
import { ParamField } from './ParamField';
import { asList, asOneOrList, getAt, isObject, lineList, lineName, ringRadii, roundStep, setAt, startsWith, type Options, type Path } from './options';
import { ownedBy, settingOf, shortName, withFreedom, withRange, withValues, freeRange, type Param } from './params';
import { memberOf, restOf, retyped, startOf, typesOf } from './schema';

const MENU_WIDTH = 340;

// The map of the paths of options as an item of a list is removed: the
// item gone, those after it one place back
export function removedFrom(list: Path, index: number) {
  return (owner: Path): Path | null => {
    if (!startsWith(owner, list) || typeof owner[list.length] !== 'number') return owner;
    const at = owner[list.length] as number;
    if (at === index) return null;
    return at > index ? [...list, at - 1, ...owner.slice(list.length + 1)] : owner;
  };
}

// Every path as it is (an edit that renames what owns parameters, e.g. a
// component or a trait renamed, without moving it)
export const inPlace = (owner: Path): Path => owner;

interface Context {
  draft: GalaxyDraft;
  options: Options;
  defs: Defs | undefined;
  params: Param[];
  project: string;
  field: number;
}

export interface InspectorProps extends Context {
  component: number | null;
  part: Part;
  problems: Problem[];
  onClose: () => void;
}

export function Inspector({ draft, options, defs, params, project, field, component, part, problems, onClose }: InspectorProps) {
  const base: Path = component === null ? [] : ['components', component];
  const owner = component === null ? options : (getAt(options, base) as Options | undefined) ?? {};
  const optionPath = [...base, ...part.option];
  const value = getAt(options, optionPath);
  const ctx: Context = { draft, options, defs, params, project, field };
  const ownerName = component === null ? 'the galaxy' : String(owner.name ?? `component ${component + 1}`);
  const set = (path: Path, next: unknown, how?: EditHow) => draft.edit((s) => ({ ...s, options: setAt(s.options, path, next) }), how);
  const own = problems.filter((p) => startsWith(p.path, optionPath) || (part.key === 'geometry' && p.path.length <= base.length + 1 && startsWith(p.path, base)));
  const rings = component === null ? null : ringRadii(owner.rnodes);
  // (the actions of the head, which stays at the top as the part scrolls:
  // e.g. what adds an item to a list)
  const [actions, setActions] = useState<HTMLDivElement | null>(null);
  return (
    <div className="gx-inspector">
      <div className="gx-inspector-head">
        <div>
          <h2>{part.title}</h2>
          <div className="gx-inspector-of">
            of <b>{ownerName}</b>
            {part.hint && <span> · {part.hint}</span>}
          </div>
        </div>
        <div className="gx-inspector-actions" ref={setActions}>
          {part.optional && (
            <button className="gx-link gx-danger" onClick={() => (set(optionPath, undefined), onClose())}>
              Remove
            </button>
          )}
        </div>
      </div>
      {own.length > 0 && (
        <div className="gx-problems">
          {own.map((p, i) => (
            <p key={i}>{p.message}</p>
          ))}
        </div>
      )}
      <div className="gx-inspector-body">
        {part.key === 'geometry' && component !== null && (
          <>
            <GeometryFields component={owner} onChange={(next) => set(base, next)} />
            <Params ctx={ctx} owned={ownedBy(params, base, true)} rings={rings} />
            <MoreOptions ctx={ctx} family="component" item={owner} path={base} without={COMPONENT_HANDLED} title="Options of the component" />
          </>
        )}
        {part.typed && <TypedPart ctx={ctx} part={part} path={optionPath} value={value} rings={rings} />}
        {part.family && <ListPart ctx={ctx} part={part} base={base} rings={rings} kind={String(owner.type ?? '')} isGalaxy={component === null} head={actions} />}
      </div>
    </div>
  );
}

// The options of a component the screen sets in its parts and geometry,
// not among its other options
const COMPONENT_HANDLED = new Set([
  'type',
  'name',
  'rnodes',
  'loose',
  'tilted',
  'rmax',
  'density',
  'kinematics',
  'emission',
  'opacity',
  'zptraits',
  'sptraits',
  'profile',
  'shape',
]);

// The parameters of a part, each a field
// (limit: the most of the sliders of lengths, e.g. the zmax of a height
// trait, if less than the field)
function Params({ ctx, owned, rings, limit }: { ctx: Context; owned: Param[]; rings: number[] | null; limit?: number }) {
  const { draft, options, params, field } = ctx;
  if (owned.length === 0) return null;
  return (
    <div className="gx-params">
      {owned.map((p) => {
        const setting = settingOf(draft.state?.properties ?? {}, p);
        const setProperty = (next: unknown, gesture?: string) =>
          draft.edit((s) => ({ ...s, properties: next === undefined ? omit(s.properties, p.name) : { ...s.properties, [p.name]: next } }), { gesture });
        const property = draft.state?.properties[p.name];
        return (
          <ParamField
            key={p.name}
            param={p}
            label={shortName(p, options, params)}
            setting={setting}
            field={limit === undefined ? field : Math.min(field, limit)}
            rings={rings}
            onValues={(values, gesture) => setProperty(withValues(property, p, values), gesture)}
            onFree={(free) => setProperty(withFreedom(property, p, setting, free, freeRange(p, field)))}
            onRange={(min, max) => setProperty(withRange(property, min, max), `${p.name}/range`)}
            onUntie={() => setProperty(undefined)}
          />
        );
      })}
    </div>
  );
}

function omit(properties: Record<string, unknown>, key: string) {
  const { [key]: _, ...rest } = properties;
  return rest;
}

// The other options of an item (those the screen does not set otherwise),
// as a form of the schema, folded at first
function MoreOptions({
  ctx,
  family,
  item,
  path,
  without,
  title = 'Options',
  open: opened = false,
}: {
  ctx: Context;
  family: string;
  item: unknown;
  path: Path;
  without: Set<string>;
  title?: string;
  open?: boolean;
}) {
  const [open, setOpen] = useState(opened);
  const { defs, draft, project } = ctx;
  const options = isObject(item) ? item : {};
  const node = restOf(defs, memberOf(defs, family, options.type), without);
  if (!defs || !node) return null;
  const shown = Object.fromEntries(Object.entries(options).filter(([key]) => !without.has(key) && key !== 'type'));
  const count = Object.keys(shown).length;
  return (
    <div className="gx-more" data-open={open || undefined}>
      <button className="gx-more-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="ui-chevron" data-opened={open}>
          ▶
        </span>
        {title}
        {count > 0 && <small>{count} set</small>}
      </button>
      {open && (
        <SchemaForm
          defs={defs}
          node={node as Schema}
          project={project}
          value={shown}
          starts={ITEM_STARTS}
          onChange={(next) => {
            const kept = Object.fromEntries(Object.entries(options).filter(([key]) => without.has(key) || key === 'type'));
            draft.edit((s) => ({ ...s, options: setAt(s.options, path, { ...kept, ...next }) }), { moved: inPlace });
          }}
        />
      )}
    </div>
  );
}

// A part of one option of a type (e.g. the light, the distance): its type,
// its lines (of an emission of lines), its parameters and other options
function TypedPart({ ctx, part, path, value, rings }: { ctx: Context; part: Part; path: Path; value: unknown; rings: number[] | null }) {
  const { defs, draft, params } = ctx;
  const family = part.typed!;
  const item = isObject(value) ? value : {};
  const set = (next: unknown) => draft.edit((s) => ({ ...s, options: setAt(s.options, path, next) }));
  const without = new Set([...HEAD_OPTIONS, ...(item.type === 'lines' ? ['lines'] : [])]);
  return (
    <>
      <div className="gx-row">
        <span className="gx-row-label">Type</span>
        <Choice items={typesOf(defs, family)} value={String(item.type ?? '')} onChange={(type) => set(retyped(defs, family, item, type))} label={`The type of the ${part.title.toLowerCase()}`} />
      </div>
      {family === 'emission' && item.type === 'lines' && <LinesField emission={item} onChange={set} />}
      <Params ctx={ctx} owned={ownedBy(params, path)} rings={rings} />
      <MoreOptions ctx={ctx} family={family} item={item} path={path} without={without} />
    </>
  );
}

// A part of a list of items of a family (e.g. the velocity traits, the
// mass components): a card for each, and a menu that adds one, in the head
// of the inspector (always in sight, however long the list); the one
// added is scrolled to
function ListPart({
  ctx,
  part,
  base,
  rings,
  kind,
  isGalaxy,
  head,
}: {
  ctx: Context;
  part: Part;
  base: Path;
  rings: number[] | null;
  kind: string;
  isGalaxy: boolean;
  head: HTMLElement | null;
}) {
  const { defs, draft, options } = ctx;
  const family = part.family!;
  const listPath = [...base, ...part.option];
  const items = asList(getAt(options, listPath));
  const traits = useComponentTraits().data as Record<string, { traits: Record<string, string[]>; directions: string[]; dispersion_directions: string[] }> | undefined;
  const allowed = traits?.[kind]?.traits[family] ?? null;
  const heights = part.heights;
  const spheroids = asList(options.components)
    .filter((c) => c.type === 'spheroid')
    .map((c) => String(c.name ?? ''));
  const names = items.map((item) => String(item.name ?? ''));
  const end = useRef<HTMLSpanElement>(null);
  const [added, setAdded] = useState(0);
  useEffect(() => {
    if (added > 0) (end.current?.previousElementSibling as HTMLElement | null)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }, [added]);
  const add = (type: string) => {
    setAdded((n) => n + 1);
    const follows = family === 'mass_component' && type === 'light_spheroid';
    const start = startOf(defs, family, type, names, follows ? { component: spheroids[0] } : {});
    draft.edit((s) => {
      let next = setAt(s.options, listPath, asOneOrList([...asList(getAt(s.options, listPath)), start]) ?? []);
      // (a density trait of a thick disk or a cone comes with its height)
      if (heights && !heights.nullable) {
        const hpath = [...base, ...heights.option];
        next = setAt(next, hpath, asOneOrList([...asList(getAt(next, hpath)), { ...HEIGHT_START }]));
      }
      return { ...s, options: next };
    });
  };
  const remove = (index: number) => {
    const hpath = heights ? [...base, ...heights.option] : null;
    const moves = [removedFrom(listPath, index), ...(hpath ? [removedFrom(hpath, index)] : [])];
    draft.edit(
      (s) => {
        let next = setAt(s.options, [...listPath, index], undefined);
        if (hpath) {
          const list = getAt(next, hpath);
          if (Array.isArray(list)) next = setAt(next, [...hpath, index], undefined);
          else if (list !== undefined && index === 0) next = setAt(next, hpath, undefined);
        }
        return { ...s, options: next };
      },
      { moved: (owner) => moves.reduce<Path | null>((path, move) => (path === null ? null : move(path)), owner) },
    );
  };
  const removable = items.length > 1 || part.optional || isGalaxy;
  return (
    <>
      {items.length === 0 && <p className="note gx-none">None yet.</p>}
      {items.map((item, i) => (
        <ItemCard
          key={i}
          ctx={ctx}
          family={family}
          path={[...listPath, i]}
          item={item}
          index={i}
          count={items.length}
          allowed={allowed}
          directions={family === 'vpt' ? (traits?.[kind]?.directions ?? null) : family === 'dpt' ? (traits?.[kind]?.dispersion_directions ?? null) : null}
          spheroids={spheroids}
          heights={heights ? { ...heights, path: [...base, ...heights.option, i], allowed: traits?.[kind]?.traits[heights.family] ?? null } : undefined}
          rings={rings}
          onRemove={removable ? () => remove(i) : undefined}
          moreOpen={family === 'mass_component'}
        />
      ))}
      <span ref={end} hidden />
      {head &&
        createPortal(
          <DropMenu
            label={`Add to the ${part.title.toLowerCase()}`}
            width={MENU_WIDTH}
            at="end"
            entries={typesOf(defs, family, allowed).map((t) => {
              const follows = family === 'mass_component' && t.value === 'light_spheroid';
              const blocked = follows && spheroids.length === 0;
              return {
                label: t.label,
                description: blocked ? 'Add a spheroid component first: this mass follows its light.' : t.description,
                disabled: blocked,
                onClick: () => add(t.value),
              };
            })}
            target={<button className="gx-add-item">+ Add {ITEM_NOUNS[family] ?? 'an item'}</button>}
          />,
          head,
        )}
    </>
  );
}

// What an item of each family is called, to add one
const ITEM_NOUNS: Record<string, string> = {
  rpt: 'a density',
  vpt: 'a motion',
  dpt: 'a dispersion',
  zpt: 'a warp',
  spt: 'a selection',
  mass_component: 'a mass component',
};

// An item of a list: its type, direction and the like, its paired height,
// its parameters and its other options
function ItemCard({
  ctx,
  family,
  path,
  item,
  index,
  count,
  allowed,
  directions,
  spheroids,
  heights,
  rings,
  onRemove,
  moreOpen,
}: {
  ctx: Context;
  family: string;
  path: Path;
  item: Options;
  index: number;
  count: number;
  allowed: string[] | null;
  directions: string[] | null;
  spheroids: string[];
  heights?: { path: Path; family: string; title: string; nullable: boolean; allowed: string[] | null };
  rings: number[] | null;
  onRemove?: () => void;
  moreOpen: boolean;
}) {
  const { defs, draft, params } = ctx;
  const member = memberOf(defs, family, item.type);
  const properties = (member?.properties ?? {}) as Record<string, Schema>;
  const set = (next: unknown, how?: EditHow) => draft.edit((s) => ({ ...s, options: setAt(s.options, path, next) }), how);
  // (the directions the component takes, titled by the schema)
  const directionChoices = choicesOf(properties.direction)?.filter((d) => directions === null || directions.includes(String(d.value))) ?? null;
  // (a mass component is named, and others name it: in its head; the name
  // of a trait is among its options)
  const named = family === 'mass_component' && 'name' in properties;
  return (
    <div className="gx-card">
      <div className="gx-card-head">
        {count > 1 && <span className="gx-card-index">{index + 1}</span>}
        <Choice
          items={typesOf(defs, family, allowed)}
          value={String(item.type ?? '')}
          onChange={(type) => set(retyped(defs, family, item, type))}
          label={`The type of item ${index + 1}`}
        />
        {directionChoices && (
          <Choice
            items={directionChoices.map((d) => ({ value: String(d.value), label: d.title }))}
            // (one not given is the default, e.g. isotropic)
            value={String(item.direction ?? properties.direction?.default ?? '')}
            onChange={(direction) => set({ ...item, direction })}
            label="Its direction"
          />
        )}
        {onRemove && <Close label="Remove it" onClick={onRemove} />}
      </div>
      {(named || 'component' in properties) && (
        <div className="gx-card-sub">
          {named && <NameField value={typeof item.name === 'string' ? item.name : ''} required={family === 'mass_component'} onCommit={(name) => set(name ? { ...item, name } : omitKey(item, 'name'), { moved: inPlace })} />}
          {'component' in properties && (
            <Choice items={spheroids.map((s) => ({ value: s, label: `follows ${s}` }))} value={String(item.component ?? '')} onChange={(component) => set({ ...item, component })} label="The spheroid whose light it follows" />
          )}
        </div>
      )}
      <Params ctx={ctx} owned={ownedBy(params, path)} rings={rings} />
      <MoreOptions ctx={ctx} family={family} item={item} path={path} without={family === 'mass_component' ? MASS_HEAD_OPTIONS : HEAD_OPTIONS} open={moreOpen} />
      {heights && <HeightRow ctx={ctx} heights={heights} rings={rings} />}
    </div>
  );
}

function omitKey(options: Options, key: string): Options {
  const { [key]: _, ...rest } = options;
  return rest;
}

// The height trait paired with an item (of a thick disk or a cone): its
// type (or none, if it can be), its parameters and options
function HeightRow({ ctx, heights, rings }: { ctx: Context; heights: { path: Path; family: string; title: string; nullable: boolean; allowed: string[] | null }; rings: number[] | null }) {
  const { defs, draft, options, params } = ctx;
  const value = getAt(options, heights.path);
  const item = isObject(value) ? value : null;
  const set = (next: unknown) =>
    draft.edit((s) => {
      // (a list of heights of nulls up to this one, as it is one of a list)
      const list = heights.path.slice(0, -1);
      const index = heights.path[heights.path.length - 1] as number;
      const current = getAt(s.options, list);
      const items: unknown[] = Array.isArray(current) ? [...current] : current === undefined ? [] : [current];
      while (items.length <= index) items.push(null);
      items[index] = next;
      const trimmed = items.every((h) => h === null) ? undefined : items.length === 1 ? items[0] : items;
      return { ...s, options: setAt(s.options, list, trimmed) };
    });
  const types = typesOf(defs, heights.family, heights.allowed);
  return (
    <div className="gx-height">
      <div className="gx-row">
        <span className="gx-row-label">{heights.title}</span>
        <Choice
          items={[...(heights.nullable ? [{ value: '', label: 'the same at every height' }] : []), ...types]}
          value={String(item?.type ?? '')}
          onChange={(type) => set(type === '' ? null : retyped(defs, heights.family, item, type))}
          label={heights.title}
        />
      </div>
      {item && (
        <>
          <Params ctx={ctx} owned={ownedBy(params, heights.path)} rings={rings} limit={typeof item.zmax === 'number' ? item.zmax : undefined} />
          <MoreOptions ctx={ctx} family={heights.family} item={item} path={heights.path} without={HEAD_OPTIONS} title={`${heights.title} options`} />
        </>
      )}
    </div>
  );
}

// A name, as typed: set as it is entered or left
function NameField({ value, onCommit, required }: { value: string; onCommit: (name: string) => void; required: boolean }) {
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    if (text === null) return;
    const name = text.trim();
    if (name !== value && (name || !required) && /^[A-Za-z_][A-Za-z0-9_]*$|^$/.test(name)) onCommit(name);
    setText(null);
  };
  return (
    <label className="gx-name">
      <span>name</span>
      <input
        value={text ?? value}
        placeholder="none"
        onChange={(event) => setText(event.currentTarget.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === 'Enter') event.currentTarget.blur();
          if (event.key === 'Escape') {
            setText(null);
            event.currentTarget.blur();
          }
        }}
        aria-label="Its name"
        spellCheck={false}
      />
    </label>
  );
}

// The geometry of a component beyond its parameters: the radii of the
// rings of a disk or cone (evenly spaced to a radius, every sep or len of
// them, or given), whether the rings of a disk tilt (posa and incl per
// ring: a warp) or wander (centre and vsys per ring); how far the light of
// a spheroid is drawn
function GeometryFields({ component, onChange }: { component: Options; onChange: (component: Options) => void }) {
  const type = component.type;
  if (type === 'spheroid') {
    return (
      <div className="gx-row">
        <span className="gx-row-label">Drawn to</span>
        <NumberOption value={Number(component.rmax)} onCommit={(rmax) => onChange({ ...component, rmax })} label="rmax" unit="″" />
      </div>
    );
  }
  if (!isDisk(type) && type !== 'cone') return null;
  const flag = (key: string, on: boolean) => (on ? { ...component, [key]: true } : omitKey(component, key));
  return (
    <>
      <RingsField value={component.rnodes} onChange={(rnodes) => onChange({ ...component, rnodes })} title={type === 'cone' ? 'Shells' : 'Rings'} />
      {isDisk(type) && (
        <div className="gx-flags">
          <Check checked={component.tilted === true} onChange={(on) => onChange(flag('tilted', on))} label="Tilted rings: orientation per ring (a warp)" />
          <Check checked={component.loose === true} onChange={(on) => onChange(flag('loose', on))} label="Loose rings: centre and systemic velocity per ring" />
        </div>
      )}
    </>
  );
}

type RingsForm = 'sep' | 'len' | 'list';

// The radii of the rings: {max, sep}, {max, len} or a list of them; a
// change of form keeps the reach of the rings and about their number
function RingsField({ value, onChange, title }: { value: unknown; onChange: (value: unknown) => void; title: string }) {
  const form: RingsForm = Array.isArray(value) ? 'list' : isObject(value) && 'len' in value ? 'len' : 'sep';
  const radii = ringRadii(value);
  const max = radii.length ? radii[radii.length - 1] : 16;
  const options = isObject(value) ? value : {};
  const toForm = (next: RingsForm) => {
    const count = Math.max(radii.length, 2);
    if (next === 'sep') onChange({ max, sep: roundStep(max / (count - 1)) });
    else if (next === 'len') onChange({ max, len: count });
    else onChange(radii);
  };
  return (
    <div className="gx-rings">
      <div className="gx-row">
        <span className="gx-row-label">{title}</span>
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
        <div className="gx-rings-row">
          <span>to</span>
          <NumberOption value={Number(options.max)} onCommit={(m) => onChange({ ...options, max: m })} label="max" unit="″" />
          {form === 'sep' ? (
            <>
              <span>every</span>
              <NumberOption value={Number(options.sep)} onCommit={(sep) => onChange({ ...options, sep })} label="sep" unit="″" />
            </>
          ) : (
            <>
              <span>in</span>
              <NumberOption value={Number(options.len)} onCommit={(len) => onChange({ ...options, len: Math.max(2, Math.round(len)) })} label="len" unit="" />
            </>
          )}
        </div>
      )}
      <RingStrip radii={radii} />
    </div>
  );
}

// The rings at their radii, as ticks along a line
function RingStrip({ radii }: { radii: number[] }) {
  if (radii.length === 0) return <p className="note">No rings: give them a radius.</p>;
  const max = radii[radii.length - 1] || 1;
  return (
    <div className="gx-ring-strip" title={radii.map((r) => `${r}″`).join(', ')}>
      <div className="gx-ring-line">
        {radii.map((r, i) => (
          <i key={i} style={{ left: `${(r / max) * 100}%` }} />
        ))}
      </div>
      <span>
        {radii.length} {radii.length === 1 ? 'ring' : 'rings'} · 0 to {Number(max.toPrecision(4))}″
      </span>
    </div>
  );
}

// A positive number, as typed: set once it reads as one
function NumberOption({ value, onCommit, label, unit }: { value: number; onCommit: (value: number) => void; label: string; unit: string }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <label className="gx-value gx-number">
      <input
        value={text ?? (Number.isFinite(value) ? String(value) : '')}
        onChange={(event) => {
          const next = event.currentTarget.value;
          setText(next);
          const number = Number(next);
          if (next.trim() !== '' && Number.isFinite(number) && number > 0) onCommit(number);
        }}
        onBlur={() => setText(null)}
        aria-label={label}
        inputMode="decimal"
      />
      {unit && <span>{unit}</span>}
    </label>
  );
}

// Radii given as a list, as typed (by commas or spaces): set once each
// reads as a number
function RadiiOption({ radii, onCommit }: { radii: number[]; onCommit: (radii: number[]) => void }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <textarea
      className="gx-radii"
      value={text ?? radii.join(', ')}
      rows={2}
      onChange={(event) => {
        const next = event.currentTarget.value;
        setText(next);
        const numbers = next.split(/[\s,]+/).filter(Boolean).map(Number);
        if (numbers.length >= 2 && numbers.every(Number.isFinite)) onCommit(numbers);
      }}
      onBlur={() => setText(null)}
      aria-label="The radii (arcsec)"
      spellCheck={false}
    />
  );
}

// The lines of an emission of lines: by default one line at the velocity
// of the spectral axis, whatever it is; else lines of the catalogue (or of
// the configuration, kept as they are), the first the one whose flux the
// density gives, the others of a ratio of flux to it
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
    <div className="gx-lines">
      <span className="gx-row-label">Lines</span>
      {lines.length === 0 ? (
        <p className="note">One line, at the velocity of the spectral axis, whatever line it is.</p>
      ) : (
        <div className="gx-line-list">
          {lines.map((line, i) => (
            <span key={i} className="gx-line" title={rests.get(lineName(line)) ?? 'A line of the configuration'}>
              <b>{lineName(line)}</b>
              <small>{i === 0 ? 'flux' : 'ratio'}</small>
              <Close label={`Remove the line ${lineName(line)}`} onClick={() => set(lines.filter((_, j) => j !== i))} />
            </span>
          ))}
        </div>
      )}
      <DropMenu
        label="Add a line"
        width={260}
        entries={catalogue.filter((l) => !named.includes(l.name)).map((l) => ({ label: l.name, description: l.rest, onClick: () => set([...lines, l.name]) }))}
        target={<button className="gx-add-item">+ Add a line</button>}
      />
    </div>
  );
}
