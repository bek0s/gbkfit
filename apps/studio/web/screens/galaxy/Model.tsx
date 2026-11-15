// The model of a galaxy, as its sheet shows it: the galaxy as a whole (its
// distance, redshift and gravity, in a strip, and the parts of its mass,
// a line each, under them, in reach however long the components are),
// and its components, each one line (its name, its type, its extent and
// its parts in a few words) and one open at a time, with its parts in the
// order the library gives them (see types.ts), each its items with their
// numbers in its line. Options at their defaults stay out of sight,
// behind "N more options"; the parts a component can have but has not
// are a row of "+ ..." to add them. A problem shows where it is, and the
// count of them at the foot goes to each in turn.
import { Popover } from '@mantine/core';
import { useState, type ReactNode } from 'react';
import { useComponentTypes, useLines } from '../../api/queries';
import { choicesOf, SchemaForm, type Defs, type Schema } from '../../components/SchemaForm';
import { significant } from '../../components/format';
import { Close } from '../../components/ui/basics';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import type { EditHow, GalaxyDraft, Problem } from './draft';
import { NumberChip, NumberChips, paramText, type NumbersContext } from './Numbers';
import { asList, asOneOrList, componentName, componentsOf, freshName, getAt, isObject, lineList, lineName, ringRadii, roundStep, setAt, startsWith, type Options, type Path } from './options';
import { ownedBy, settingOf } from './params';
import { memberOf, restOf, retyped, startOf, typesOf, type TypeChoice } from './schema';
import { extentOf, HEAD_OPTIONS, nameBase, partsOfType, referencesOf, retypedComponent, settingsOfType, type ComponentTypes, type PartInfo } from './types';

// What is open in the sheet: a component or a part of the mass
export type Open = { kind: 'component' | 'mass'; index: number } | null;

interface Ctx extends NumbersContext {
  defs: Defs | undefined;
  types: ComponentTypes | undefined;
  project: string;
  problems: Problem[];
  set: (path: Path, value: unknown, how?: EditHow) => void;
  // The length of an arcsec at the distance of the galaxy (kpc), if known
  kpcPerArcsec: number | null;
}

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
// component renamed, without moving it)
export const inPlace = (owner: Path): Path => owner;

export interface ModelProps {
  draft: GalaxyDraft;
  project: string;
  field: number;
  colours: string[];
  open: Open;
  onOpen: (open: Open) => void;
  isolated: string | null;
  onIsolate: (name: string | null) => void;
}

export function Model({ draft, project, field, colours, open, onOpen, isolated, onIsolate }: ModelProps) {
  const types = useComponentTypes().data as ComponentTypes | undefined;
  const options = draft.state?.options ?? {};
  const components = componentsOf(options);
  const mass = asList(options.mass);
  const set = (path: Path, value: unknown, how?: EditHow) => draft.edit((s) => ({ ...s, options: setAt(s.options, path, value) }), how);
  const ctx: Ctx = { draft, options, params: draft.params, field, defs: draft.defs, types, project, problems: draft.problems, set, kpcPerArcsec: kpcPerArcsecOf(draft) };
  const colourOf = (name: unknown) => {
    const at = components.findIndex((c, i) => componentName(c, i) === name);
    return at < 0 ? 'var(--faint)' : `var(--${colours[at]})`;
  };

  const addComponent = (type: string) => {
    draft.edit(
      (s) => {
        const list = componentsOf(s.options);
        // (several components are all named: one without a name is named
        // after its type, and its parameters with it)
        const named = list.map((c, i) => (typeof c.name === 'string' && c.name ? c : { ...c, name: freshName(list.map((o) => o.name), c.type ? nameBase(String(c.type)) : `component${i + 1}`) }));
        const added = { ...startOf(draft.defs, 'component', type), name: freshName(named.map((c) => c.name), nameBase(type)) };
        return { ...s, options: { ...s.options, components: [...named, added] } };
      },
      { moved: inPlace },
    );
    onOpen({ kind: 'component', index: components.length });
  };

  const addMass = (type: string) => {
    const refs = referencesOf(draft.defs, 'mass_component', type);
    // (one that follows a component: the first of its kind, e.g. a traced
    // disk the first disk; else the first)
    const kin = components.filter((c) => nameBase(String(c.type)) === nameBase(type));
    const follow = kin[0] ?? components[0];
    const extra = Object.fromEntries(Object.entries(refs).filter(([, what]) => what === 'components').map(([key]) => [key, follow ? componentName(follow, components.indexOf(follow)) : undefined]));
    draft.edit((s) => {
      let next = s.options;
      // (mass needs the distance of its galaxy, to turn arcsec into kpc)
      if (next.distance === undefined) next = setAt(next, ['distance'], { type: 'parameter' });
      const names = asList(next.mass).map((m) => m.name);
      const base = typeof extra.component === 'string' && !names.includes(extra.component) ? extra.component : nameBase(type);
      const item = startOf(draft.defs, 'mass_component', type, [], { ...extra, name: freshName(names, base) });
      return { ...s, options: setAt(next, ['mass'], [...asList(next.mass), item]) };
    });
    onOpen({ kind: 'mass', index: mass.length });
  };

  const componentTypes = Object.keys(types ?? {}).map((type) => ({ value: type, ...titled(draft.defs, 'component', type) }));
  const massTypes = typesOf(draft.defs, 'mass_component');
  const general = draft.problems.filter((p) => !startsWith(p.path, ['components']) && !startsWith(p.path, ['mass']));
  // (the problems of a component or a part of the mass: the count goes to
  // each in turn, opening it and bringing its first problem in sight)
  const placed = draft.problems.filter((p) => !general.includes(p) && typeof p.path[1] === 'number');
  const [turn, setTurn] = useState(0);
  const goToProblem = () => {
    const problem = placed[turn % placed.length];
    setTurn(turn + 1);
    const [section, index] = problem.path as [string, number];
    onOpen({ kind: section === 'mass' ? 'mass' : 'component', index });
    // (once it is open)
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        const row = document.querySelector(`[data-at="${section}/${index}"]`);
        (row?.querySelector('.gm-why') ?? row)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }),
    );
  };

  return (
    <div className="gm">
      <Strip ctx={ctx} hasMass={mass.length > 0} />

      <div className="gm-sec">
        <span className="gm-over">
          Mass<b>{mass.length}</b>
        </span>
        <DropMenu
          label="Add a part of the mass"
          width={320}
          at="end"
          entries={massTypes.map((t) => ({ label: t.label, description: t.description, onClick: () => addMass(t.value) }))}
          target={<button className="gm-add">+ Add</button>}
        />
      </div>
      {mass.length === 0 ? (
        <p className="gm-empty">No mass: the motions are as the traits give them.</p>
      ) : (
        <div className="gm-mass">
          {mass.map((item, i) => (
            <MassRow
              key={i}
              ctx={ctx}
              index={i}
              item={item}
              colourOf={colourOf}
              names={components.map(componentName)}
              open={open?.kind === 'mass' && open.index === i}
              onToggle={() => onOpen(open?.kind === 'mass' && open.index === i ? null : { kind: 'mass', index: i })}
              onRemoved={() => onOpen(null)}
            />
          ))}
        </div>
      )}

      <div className="gm-sec">
        <span className="gm-over">
          Components<b>{components.length}</b>
        </span>
        <DropMenu
          label="Add a component"
          width={320}
          at="end"
          entries={componentTypes.map((t) => ({ label: t.label, description: t.description, onClick: () => addComponent(t.value) }))}
          target={<button className="gm-add">+ Add</button>}
        />
      </div>
      {components.length === 0 && <p className="gm-empty">No components yet: add one to see the galaxy.</p>}
      {components.map((component, i) => (
        <ComponentCard
          key={i}
          ctx={ctx}
          index={i}
          component={component}
          colour={colours[i]}
          open={open?.kind === 'component' && open.index === i}
          onToggle={() => onOpen(open?.kind === 'component' && open.index === i ? null : { kind: 'component', index: i })}
          isolated={isolated === componentName(component, i)}
          onIsolate={(on) => onIsolate(on ? componentName(component, i) : null)}
          onOpen={onOpen}
        />
      ))}

      <div className="gm-problems" data-bad={draft.problems.length > 0 || undefined}>
        {draft.problems.length === 0 ? (
          <span>No problems: the library builds the galaxy.</span>
        ) : (
          <span>
            {placed.length > 0 ? (
              <button className="gm-link" onClick={goToProblem} title="Open where the next one is">
                {draft.problems.length} {draft.problems.length === 1 ? 'problem' : 'problems'}
              </button>
            ) : (
              `${draft.problems.length} ${draft.problems.length === 1 ? 'problem' : 'problems'}`
            )}
            {placed.length > 0 && ', shown where they are'}
          </span>
        )}
        {general.map((p, i) => (
          <p key={i} className="gm-why">
            {problemText(p.message)}
          </p>
        ))}
        {draft.describeError && <p className="gm-why">{draft.describeError.message}</p>}
      </div>
    </div>
  );
}

// The length of an arcsec at the distance of a galaxy (kpc): of its
// parameter of distance in Mpc (D times pi / 648000 Mpc, in kpc), if it has
// one with a value
export function kpcPerArcsecOf(draft: GalaxyDraft): number | null {
  const param = draft.params.find((p) => p.owner.length === 1 && p.owner[0] === 'distance');
  if (!param || param.unit !== 'Mpc') return null;
  const setting = settingOf(draft.state?.properties ?? {}, param);
  const mpc = setting.kind === 'fixed' || setting.kind === 'free' ? setting.values[0] : undefined;
  return typeof mpc === 'number' && mpc > 0 ? (mpc * 1000 * Math.PI) / 648000 : null;
}

// The title of a type of a family and what it is
function titled(defs: Defs | undefined, family: string, type: string): { label: string; description?: string } {
  return typesOf(defs, family).find((t) => t.value === type) ?? { label: type };
}

// ---------------------------------------------------------------------
// The galaxy as a whole

// The galaxy as a whole: its distance, redshift and gravity, each a tile.
// A click on the tile of the distance or the gravity opens the menu of
// their types; their numbers are chips in the tile, each opening to set;
// the redshift is typed in its tile.
function Strip({ ctx, hasMass }: { ctx: Ctx; hasMass: boolean }) {
  const { defs, options, params, draft, set, project } = ctx;
  const distance = isObject(options.distance) ? options.distance : null;
  const gravity = isObject(options.gravity) ? options.gravity : null;
  const distanceParam = params.find((p) => p.owner.length === 1 && p.owner[0] === 'distance');
  const setting = distanceParam ? settingOf(draft.state?.properties ?? {}, distanceParam) : null;
  const scale = ctx.kpcPerArcsec === null ? null : `1″ = ${significant(ctx.kpcPerArcsec * 1000, 3)} pc`;
  const distanceType = String(distance?.type ?? '');
  const gravityType = String(gravity?.type ?? 'newton');
  const z = typeof options.redshift === 'number' ? options.redshift : 0;
  const gravityParams = ownedBy(params, ['gravity']);
  const gravityRest = restOf(defs, memberOf(defs, 'gravity', gravity?.type), new Set(['type']));
  const typeEntries = (family: string, current: string, onChange: (type: string) => void, none?: string): MenuEntry[] => [
    ...(none ? [{ label: none, current: current === '', onClick: () => onChange('') }] : []),
    ...typesOf(defs, family).map((t) => ({ label: t.label, description: t.description, current: t.value === current, onClick: () => t.value !== current && onChange(t.value) })),
  ];
  let distanceValue: ReactNode = <span className="gm-tv">None</span>;
  let distanceNote = 'angles only';
  if (distance && distanceParam && setting) {
    distanceValue = <NumberChip ctx={ctx} param={distanceParam} rings={null} bare />;
    distanceNote = scale ?? '';
  } else if (distance) {
    distanceValue = <span className="gm-tv">{titled(defs, 'distance', distanceType).label}</span>;
    distanceNote = scale ?? 'of its redshift';
  }
  return (
    <div className="gm-strip">
      <TypeTile
        label="Distance"
        menu="The distance of the galaxy"
        entries={typeEntries('distance', distanceType, (type) => set(['distance'], type === '' ? undefined : retyped(defs, 'distance', distance, type)), 'None')}
        note={distanceNote}
      >
        {distanceValue}
      </TypeTile>
      <div className="gm-tile gm-tile-z">
        <span className="gm-tl">Redshift</span>
        <NumberText value={z} label="The redshift of the galaxy" onCommit={(value) => set(['redshift'], value > 0 ? value : undefined)} bare />
        <span className="gm-tc">its lines and dimming</span>
      </div>
      <TypeTile
        label="Gravity"
        menu="The gravity of the galaxy"
        entries={typeEntries('gravity', gravityType, (type) => set(['gravity'], type === 'newton' ? undefined : retyped(defs, 'gravity', gravity, type)))}
        note={hasMass ? 'of the mass below' : 'with mass only'}
      >
        <span className="gm-tv">{titled(defs, 'gravity', gravityType).label}</span>
        {gravityParams.map((p) => (
          <NumberChip key={p.name} ctx={ctx} param={p} rings={null} />
        ))}
        {gravityRest && defs && gravity && (
          <Popover position="bottom-start" offset={4} width={340} shadow="md" withinPortal trapFocus={false} classNames={{ dropdown: 'gm-pop' }}>
            <Popover.Target>
              <button className="gm-opt" title="Its other options">
                options
              </button>
            </Popover.Target>
            <Popover.Dropdown>
              <div className="gm-popbody">
                <SchemaForm defs={defs} node={gravityRest} project={project} value={omit(gravity, 'type')} onChange={(next) => set(['gravity'], { type: gravity.type, ...next })} />
              </div>
            </Popover.Dropdown>
          </Popover>
        )}
      </TypeTile>
    </div>
  );
}

// A tile of the strip of a choice of types (the distance, the gravity): a
// click opens the menu of its types; its value, and the chips of its
// numbers, which open on their own
function TypeTile({ label, menu, entries, note, children }: { label: string; menu: string; entries: MenuEntry[]; note: string; children: ReactNode }) {
  return (
    <DropMenu
      label={menu}
      width={300}
      entries={entries}
      target={
        <div
          className="gm-tile"
          role="button"
          tabIndex={0}
          title={`The ${label.toLowerCase()}: click to change its kind`}
          onKeyDown={(event) => event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ') && (event.preventDefault(), event.currentTarget.click())}
        >
          <span className="gm-tl">
            {label}
            <span className="gm-caret" aria-hidden>
              ▾
            </span>
          </span>
          {/* (the chips and what they open are their own, not the menu's;
          the value in words is the tile's) */}
          <span className="gm-tvs" onClick={(event) => event.target !== event.currentTarget && !(event.target as HTMLElement).classList.contains('gm-tv') && event.stopPropagation()}>
            {children}
          </span>
          <span className="gm-tc">{note || ' '}</span>
        </div>
      }
    />
  );
}

// ---------------------------------------------------------------------
// A component

function ComponentCard({
  ctx,
  index,
  component,
  colour,
  open,
  onToggle,
  isolated,
  onIsolate,
  onOpen,
}: {
  ctx: Ctx;
  index: number;
  component: Options;
  colour: string;
  open: boolean;
  onToggle: () => void;
  isolated: boolean;
  onIsolate: (on: boolean) => void;
  onOpen: (open: Open) => void;
}) {
  const { defs, types, problems, draft, options, params, set } = ctx;
  const [renaming, setRenaming] = useState(false);
  const base: Path = ['components', index];
  const components = componentsOf(options);
  const parts = partsOfType(defs, types, component.type);
  const radii = ringRadii(component.rnodes);
  const own = problems.filter((p) => startsWith(p.path, base));
  const typeTitle = titled(defs, 'component', String(component.type)).label;

  const rename = (name: string) => {
    setRenaming(false);
    if (!name || components.some((c, i) => i !== index && c.name === name)) return;
    set([...base, 'name'], name, { moved: inPlace });
  };
  const duplicate = () => {
    const oldName = typeof component.name === 'string' ? component.name : null;
    const copyName = freshName(
      components.map((c) => c.name),
      `${oldName ?? String(component.type)}_copy`.replace(/^_/, ''),
    );
    draft.edit(
      (s) => {
        const list = componentsOf(s.options);
        // (its values too: the parameters of a named component are named
        // after it)
        const properties = { ...s.properties };
        if (oldName) {
          for (const p of ownedBy(params, base)) {
            if (p.name.startsWith(`${oldName}_`) && p.name in s.properties) properties[`${copyName}${p.name.slice(oldName.length)}`] = s.properties[p.name];
          }
        }
        return { options: { ...s.options, components: [...list.slice(0, index + 1), { ...structuredClone(component), name: copyName }, ...list.slice(index + 1)] }, properties };
      },
      // (those after it, a place on)
      { moved: (owner) => (owner[0] === 'components' && typeof owner[1] === 'number' && owner[1] > index ? ['components', owner[1] + 1, ...owner.slice(2)] : owner) },
    );
    onOpen({ kind: 'component', index: index + 1 });
  };
  const remove = () => {
    if (isolated) onIsolate(false);
    set(base, undefined, { moved: removedFrom(['components'], index) });
    onOpen(null);
  };
  const typeEntries: MenuEntry[] = Object.keys(types ?? {}).map((type) => ({
    label: titled(defs, 'component', type).label,
    description: titled(defs, 'component', type).description,
    current: type === component.type,
    onClick: () => type !== component.type && set(base, retypedComponent(defs, types, component, type)),
  }));
  return (
    <div className="gm-comp" data-open={open || undefined} data-bad={own.length > 0 || undefined} data-at={`components/${index}`}>
      <div
        className="gm-crow"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={() => !renaming && onToggle()}
        onKeyDown={(event) => {
          if (renaming || event.target !== event.currentTarget) return;
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onToggle();
          }
        }}
      >
        <span className="gm-dot" style={{ background: `var(--${colour})` }} />
        <span className="gm-cname">
          {renaming ? (
            <span onClick={(event) => event.stopPropagation()}>
              <NameText value={typeof component.name === 'string' ? component.name : ''} required onCommit={rename} onCancel={() => setRenaming(false)} autoFocus />
            </span>
          ) : (
            componentName(component, index)
          )}
          <span>{typeTitle}</span>
        </span>
        <span className="gm-cext">{extentOf(component, radii)}</span>
        <span className="gm-cmenu" onClick={(event) => event.stopPropagation()}>
          <DropMenu
            label={`The component ${componentName(component, index)}`}
            width={240}
            at="end"
            entries={[
              { label: 'Rename', onClick: () => setRenaming(true) },
              { label: 'Change its type', entries: typeEntries },
              { label: isolated ? 'Show all in the views' : 'Show it alone in the views', onClick: () => onIsolate(!isolated) },
              { label: 'Duplicate', onClick: duplicate },
              { label: 'Remove', onClick: remove },
            ]}
            target={
              <button className="gm-more" aria-label="Its actions" title="Rename, change its type, duplicate, remove">
                ⋯
              </button>
            }
          />
        </span>
        {!open && <span className="gm-csum">{glanceOf(ctx, component, parts)}</span>}
      </div>
      {open && <ComponentBody ctx={ctx} index={index} component={component} parts={parts} rings={radii.length ? radii : null} />}
    </div>
  );
}

// A component in a few words: its parts, each its items' types (a height
// after its trait, a direction after its speed), and the lines it emits
function glanceOf(ctx: Ctx, component: Options, parts: PartInfo[]): ReactNode {
  const { defs } = ctx;
  const words: ReactNode[] = [];
  for (const part of parts) {
    const value = component[part.option];
    if (value === undefined || value === null || part.family === null) continue;
    const items = asList(value);
    const heights = part.heights ? (Array.isArray(component[part.heights.option]) ? (component[part.heights.option] as unknown[]) : [component[part.heights.option]]) : [];
    const text = items.map((item, i) => {
      const height = heights.length === 1 ? (i === items.length - 1 ? heights[0] : undefined) : heights[i];
      return (
        <span key={i}>
          {i > 0 && ' + '}
          {titled(defs, part.family!, String(item.type)).label.toLowerCase()}
          {typeof item.direction === 'string' && <span className="gm-dir">{item.direction}</span>}
          {isObject(height) && ` × ${titled(defs, part.heights!.family!, String(height.type)).label.toLowerCase()}`}
          {part.family === 'emission' && item.type === 'lines' && lineList(item.lines).length > 0 && ` ${lineList(item.lines).map(lineName).join(', ')}`}
        </span>
      );
    });
    words.push(<span key={part.option}>{text}</span>);
  }
  return words.flatMap((word, i) => (i ? [<span key={`s${i}`}> · </span>, word] : [word]));
}

function ComponentBody({ ctx, index, component, parts, rings }: { ctx: Ctx; index: number; component: Options; parts: PartInfo[]; rings: number[] | null }) {
  const { params, problems } = ctx;
  const base: Path = ['components', index];
  const present = parts.filter((p) => component[p.option] !== undefined || !p.optional);
  const absent = parts.filter((p) => component[p.option] === undefined && p.optional);
  const frame = ownedBy(params, base, true);
  // (the problems of the component itself, not of one of its parts)
  const partOptions = new Set(parts.flatMap((p) => [p.option, ...(p.heights ? [p.heights.option] : [])]));
  const general = problems.filter((p) => startsWith(p.path, base) && !(typeof p.path[2] === 'string' && partOptions.has(p.path[2])));

  const rows: ReactNode[] = [];
  const frameRow = frame.length > 0 && (
    <div className="gm-fam" key="frame">
      <span className="gm-fl" title="Its centre, orientation and systemic velocity, and its other parameters of its own">
        Frame
      </span>
      <div className="gm-fc">
        <div className="gm-trait">
          <NumberChips ctx={ctx} owned={frame} rings={rings} />
        </div>
      </div>
    </div>
  );
  if (!present.some((p) => p.def === 'Nodes')) rows.push(frameRow);
  for (const part of present) {
    rows.push(<PartRow key={part.option} ctx={ctx} part={part} base={base} rings={rings} component={component} />);
    if (part.def === 'Nodes') rows.push(frameRow);
  }

  return (
    <div className="gm-cbody">
      {general.map((p, i) => (
        <p key={i} className="gm-why gm-why-top">
          {problemText(p.message)}
        </p>
      ))}
      {rows}
      {absent.length > 0 && (
        <div className="gm-addfam">
          {absent.map((part) => (
            <AddPart key={part.option} ctx={ctx} part={part} base={base} />
          ))}
        </div>
      )}
      <Settings ctx={ctx} component={component} base={base} />
    </div>
  );
}

// A part a component does not have, to add: of one type, added at once;
// else its types to choose from
function AddPart({ ctx, part, base }: { ctx: Ctx; part: PartInfo; base: Path }) {
  const { defs, set } = ctx;
  const path = [...base, part.option];
  const label = `+ ${part.title.toLowerCase()}`;
  if (part.family === null) {
    const start = part.def && defs?.[part.def] ? startOfDef(defs[part.def]) : {};
    return (
      <button className="gm-addchip" title={part.description} onClick={() => set(path, start)}>
        {label}
      </button>
    );
  }
  const types = typesOf(defs, part.family, part.types);
  const add = (type: string) => set(path, withDirection(startOf(defs, part.family!, type), part));
  if (types.length === 1) {
    return (
      <button className="gm-addchip" title={part.description} onClick={() => add(types[0].value)}>
        {label}
      </button>
    );
  }
  return (
    <DropMenu
      label={`Add ${part.title.toLowerCase()}`}
      width={300}
      entries={types.map((t) => ({ label: t.label, description: t.description, onClick: () => add(t.value) }))}
      target={
        <button className="gm-addchip" title={part.description}>
          {label}
        </button>
      }
    />
  );
}

// The start of options of their own (e.g. the clouds of a disk): their
// example (see the library's start_config)
function startOfDef(def: Schema): Options {
  const example = (def.examples as unknown[] | undefined)?.[0];
  return isObject(example) ? example : {};
}

// A new trait along a direction the component takes (its start's, if it
// takes it; else the first)
function withDirection(item: Options, part: PartInfo): Options {
  if (!part.directions || typeof item.direction !== 'string' || part.directions.includes(item.direction)) return item;
  return { ...item, direction: part.directions[0] };
}

// The settings of a component: its options that are not its parts, out of
// sight behind "N more options" (their titles)
function Settings({ ctx, component, base }: { ctx: Ctx; component: Options; base: Path }) {
  const { defs, types, set, project } = ctx;
  const [open, setOpen] = useState(false);
  const keys = settingsOfType(defs, types, component.type);
  const others = Object.keys(propertiesOfType(defs, component.type)).filter((key) => !keys.includes(key));
  const node = restOf(defs, memberOf(defs, 'component', component.type), new Set(others));
  if (!defs || !node || keys.length === 0) return null;
  const properties = (node.properties ?? {}) as Record<string, Schema>;
  const titles = keys.map((key) => (typeof properties[key]?.title === 'string' ? (properties[key].title as string) : key).toLowerCase());
  const given = keys.filter((key) => component[key] !== undefined).length;
  return (
    <div className="gm-settings">
      <button className="gm-morel" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? 'Fewer options' : `${keys.length} more ${keys.length === 1 ? 'option' : 'options'}`}
        {!open && <span> · {titles.join(', ')}</span>}
        {given > 0 && <small>{given} set</small>}
      </button>
      {open && (
        <div className="gm-more-form">
          <SchemaForm
            defs={defs}
            node={node}
            project={project}
            value={Object.fromEntries(keys.filter((key) => component[key] !== undefined).map((key) => [key, component[key]]))}
            onChange={(next) => {
              const kept = Object.fromEntries(Object.entries(component).filter(([key]) => !keys.includes(key)));
              set(base, { ...kept, ...next }, { moved: inPlace });
            }}
          />
        </div>
      )}
    </div>
  );
}

// The options of a type of a family, by key
function propertiesOfType(defs: Defs | undefined, type: unknown): Record<string, Schema> {
  const member = memberOf(defs, 'component', type);
  return (member?.properties ?? {}) as Record<string, Schema>;
}

// ---------------------------------------------------------------------
// A part of a component

function PartRow({ ctx, part, base, rings, component }: { ctx: Ctx; part: PartInfo; base: Path; rings: number[] | null; component: Options }) {
  const { options, problems, set } = ctx;
  const path = [...base, part.option];
  const value = getAt(options, path);
  const own = problems.filter((p) => startsWith(p.path, path) || (part.heights !== null && startsWith(p.path, [...base, part.heights.option])));
  let content: ReactNode;
  if (part.def === 'Nodes') content = <RingsRow ctx={ctx} component={component} value={value} onChange={(next) => set(path, next)} />;
  else if (part.family !== null && part.many) content = <TraitList ctx={ctx} part={part} base={base} rings={rings} />;
  else if (part.family !== null)
    content = <OneItem ctx={ctx} part={part} path={path} rings={rings} onRemove={part.optional ? () => set(path, undefined) : undefined} />;
  else content = <ObjectItem ctx={ctx} part={part} path={path} onRemove={part.optional ? () => set(path, undefined) : undefined} />;
  return (
    <div className="gm-fam" data-bad={own.length > 0 || undefined}>
      <span className="gm-fl" title={part.description}>
        {part.title}
      </span>
      <div className="gm-fc">
        {value === undefined && !part.optional ? <AddPart ctx={ctx} part={part} base={base} /> : content}
        {own.map((p, i) => (
          <p key={i} className="gm-why">
            {problemText(p.message)}
          </p>
        ))}
      </div>
    </div>
  );
}

// A list of traits (e.g. the velocity of a disk): a line for each, with
// the height paired with it (or one that they share, after them), and what
// adds one
function TraitList({ ctx, part, base, rings }: { ctx: Ctx; part: PartInfo; base: Path; rings: number[] | null }) {
  const { defs, draft, options } = ctx;
  const family = part.family!;
  const listPath = [...base, part.option];
  const items = asList(getAt(options, listPath));
  const heights = part.heights;
  const hpath = heights ? [...base, heights.option] : null;
  const given = hpath ? getAt(options, hpath) : undefined;
  // (one height trait that every item shares: one alone, of several items;
  // else one for each item)
  const shared = !!heights && items.length > 1 && given !== undefined && given !== null && !Array.isArray(given);
  // (the height a new trait starts with: a copy of the last, else the start
  // of the first type the component takes)
  const newHeight = (): Options | null => {
    const last = asList(given).at(-1);
    if (last) return structuredClone(last);
    const first = heights?.family ? typesOf(defs, heights.family, heights.types)[0] : undefined;
    return first ? startOf(defs, heights!.family!, first.value) : null;
  };
  const add = (type: string) => {
    const start = withDirection(startOf(defs, family, type, items.map((item) => String(item.name ?? ''))), part);
    draft.edit((s) => {
      let next = setAt(s.options, listPath, asOneOrList([...asList(getAt(s.options, listPath)), start]) ?? []);
      // (a trait whose height must be given comes with one, unless they
      // share one)
      if (heights && !heights.nullable && !shared && hpath) {
        const height = newHeight();
        if (height) next = setAt(next, hpath, asOneOrList([...asList(getAt(next, hpath)), height]));
      }
      return { ...s, options: next };
    });
  };
  const remove = (index: number) => {
    const perItem = heights && !shared ? hpath : null;
    const moves = [removedFrom(listPath, index), ...(perItem ? [removedFrom(perItem, index)] : [])];
    draft.edit(
      (s) => {
        let next = setAt(s.options, [...listPath, index], undefined);
        if (perItem) {
          const list = getAt(next, perItem);
          if (Array.isArray(list)) next = setAt(next, [...perItem, index], undefined);
          else if (list !== undefined && index === 0) next = setAt(next, perItem, undefined);
        }
        return { ...s, options: next };
      },
      { moved: (owner) => moves.reduce<Path | null>((path, move) => (path === null ? null : move(path)), owner) },
    );
  };
  const share = (one: boolean) =>
    draft.edit((s) => {
      const current = Array.isArray(getAt(s.options, hpath!)) ? (getAt(s.options, hpath!) as unknown[]) : [getAt(s.options, hpath!)];
      const first = current.find(isObject) ?? newHeight();
      const next = one ? first : items.map((_, i) => structuredClone(isObject(current[i]) ? current[i] : heights!.nullable ? null : first));
      return { ...s, options: setAt(s.options, hpath!, next) };
    });
  const removable = items.length > 1 || part.optional;
  const types = typesOf(defs, family, part.types);
  // (heights that can be none show where they are given; none given, what
  // gives one to all)
  const hlist = Array.isArray(given) ? given : given === undefined || given === null ? [] : [given];
  const noHeights = !!heights && heights.nullable && !hlist.some(isObject);
  const shown = (i: number) => !!heights && !shared && (!heights.nullable || isObject(hlist[i]));
  return (
    <div className="gm-list">
      {items.map((item, i) => (
        <div key={i} className="gm-item">
          <ItemLine ctx={ctx} part={part} path={[...listPath, i]} item={item} rings={rings} onRemove={removable ? () => remove(i) : undefined} />
          {shown(i) && <HeightLine ctx={ctx} part={heights!} path={[...hpath!, i]} count={items.length} rings={rings} />}
        </div>
      ))}
      {heights && shared && <HeightLine ctx={ctx} part={heights} path={[...hpath!, 0]} count={1} rings={rings} all />}
      <div className="gm-list-foot">
        <DropMenu
          label={`Add to the ${part.title.toLowerCase()}`}
          width={300}
          entries={types.map((t) => ({ label: t.label, description: t.description, onClick: () => add(t.value) }))}
          target={<button className="gm-add gm-add-small">+ {part.title.toLowerCase()}</button>}
        />
        {noHeights && hpath && (
          <button className="gm-add gm-add-small gm-hover" title={heights!.description} onClick={() => ctx.set(hpath, newHeight())}>
            + height
          </button>
        )}
        {heights && !noHeights && items.length > 1 && (shared || given !== undefined) && (
          <button className="gm-link gm-hover" onClick={() => share(!shared)}>
            {shared ? 'a height for each' : 'one height for all'}
          </button>
        )}
      </div>
    </div>
  );
}

// One item of a family (e.g. the light of a component)
function OneItem({ ctx, part, path, rings, onRemove }: { ctx: Ctx; part: PartInfo; path: Path; rings: number[] | null; onRemove?: () => void }) {
  const item = getAt(ctx.options, path);
  return <ItemLine ctx={ctx} part={part} path={path} item={isObject(item) ? item : {}} rings={rings} onRemove={onRemove} />;
}

// An item's line: its type (and direction), its numbers, the lines of an
// emission of lines; its other options behind "options", and what removes
// it, as the line is pointed at
function ItemLine({ ctx, part, path, item, rings, onRemove }: { ctx: Ctx; part: PartInfo; path: Path; item: Options; rings: number[] | null; onRemove?: () => void }) {
  const { defs, params, set, project } = ctx;
  const family = part.family!;
  const [more, setMore] = useState(false);
  const member = memberOf(defs, family, item.type);
  const properties = (member?.properties ?? {}) as Record<string, Schema>;
  const lines = family === 'emission' && item.type === 'lines';
  const without = new Set(['type', ...(part.directions ? ['direction'] : []), ...(lines ? ['lines'] : [])]);
  const rest = restOf(defs, member, without);
  const given = Object.keys(item).filter((key) => !without.has(key)).length;
  const directions = part.directions ? choicesOf(properties.direction)?.filter((d) => part.directions!.includes(String(d.value))) : null;
  const direction = String(item.direction ?? properties.direction?.default ?? '');
  return (
    <>
      <div className="gm-trait">
        <TypePick items={typesOf(defs, family, part.types)} value={String(item.type ?? '')} text={String(item.type ?? '')} label={`The type of the ${part.title.toLowerCase()}`} onChange={(type) => set(path, retyped(defs, family, item, type))} />
        {directions && directions.length > 0 && (
          <DropMenu
            label="Its direction"
            width={240}
            entries={directions.map((d) => ({ label: d.title, current: String(d.value) === direction, onClick: () => set(path, { ...item, direction: d.value }) }))}
            target={
              <button className="gm-dir gm-dir-pick" title={directions.find((d) => String(d.value) === direction)?.title}>
                {direction}
              </button>
            }
          />
        )}
        <NumberChips ctx={ctx} owned={ownedBy(params, path)} rings={rings} />
        {lines && <LinesLine emission={item} onChange={(next) => set(path, next)} />}
        {rest && <MoreButton on={more} count={given} onClick={() => setMore(!more)} />}
        {onRemove && <Close label={`Remove it`} onClick={onRemove} />}
      </div>
      {more && rest && defs && (
        <div className="gm-more-form">
          <SchemaForm
            defs={defs}
            node={rest}
            project={project}
            value={omitKeys(item, without)}
            onChange={(next) => set(path, { ...Object.fromEntries(Object.entries(item).filter(([key]) => without.has(key))), ...next }, { moved: inPlace })}
          />
        </div>
      )}
    </>
  );
}

// The height trait paired with a trait (or the one they share): its type,
// or none if it can be, and its numbers
function HeightLine({ ctx, part, path, count, rings, all }: { ctx: Ctx; part: PartInfo; path: Path; count: number; rings: number[] | null; all?: boolean }) {
  const { defs, draft, options, params, project } = ctx;
  const [more, setMore] = useState(false);
  const value = getAt(options, path);
  const item = isObject(value) ? value : null;
  const family = part.family!;
  const set = (next: unknown) =>
    draft.edit((s) => {
      // (a list of a height for each item, of nulls but this one; one
      // alone is the one they share)
      const list = path.slice(0, -1);
      const index = path[path.length - 1] as number;
      const current = getAt(s.options, list);
      const items: unknown[] = Array.isArray(current) ? [...current] : current === undefined ? [] : [current];
      while (items.length < Math.max(count, index + 1)) items.push(null);
      items[index] = next;
      const trimmed = items.every((h) => h === null) ? undefined : items.length === 1 ? items[0] : items;
      return { ...s, options: setAt(s.options, list, trimmed) };
    });
  const member = memberOf(defs, family, item?.type);
  const rest = item ? restOf(defs, member, new Set(['type'])) : null;
  return (
    <>
      <div className="gm-trait gm-height">
        <span className="gm-times">{all ? 'height of all' : 'height'}</span>
        <TypePick
          items={typesOf(defs, family, part.types)}
          value={String(item?.type ?? '')}
          text={item ? String(item.type) : undefined}
          none={part.nullable ? 'the same at every height' : undefined}
          label={part.title}
          onChange={(type) => set(type === '' ? null : retyped(defs, family, item, type))}
        />
        {item && <NumberChips ctx={ctx} owned={ownedBy(params, path)} rings={rings} limit={typeof item.zmax === 'number' ? item.zmax : undefined} />}
        {rest && <MoreButton on={more} count={Object.keys(item ?? {}).filter((key) => key !== 'type').length} onClick={() => setMore(!more)} />}
      </div>
      {more && rest && defs && item && (
        <div className="gm-more-form">
          <SchemaForm defs={defs} node={rest} project={project} value={omit(item, 'type')} onChange={(next) => set({ type: item.type, ...next })} />
        </div>
      )}
    </>
  );
}

// Options of their own (e.g. the clouds of a disk), as a form
function ObjectItem({ ctx, part, path, onRemove }: { ctx: Ctx; part: PartInfo; path: Path; onRemove?: () => void }) {
  const { defs, options, set, project } = ctx;
  const value = getAt(options, path);
  if (!defs || !part.def || !defs[part.def]) return null;
  return (
    <div className="gm-object">
      <SchemaForm defs={defs} def={part.def} project={project} value={isObject(value) ? value : {}} onChange={(next) => set(path, next)} />
      {onRemove && <Close label={`Remove the ${part.title.toLowerCase()}`} onClick={onRemove} />}
    </div>
  );
}

// ---------------------------------------------------------------------
// A part of the mass

function MassRow({
  ctx,
  index,
  item,
  colourOf,
  names,
  open,
  onToggle,
  onRemoved,
}: {
  ctx: Ctx;
  index: number;
  item: Options;
  colourOf: (name: unknown) => string;
  names: string[];
  open: boolean;
  onToggle: () => void;
  onRemoved: () => void;
}) {
  const { defs, params, problems, set, project } = ctx;
  const [more, setMore] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const path: Path = ['mass', index];
  const refs = referencesOf(defs, 'mass_component', item.type);
  const follows = Object.entries(refs).filter(([, what]) => what === 'components').map(([key]) => key);
  const followed = follows.map((key) => item[key]).find((name) => typeof name === 'string');
  const own = problems.filter((p) => startsWith(p.path, path));
  const owned = ownedBy(params, path);
  const without = new Set([...HEAD_OPTIONS, ...follows]);
  const member = memberOf(defs, 'mass_component', item.type);
  const rest = restOf(defs, member, without);
  const typeTitle = titled(defs, 'mass_component', String(item.type)).label;
  // (in a few words: its type, or the types of what it is made of, e.g. the
  // profile of a halo; what it follows; its numbers)
  const inner = Object.entries(item)
    .filter(([key, value]) => key !== 'type' && isObject(value) && typeof value.type === 'string')
    .map(([key, value]) => titled(defs, familyOfOption(defs, 'mass_component', item.type, key) ?? '', String((value as Options).type)).label);
  const words = [inner.length ? inner.join(' · ') : typeTitle, followed ? `follows ${String(followed)}` : null, ...owned.slice(0, 3).map((p) => paramText(ctx, p))].filter(Boolean);
  const typeEntries: MenuEntry[] = typesOf(defs, 'mass_component').map((t) => ({
    label: t.label,
    description: t.description,
    current: t.value === item.type,
    onClick: () => set(path, retyped(defs, 'mass_component', item, t.value)),
  }));
  return (
    <div className="gm-mrow" data-open={open || undefined} data-bad={own.length > 0 || undefined} data-at={`mass/${index}`}>
      <div className="gm-mhead" role="button" tabIndex={0} aria-expanded={open} onClick={() => !renaming && onToggle()} onKeyDown={(event) => event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ') && (event.preventDefault(), onToggle())}>
        <span className="gm-dot" style={{ background: followed ? colourOf(followed) : 'var(--faint)' }} />
        {renaming ? (
          <span onClick={(event) => event.stopPropagation()}>
            <NameText value={String(item.name ?? '')} required onCommit={(name) => (setRenaming(false), name && set([...path, 'name'], name, { moved: inPlace }))} onCancel={() => setRenaming(false)} autoFocus />
          </span>
        ) : (
          <b>{String(item.name ?? '')}</b>
        )}
        <span className="gm-msum">{words.join(' · ')}</span>
        <span className="gm-cmenu" onClick={(event) => event.stopPropagation()}>
          <DropMenu
            label={`The mass ${String(item.name ?? '')}`}
            width={240}
            at="end"
            entries={[
              { label: 'Rename', onClick: () => setRenaming(true) },
              { label: 'Change its type', entries: typeEntries },
              {
                label: 'Remove',
                onClick: () => {
                  set(path, undefined, { moved: removedFrom(['mass'], index) });
                  onRemoved();
                },
              },
            ]}
            target={
              <button className="gm-more" aria-label="Its actions">
                ⋯
              </button>
            }
          />
        </span>
      </div>
      {open && (
        <div className="gm-mbody">
          <div className="gm-trait">
            {follows.map((key) => (
              <span key={key} className="gm-follow">
                follows
                <TypePick items={names.map((name) => ({ value: name, label: name }))} value={String(item[key] ?? '')} text={String(item[key] ?? '')} label="The component it follows" onChange={(name) => set([...path, key], name)} />
              </span>
            ))}
            {inner.length > 0 &&
              Object.entries(item)
                .filter(([key, value]) => key !== 'type' && isObject(value) && typeof value.type === 'string')
                .map(([key, value]) => {
                  const family = familyOfOption(defs, 'mass_component', item.type, key);
                  return family ? (
                    <TypePick
                      key={key}
                      items={typesOf(defs, family)}
                      value={String((value as Options).type)}
                      text={String((value as Options).type)}
                      label={`Its ${key}`}
                      onChange={(type) => set([...path, key], retyped(defs, family, value, type))}
                    />
                  ) : null;
                })}
            <NumberChips ctx={ctx} owned={owned} rings={null} />
            {rest && <MoreButton on={more} count={Object.keys(item).filter((key) => !without.has(key)).length} onClick={() => setMore(!more)} />}
          </div>
          {more && rest && defs && (
            <div className="gm-more-form">
              <SchemaForm
                defs={defs}
                node={rest}
                project={project}
                value={omitKeys(item, without)}
                onChange={(next) => set(path, { ...Object.fromEntries(Object.entries(item).filter(([key]) => without.has(key))), ...next }, { moved: inPlace })}
              />
            </div>
          )}
          {own.map((p, i) => (
            <p key={i} className="gm-why">
              {problemText(p.message)}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------
// Small parts

// A type of a family to choose (e.g. of a velocity trait): its title (or
// the text given, e.g. its type), a menu of the types, each with what it
// is; none, if it can be
function TypePick({ items, value, onChange, label, none, text }: { items: TypeChoice[]; value: string; onChange: (type: string) => void; label: string; none?: string; text?: string }) {
  const current = items.find((t) => t.value === value);
  const entries: MenuEntry[] = [
    ...(none ? [{ label: none, current: value === '', onClick: () => onChange('') }] : []),
    ...items.map((t) => ({ label: t.label, description: t.description, current: t.value === value, onClick: () => onChange(t.value) })),
  ];
  return (
    <DropMenu
      label={label}
      width={300}
      entries={entries}
      target={
        <button className="gm-ty" title={current?.label}>
          {text ?? current?.label ?? (value === '' ? (none ?? 'Choose…') : value)}
        </button>
      }
    />
  );
}

// What opens the other options of an item (how many are given)
function MoreButton({ on, count = 0, onClick }: { on: boolean; count?: number; onClick: () => void }) {
  return (
    <button className="gm-opt" data-on={on || undefined} onClick={onClick} aria-expanded={on} title="Its other options">
      {count > 0 ? `options · ${count}` : 'options'}
    </button>
  );
}

// A name, as typed: set as it is entered or left (Escape: as it was)
function NameText({ value, onCommit, required, onCancel, autoFocus }: { value: string; onCommit: (name: string) => void; required: boolean; onCancel?: () => void; autoFocus?: boolean }) {
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    const name = (text ?? value).trim();
    if (name !== value && (name || !required) && /^[A-Za-z_][A-Za-z0-9_]*$|^$/.test(name)) onCommit(name);
    else onCancel?.();
    setText(null);
  };
  return (
    <input
      className="gm-name"
      value={text ?? value}
      placeholder={required ? 'name' : 'no name'}
      autoFocus={autoFocus}
      onChange={(event) => setText(event.currentTarget.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') {
          setText(null);
          onCancel?.();
        }
      }}
      aria-label="Its name"
      spellCheck={false}
    />
  );
}

// A number of an option (not a parameter), as typed: set once it reads as
// one, at least 0
function NumberText({ value, onCommit, label, unit, bare }: { value: number; onCommit: (value: number) => void; label: string; unit?: string; bare?: boolean }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <label className={bare ? 'gm-numtext gm-numtext-bare' : 'gm-numtext'}>
      <input
        value={text ?? (Number.isFinite(value) ? String(value) : '')}
        onChange={(event) => {
          const next = event.currentTarget.value;
          setText(next);
          const number = Number(next);
          if (next.trim() !== '' && Number.isFinite(number) && number >= 0) onCommit(number);
        }}
        onBlur={() => setText(null)}
        aria-label={label}
        inputMode="decimal"
      />
      {unit && <span>{unit}</span>}
    </label>
  );
}

type RingsForm = 'sep' | 'len' | 'list';

// The radii of the rings: {max, sep}, {max, len} or a list of them; a
// change of form keeps the reach of the rings and about their number
function RingsLine({ value, onChange, unit }: { value: unknown; onChange: (value: unknown) => void; unit: string }) {
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
  const forms: { value: RingsForm; label: string }[] = [
    { value: 'sep', label: 'every' },
    { value: 'len', label: 'count' },
    { value: 'list', label: 'radii' },
  ];
  return (
    <div className="gm-rings">
      <div className="gm-trait">
        <DropMenu
          label={`How the ${unit} are given`}
          width={220}
          entries={[
            { label: 'Evenly, every so far', current: form === 'sep', onClick: () => toForm('sep') },
            { label: 'Evenly, so many', current: form === 'len', onClick: () => toForm('len') },
            { label: 'At the radii given', current: form === 'list', onClick: () => toForm('list') },
          ]}
          target={<button className="gm-ty">{forms.find((f) => f.value === form)?.label}</button>}
        />
        {form === 'list' ? (
          <span className="gm-note">0 to {Number(max.toPrecision(4))}″</span>
        ) : (
          <>
            <span className="gm-word">to</span>
            <NumberText value={Number(options.max)} onCommit={(m) => m > 0 && onChange({ ...options, max: m })} label="The largest radius" unit="″" />
            {form === 'sep' ? (
              <>
                <span className="gm-word">every</span>
                <NumberText value={Number(options.sep)} onCommit={(sep) => sep > 0 && onChange({ ...options, sep })} label="The separation" unit="″" />
              </>
            ) : (
              <>
                <span className="gm-word">in</span>
                <NumberText value={Number(options.len)} onCommit={(len) => onChange({ ...options, len: Math.max(2, Math.round(len)) })} label="The number" />
              </>
            )}
          </>
        )}
        <span className="gm-note">
          {radii.length} {unit}
        </span>
      </div>
      {form === 'list' && <RadiiText radii={radii} onCommit={onChange} />}
      {radii.length > 0 && (
        <div className="gm-ring-line" title={radii.map((r) => `${r}″`).join(', ')}>
          {radii.map((r, i) => (
            <i key={i} style={{ left: `${(r / (max || 1)) * 100}%` }} />
          ))}
        </div>
      )}
    </div>
  );
}

// Radii given as a list, as typed (by commas or spaces): set once each
// reads as a number
function RadiiText({ radii, onCommit }: { radii: number[]; onCommit: (radii: number[]) => void }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <textarea
      className="gm-radii"
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

// The lines of an emission of lines, in its line: by default one line at
// the velocity of the spectral axis, whatever it is; else lines of the
// catalogue (or of the configuration, kept as they are), the first the one
// whose flux the density gives, the others of a ratio of flux to it
function LinesLine({ emission, onChange }: { emission: Options; onChange: (emission: Options) => void }) {
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
    <>
      {lines.length === 0 ? (
        <span className="gm-note">one line, at the velocity of the data</span>
      ) : (
        lines.map((line, i) => (
          <span key={i} className="gm-line" title={`${rests.get(lineName(line)) ?? 'A line of the configuration'}${i === 0 ? ' · its flux from the density' : ' · a ratio of flux to the first'}`}>
            {lineName(line)}
            <button className="gm-x" aria-label={`Remove the line ${lineName(line)}`} onClick={() => set(lines.filter((_, j) => j !== i))}>
              ×
            </button>
          </span>
        ))
      )}
      <DropMenu
        label="Add a line"
        width={260}
        entries={catalogue.filter((l) => !named.includes(l.name)).map((l) => ({ label: l.name, description: l.rest, onClick: () => set([...lines, l.name]) }))}
        target={<button className="gm-add gm-add-small gm-hover">+ line</button>}
      />
    </>
  );
}

// A problem as people read it: without the name of the error of Python
// (e.g. 'ValueError: ')
function problemText(message: string): string {
  return message.replace(/^[A-Za-z]+Error: /, '');
}

function omit(options: Options, key: string): Options {
  const { [key]: _, ...rest } = options;
  return rest;
}

function omitKeys(options: Options, keys: Set<string>): Options {
  return Object.fromEntries(Object.entries(options).filter(([key]) => !keys.has(key)));
}

// The rings (or shells) of a component in a few words: how many, to what
// radius, how far that is, and whether they tilt or wander; a click opens
// how they are given
function RingsRow({ ctx, component, value, onChange }: { ctx: Ctx; component: Options; value: unknown; onChange: (value: unknown) => void }) {
  const unit = component.type === 'cone' ? 'shells' : 'rings';
  const radii = ringRadii(value);
  const max = radii.length ? radii[radii.length - 1] : 0;
  const notes = [
    ctx.kpcPerArcsec === null || !max ? null : `${significant(max * ctx.kpcPerArcsec, 3)} kpc`,
    component.tilted === true ? 'tilted' : null,
    component.loose === true ? 'loose' : null,
  ].filter(Boolean);
  return (
    <div className="gm-trait">
      <Popover position="bottom-start" offset={4} width={340} shadow="md" withinPortal trapFocus={false} classNames={{ dropdown: 'gm-pop' }}>
        <Popover.Target>
          <button className="gm-num gm-text" title={`How its ${unit} are given`}>
            {radii.length} {unit} · 0–{Number(max.toPrecision(4))}″
          </button>
        </Popover.Target>
        <Popover.Dropdown>
          <RingsLine value={value} onChange={onChange} unit={unit} />
        </Popover.Dropdown>
      </Popover>
      {notes.length > 0 && <span className="gm-note">{notes.join(' · ')}</span>}
    </div>
  );
}

// The family of an option of a type of a family (e.g. 'halo_profile', of
// the profile of a halo), if its value is a member of one
function familyOfOption(defs: Defs | undefined, family: string, type: unknown, option: string): string | null {
  const property = (memberOf(defs, family, type)?.properties as Record<string, Schema> | undefined)?.[option];
  if (!property) return null;
  const branches = (property.anyOf as Schema[] | undefined) ?? [property];
  const ref = branches.map((b) => b.$ref).find((r): r is string => typeof r === 'string');
  return ref ? ref.replace('#/$defs/', '') : null;
}
