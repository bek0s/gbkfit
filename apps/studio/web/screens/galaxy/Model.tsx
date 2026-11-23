// The model of a galaxy, as its sheet shows it: the galaxy as a whole (its
// distance and redshift, on one line), the parts of its mass (under its
// gravity, a row each, in reach however long the components are), and its
// components, a row each (its name, its type, its extent and its parts in
// a few words). Rows open, as many as wanted, and stay open as the page
// is reloaded. An open component is in sections in the order of its
// configuration (its geometry, its traits, its light, its physics), each
// part a group under its title and its keys in the configuration, with
// what adds to it, and each single setting a line of its own. Every line
// of the card is what it is, then its value, in one column for the whole
// card; a parameter is a line of its own, under what owns it. A group
// folds to one line: what it has, in a few words; which are open is
// remembered. A problem shows where it is, and the count of them at the
// foot goes to each in turn.
import { Popover } from '@mantine/core';
import { Fragment, useCallback, useRef, useState, type ReactNode } from 'react';
import { useComponentTypes, useLines } from '../../api/queries';
import { choicesOf, SchemaForm, type Defs, type Schema } from '../../components/SchemaForm';
import { significant } from '../../components/format';
import { Close } from '../../components/ui/basics';
import { Expand } from '../../components/ui/motion';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import { useRemembered } from '../../shell/remember';
import type { EditHow, GalaxyDraft, Problem } from './draft';
import { DirectionMark, RingMark } from './Marks';
import { freeCount, NumberChip, NumberRows, paramText, type NumbersContext } from './Numbers';
import { asList, asOneOrList, componentName, componentsOf, freshName, getAt, isObject, itemText, lineList, lineName, ringRadii, roundStep, setAt, startsWith, type Options, type Path } from './options';
import { ownedBy, settingOf, type Param } from './params';
import { memberOf, propertiesOf, restOf, retyped, startOf, titled, typesOf, type TypeChoice } from './schema';
import { extentOf, familyOfOption, HEAD_OPTIONS, nameBase, optionTitle, partsOfType, referencesOf, settingsOfType, withReferencesRenamed, type ComponentTypes, type PartInfo } from './types';

// A row open in the sheet: a component or a part of the mass
export type OpenItem = { kind: 'component' | 'mass'; index: number };

// The rows open, as one is opened (last), closed, or removed (those after
// it, of its kind, a place back) or copied (those after it a place on)
const isOpenIn = (open: OpenItem[], kind: OpenItem['kind'], index: number) => open.some((o) => o.kind === kind && o.index === index);
const opened = (open: OpenItem[], kind: OpenItem['kind'], index: number): OpenItem[] => [...open.filter((o) => !(o.kind === kind && o.index === index)), { kind, index }];
const closed = (open: OpenItem[], kind: OpenItem['kind'], index: number): OpenItem[] => open.filter((o) => !(o.kind === kind && o.index === index));
const shifted = (open: OpenItem[], kind: OpenItem['kind'], from: number, by: 1 | -1): OpenItem[] => open.map((o) => (o.kind === kind && o.index >= from ? { kind, index: o.index + by } : o));
const removedRow = (open: OpenItem[], kind: OpenItem['kind'], index: number) => shifted(closed(open, kind, index), kind, index + 1, -1);

interface Ctx extends NumbersContext {
  defs: Defs | undefined;
  types: ComponentTypes | undefined;
  project: string;
  problems: Problem[];
  set: (path: Path, value: unknown, how?: EditHow) => void;
  // The length of an arcsec at the distance of the galaxy (kpc), if known
  kpcPerArcsec: number | null;
  folds: Folds;
  // What keeps a row where it is in its pane as what is around it opens or
  // closes (see useKeptInPlace)
  keep: (row: HTMLElement) => void;
}

// The groups of the rows that are open, by the name of their owner (a
// component, or a part of the mass: see massFolds) and the key of the
// group (e.g. 'vptraits'): all folded at first, and kept as the rows close
// and open, and as the page is reloaded (in this browser, for the galaxy)
interface Folds {
  isOpen: (owner: string, group: string) => boolean;
  set: (owner: string, groups: string[], open: boolean) => void;
  rename: (from: string, to: string) => void;
}

// The owner of the groups of a part of the mass, among the folds (apart
// from those of a component of the same name)
const massFolds = (name: string) => `mass:${name}`;

function useFolds(project: string, galaxy: string): Folds {
  const [open, setOpen] = useRemembered<Record<string, string[]>>(`galaxy-folds:${project}/${galaxy}`, {});
  return {
    isOpen: (component, group) => open[component]?.includes(group) ?? false,
    set: (component, groups, on) =>
      setOpen((all) => {
        const others = (all[component] ?? []).filter((group) => !groups.includes(group));
        return { ...all, [component]: on ? [...others, ...groups] : others };
      }),
    rename: (from, to) => setOpen(({ [from]: moved, ...rest }) => (moved ? { ...rest, [to]: moved } : rest)),
  };
}

// A group of a card that folds (see Folds): whether it is open; what opens
// or folds it, kept where it is (all: every group of its card, as it goes);
// what opens it (e.g. as something is added to it)
interface Fold {
  open: boolean;
  toggle: (row: HTMLElement, all: boolean) => void;
  show: () => void;
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
  // The name of the galaxy (what the sheet remembers is its own)
  galaxy: string;
  field: number;
  // The colours of the components and of the parts of the mass (tokens)
  colours: string[];
  massColours: string[];
  open: OpenItem[];
  onOpen: (open: OpenItem[]) => void;
  isolated: string | null;
  onIsolate: (name: string | null) => void;
}

export function Model({ draft, project, galaxy, field, colours, massColours, open, onOpen, isolated, onIsolate }: ModelProps) {
  const types = useComponentTypes().data as ComponentTypes | undefined;
  const options = draft.state?.options ?? {};
  const components = componentsOf(options);
  const mass = asList(options.mass);
  const set = (path: Path, value: unknown, how?: EditHow) => draft.edit((s) => ({ ...s, options: setAt(s.options, path, value) }), how);
  const folds = useFolds(project, galaxy);
  const { room, keep } = useKeptInPlace();
  const ctx: Ctx = { draft, options, params: draft.params, field, defs: draft.defs, types, project, problems: draft.problems, set, kpcPerArcsec: kpcPerArcsecOf(draft), folds, keep };
  // (a row opened or closed, kept where it is under the pointer)
  const toggle = (kind: OpenItem['kind'], index: number, row: HTMLElement) => {
    keep(row);
    onOpen(isOpenIn(open, kind, index) ? closed(open, kind, index) : opened(open, kind, index));
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
    onOpen(opened(open, 'component', components.length));
  };

  const addMass = (type: string) => {
    const refs = referencesOf(draft.defs, 'mass_component', type);
    // (one that follows a component: the first it can follow; else the
    // first)
    const follow = followable(components, type)[0] ?? components[0];
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
    onOpen(opened(open, 'mass', mass.length));
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
    onOpen(opened(open, section === 'mass' ? 'mass' : 'component', index));
    // (once it is open)
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        const row = document.querySelector(`[data-at="${section}/${index}"]`);
        (row?.querySelector('.gm-why') ?? row)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }),
    );
  };
  const problemCount = `${draft.problems.length} ${draft.problems.length === 1 ? 'problem' : 'problems'}`;

  return (
    <div className="gm">
      <GalaxyLine ctx={ctx} />

      <div className="gm-sec">
        <span className="gm-over">
          Mass<b>{mass.length}</b>
        </span>
        <Gravity ctx={ctx} hasMass={mass.length > 0} />
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
        <div className="gm-rows">
          {mass.map((item, i) => (
            <MassRow
              key={i}
              ctx={ctx}
              index={i}
              item={item}
              colour={massColours[i]}
              components={components}
              open={isOpenIn(open, 'mass', i)}
              onToggle={(row) => toggle('mass', i, row)}
              onRemoved={() => onOpen(removedRow(open, 'mass', i))}
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
      {components.length === 0 ? (
        <p className="gm-empty">No components: nothing emits light.</p>
      ) : (
        <div className="gm-rows">
          {components.map((component, i) => (
            <ComponentCard
              key={i}
              ctx={ctx}
              index={i}
              component={component}
              colour={colours[i]}
              open={isOpenIn(open, 'component', i)}
              onToggle={(row) => toggle('component', i, row)}
              isolated={isolated === componentName(component, i)}
              onIsolate={(on) => onIsolate(on ? componentName(component, i) : null)}
              onDuplicated={() => onOpen(opened(shifted(open, 'component', i + 1, 1), 'component', i + 1))}
              onRemoved={() => onOpen(removedRow(open, 'component', i))}
            />
          ))}
        </div>
      )}

      {(draft.problems.length > 0 || draft.describeError) && (
        <div className="gm-problems">
          {draft.problems.length > 0 && (
            <span>
              {placed.length > 0 ? (
                <button className="gm-link" onClick={goToProblem} title="Open where the next one is">
                  {problemCount}
                </button>
              ) : (
                problemCount
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
      )}
      <div ref={room} aria-hidden />
    </div>
  );
}

// The row of a card that is opened or closed kept where it is in its pane,
// under the pointer (else a card above it that closes moves it up, and a
// pane that gets shorter scrolls back): while the cards move, the pane
// follows the row, with room under the cards if it needs it, which goes
// as the pane is scrolled back up
function useKeptInPlace() {
  const room = useRef<HTMLDivElement>(null);
  const shrinking = useRef<(() => void) | null>(null);
  const keep = useCallback((row: HTMLElement) => {
    const pane = scrollParent(row);
    const spacer = room.current;
    if (!pane || !spacer) return;
    const top = row.getBoundingClientRect().top;
    // (longer than the cards take to open and close; see ui/motion.tsx)
    const end = performance.now() + 400;
    // (the room the pane needs under the cards to be scrolled so far: what
    // they lack of it; none at its top)
    const roomFor = (scroll: number) => (scroll > 0 ? Math.max(0, scroll + pane.clientHeight - (pane.scrollHeight - spacer.offsetHeight)) : 0);
    const follow = () => {
      const scroll = pane.scrollTop + row.getBoundingClientRect().top - top;
      spacer.style.height = `${roomFor(scroll)}px`;
      pane.scrollTop = scroll;
      if (performance.now() < end) requestAnimationFrame(follow);
      else pane.style.overflowAnchor = '';
    };
    // (the browser's own anchoring of the scroll would fight it)
    pane.style.overflowAnchor = 'none';
    requestAnimationFrame(follow);
    if (shrinking.current) pane.removeEventListener('scroll', shrinking.current);
    const shrink = () => {
      if (performance.now() < end) return;
      const next = Math.min(spacer.offsetHeight, roomFor(pane.scrollTop));
      spacer.style.height = `${next}px`;
      if (next === 0) pane.removeEventListener('scroll', shrink);
    };
    shrinking.current = shrink;
    pane.addEventListener('scroll', shrink);
  }, []);
  return { room, keep };
}

// The nearest ancestor of an element that scrolls
function scrollParent(element: HTMLElement): HTMLElement | null {
  for (let at = element.parentElement; at; at = at.parentElement) {
    const { overflowY } = getComputedStyle(at);
    if (overflowY === 'auto' || overflowY === 'scroll') return at;
  }
  return null;
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

// ---------------------------------------------------------------------
// The galaxy as a whole

// The menu of the types of a family (e.g. of the distance), the current
// marked; a first choice of none, if given
function typeEntries(defs: Defs | undefined, family: string, current: string, onChange: (type: string) => void, none?: string): MenuEntry[] {
  return [
    ...(none ? [{ label: none, current: current === '', onClick: () => onChange('') }] : []),
    ...typesOf(defs, family).map((t) => ({ label: t.label, description: t.description, current: t.value === current, onClick: () => t.value !== current && onChange(t.value) })),
  ];
}

// The galaxy as a whole, on one line: its distance (its kind a menu, its
// number a chip that opens to set, and the length of an arcsec there) and
// its redshift, typed in place
function GalaxyLine({ ctx }: { ctx: Ctx }) {
  const { defs, options, params, set } = ctx;
  const distance = isObject(options.distance) ? options.distance : null;
  const distanceParam = params.find((p) => p.owner.length === 1 && p.owner[0] === 'distance');
  const distanceType = String(distance?.type ?? '');
  const z = typeof options.redshift === 'number' ? options.redshift : 0;
  const scale = ctx.kpcPerArcsec === null ? null : `1″ = ${significant(ctx.kpcPerArcsec * 1000, 3)} pc`;
  let value: ReactNode = <span className="gm-gnote">none: angles only</span>;
  if (distance && distanceParam) value = <NumberChip ctx={ctx} param={distanceParam} rings={null} bare />;
  else if (distance) value = <span className="gm-gv">{titled(defs, 'distance', distanceType).label.toLowerCase()}</span>;
  return (
    <div className="gm-galaxy">
      <span className="gm-gitem">
        <DropMenu
          label="The distance of the galaxy"
          width={300}
          entries={typeEntries(defs, 'distance', distanceType, (type) => set(['distance'], type === '' ? undefined : retyped(defs, 'distance', distance, type)), 'None')}
          target={
            <button className="gm-glabel" title="The distance of the galaxy: click to change its kind">
              Distance
            </button>
          }
        />
        {value}
        {scale && <span className="gm-gnote">{scale}</span>}
      </span>
      <span className="gm-gitem" title="The redshift of the galaxy: of its lines, and the dimming of its light">
        <span className="gm-glabel-text">Redshift</span>
        <NumberText value={z} label="The redshift of the galaxy" onCommit={(value) => set(['redshift'], value > 0 ? value : undefined)} bare />
      </span>
    </div>
  );
}

// The gravity of the galaxy, in the head of its mass (how its mass moves
// what it holds): its kind, a menu; its numbers, chips; its other options
// behind "options"
function Gravity({ ctx, hasMass }: { ctx: Ctx; hasMass: boolean }) {
  const { defs, params, options, set, project } = ctx;
  const gravity = isObject(options.gravity) ? options.gravity : null;
  const gravityType = String(gravity?.type ?? 'newton');
  const gravityParams = ownedBy(params, ['gravity']);
  const rest = restOf(defs, memberOf(defs, 'gravity', gravity?.type), new Set(['type']));
  return (
    <span className="gm-gravity" data-idle={!hasMass || undefined}>
      <DropMenu
        label="The gravity of the galaxy"
        width={300}
        entries={typeEntries(defs, 'gravity', gravityType, (type) => set(['gravity'], type === 'newton' ? undefined : retyped(defs, 'gravity', gravity, type)))}
        target={
          <button className="gm-ty" title={hasMass ? 'The gravity of the mass' : 'The gravity of the mass (it has none yet)'}>
            {titled(defs, 'gravity', gravityType).label}
          </button>
        }
      />
      {gravityParams.map((p) => (
        <NumberChip key={p.name} ctx={ctx} param={p} rings={null} />
      ))}
      {rest && defs && gravity && (
        <Popover position="bottom-start" offset={4} width={340} shadow="md" withinPortal trapFocus={false} classNames={{ dropdown: 'gm-pop' }}>
          <Popover.Target>
            <button className="gm-opt" title="Its other options">
              options
            </button>
          </Popover.Target>
          <Popover.Dropdown>
            <div className="gm-popbody">
              <SchemaForm defs={defs} node={rest} project={project} value={omit(gravity, 'type')} onChange={(next) => set(['gravity'], { type: gravity.type, ...next })} />
            </div>
          </Popover.Dropdown>
        </Popover>
      )}
    </span>
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
  onDuplicated,
  onRemoved,
}: {
  ctx: Ctx;
  index: number;
  component: Options;
  colour: string;
  open: boolean;
  onToggle: (row: HTMLElement) => void;
  isolated: boolean;
  onIsolate: (on: boolean) => void;
  // (what the sheet does as it is copied (the copy after it, open) or
  // removed)
  onDuplicated: () => void;
  onRemoved: () => void;
}) {
  const { defs, types, problems, draft, options, params, set } = ctx;
  const [renaming, setRenaming] = useState(false);
  const base: Path = ['components', index];
  const components = componentsOf(options);
  const parts = partsOfType(defs, types, component.type);
  const radii = ringRadii(component.rnodes);
  const own = problems.filter((p) => startsWith(p.path, base));
  const typeTitle = titled(defs, 'component', String(component.type)).label;

  // (what refers to it by name, e.g. the mass that follows it, with it)
  const rename = (name: string) => {
    setRenaming(false);
    if (!name || components.some((c, i) => i !== index && c.name === name)) return;
    const old = component.name;
    // (its open groups with it)
    ctx.folds.rename(componentName(component, index), name);
    draft.edit(
      (s) => {
        const named = setAt(s.options, [...base, 'name'], name);
        return { ...s, options: typeof old === 'string' ? withReferencesRenamed(defs, named, 'components', old, name) : named };
      },
      { moved: inPlace },
    );
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
    onDuplicated();
  };
  const remove = () => {
    if (isolated) onIsolate(false);
    set(base, undefined, { moved: removedFrom(['components'], index) });
    onRemoved();
  };
  return (
    <div className="gm-row gm-comp" data-open={open || undefined} data-bad={own.length > 0 || undefined} data-at={`components/${index}`}>
      <div
        className="gm-crow"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={(event) => !renaming && onToggle(event.currentTarget)}
        onKeyDown={(event) => {
          if (renaming || event.target !== event.currentTarget) return;
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onToggle(event.currentTarget);
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
        {/* (open, its rings say it) */}
        <span className="gm-cext">{open ? '' : extentOf(component, radii)}</span>
        <span className="gm-cmenu" onClick={(event) => event.stopPropagation()}>
          <DropMenu
            label={`The component ${componentName(component, index)}`}
            width={240}
            at="end"
            // (its rename gives the focus to the field of its name)
            returnFocus={false}
            entries={[
              { label: 'Rename', onClick: () => setRenaming(true) },
              { label: isolated ? 'Show all in the views' : 'Show it alone in the views', onClick: () => onIsolate(!isolated) },
              { label: 'Duplicate', onClick: duplicate },
              { label: 'Remove', onClick: remove },
            ]}
            target={
              <button className="gm-more" aria-label="Its actions" title="Rename, duplicate, remove">
                ⋯
              </button>
            }
          />
        </span>
        {!open && <span className="gm-csum">{glanceOf(ctx, component, parts)}</span>}
      </div>
      <Expand open={open}>
        <ComponentBody ctx={ctx} index={index} component={component} parts={parts} rings={radii.length ? radii : null} />
      </Expand>
    </div>
  );
}

// A component in a few words: its parts, each its title and its items'
// types (a height after its trait, a direction after its speed), and the
// lines it emits
function glanceOf(ctx: Ctx, component: Options, parts: PartInfo[]): ReactNode {
  const { defs } = ctx;
  const words: ReactNode[] = [];
  for (const part of parts) {
    const value = component[part.option];
    if (value === undefined || value === null || part.family === null) continue;
    const items = asList(value);
    if (items.length === 0) continue;
    const heights = part.heights ? (Array.isArray(component[part.heights.option]) ? (component[part.heights.option] as unknown[]) : [component[part.heights.option]]) : [];
    const text = items.map((item, i) => {
      const height = heights.length === 1 ? (i === items.length - 1 ? heights[0] : undefined) : heights[i];
      return (
        <span key={i}>
          {i > 0 && ' + '}
          {titled(defs, part.family!, String(item.type)).label.toLowerCase()}
          {typeof item.direction === 'string' && (
            <span className="gm-dir">
              <DirectionMark direction={item.direction} />
              {item.direction}
            </span>
          )}
          {isObject(height) && ` × ${titled(defs, part.heights!.family!, String(height.type)).label.toLowerCase()}`}
          {part.family === 'emission' && item.type === 'lines' && lineList(item.lines).length > 0 && ` ${lineList(item.lines).map(lineName).join(', ')}`}
        </span>
      );
    });
    words.push(
      <span key={part.option} className="gm-glance">
        {/* (the light says what it is: e.g. "emission lines") */}
        {part.family !== 'emission' && <span className="gm-pw">{part.title.toLowerCase()}</span>}
        {text}
      </span>,
    );
  }
  return words;
}

// ---------------------------------------------------------------------
// The open card of a component: sections in the order of its
// configuration (its geometry, its traits, its light, its physics), each
// of groups under a subtitle (a part, with its keys in the configuration
// and what adds to it, its lines under it) and lines of single settings

type Section = 'geometry' | 'traits' | 'light' | 'physics';
const SECTIONS: { key: Section; title: string }[] = [
  { key: 'geometry', title: 'Geometry' },
  { key: 'traits', title: 'Traits' },
  { key: 'light', title: 'Light' },
  { key: 'physics', title: 'Physics' },
];

// The options of the rings of a component (or its shells, or the nodes of
// the traits of a spheroid), shown with them: their width and
// interpolation, the sides of a cone, and a centre and an orientation per
// ring (the switches of the lines of its frame, if it has them)
const RING_OPTIONS = ['tilted', 'loose', 'sides', 'interp', 'rstep'];
// The extent of a spheroid, at the top of its geometry
const EXTENT_OPTIONS = ['rmax'];
// The options of the motion of a spheroid (of its orbits), with the
// parameters of its motion
const MOTION_OPTIONS = ['norbits', 'turns', 'seed'];
// The parts of the light of a component, and of its physics (single
// settings of how it holds itself up and what it is made of)
const LIGHT_PARTS = ['emission', 'opacity'];
const PHYSICS_PARTS = ['pressure_support', 'clouds', 'rhtraits', 'equilibrium'];
// The parameters of a component that place it (see its frame): its
// centre, systemic velocity, orientation and the wall of a cone
const FRAME_PARAMS = ['xpos', 'ypos', 'vsys', 'posa', 'incl', 'wall'];
// The lines of the frame, each with the option that gives it a value per
// ring: its centre (and systemic velocity), and its orientation
const FRAME_LINES = [
  { title: 'Centre', keys: ['xpos', 'ypos', 'vsys'], option: 'loose' },
  { title: 'Orientation', keys: ['posa', 'incl'], option: 'tilted' },
];

// Whether a parameter is of a key of its component (e.g. 'disk_posa', of
// 'posa')
function isOfKey(param: Param, key: string): boolean {
  return param.name === key || param.name.endsWith(`_${key}`);
}

// The section of a part: its lists of traits in the traits, its light in
// the light, its pressure support, clouds and vertical equilibrium (the
// heights of a thin disk, the dispersion of a thick one) in the physics;
// the others (its rings, the profile and shape
// of a spheroid) in the geometry
function sectionOf(part: PartInfo): Section {
  if (LIGHT_PARTS.includes(part.option)) return 'light';
  if (PHYSICS_PARTS.includes(part.option)) return 'physics';
  if (part.family !== null && part.many) return 'traits';
  return 'geometry';
}

// The title of a part under its section: its own, unless it is that of
// the section (e.g. the light of a component, 'Emission' in Light)
function subtitleOf(part: PartInfo, section: Section): string {
  const title = SECTIONS.find((s) => s.key === section)!.title;
  return part.title.toLowerCase() === title.toLowerCase() ? part.option.charAt(0).toUpperCase() + part.option.slice(1) : part.title;
}

// The keys of a part in the configuration, as its subtitle shows them;
// none if they only say its title again (e.g. 'opacity', or 'equilibrium'
// of 'Equilibrium dispersion')
function keysOf(title: string, keys: string[]): string | null {
  const plain = (text: string) => ` ${text.toLowerCase().replace(/_/g, ' ')} `;
  return keys.length === 1 && plain(title).includes(plain(keys[0])) ? null : keys.join(' · ');
}

// A group of the card: its subtitle (its keys and what adds to it), its
// lines under it, and its problems. Folded, its subtitle and beside it what
// it has in a few words (its summary) and how many of its numbers a fit
// frees; its problems still under it.
function Group({
  title,
  keys,
  description,
  add,
  problems = [],
  fold,
  summary,
  free = 0,
  children,
}: {
  title: string;
  keys?: string | null;
  description?: string;
  add?: ReactNode;
  problems?: Problem[];
  fold: Fold;
  summary: ReactNode;
  free?: number;
  children: ReactNode;
}) {
  return (
    <div className="gm-group" data-bad={problems.length > 0 || undefined} data-folded={!fold.open || undefined}>
      <div className="gm-sub">
        <button className="gm-fold" aria-expanded={fold.open} title={`${fold.open ? 'Fold' : 'Open'} it (Alt: every group)`} onClick={(event) => fold.toggle(event.currentTarget, event.altKey)}>
          <svg viewBox="0 0 10 10" aria-hidden="true">
            <path d="M3.5 1.8 6.8 5 3.5 8.2" />
          </svg>
          <span className="gm-sub-title" title={description}>
            {title}
          </span>
        </button>
        {fold.open ? (
          <>
            {keys && <code className="gm-key">{keys}</code>}
            {add && <span className="gm-sub-add">{add}</span>}
          </>
        ) : (
          <span className="gm-sum">
            <span className="gm-sum-text">{summary}</span>
            {free > 0 && <span className="gm-free">{free} free</span>}
          </span>
        )}
      </div>
      <Expand open={fold.open}>
        <div className="gm-glines">{children}</div>
      </Expand>
      <Problems list={problems} />
    </div>
  );
}

// The problems of a part or a group, under it
function Problems({ list }: { list: Problem[] }) {
  return (
    <>
      {list.map((p, i) => (
        <p key={i} className="gm-why">
          {problemText(p.message)}
        </p>
      ))}
    </>
  );
}

function ComponentBody({ ctx, index, component, parts, rings }: { ctx: Ctx; index: number; component: Options; parts: PartInfo[]; rings: number[] | null }) {
  const { defs, types, params, problems } = ctx;
  const base: Path = ['components', index];
  const settings = settingsOfType(defs, types, component.type);
  const has = (part: PartInfo) => {
    const value = component[part.option];
    return value !== undefined && value !== null && !(Array.isArray(value) && value.length === 0);
  };
  const problemsOf = (options: string[]) => problems.filter((p) => options.some((option) => startsWith(p.path, [...base, option])));
  // (its own parameters: those that place it, its frame; of a spheroid
  // that moves by the Jeans equations or orbits, the others those of its
  // motion; of others, those of its light, e.g. the flux of a point)
  const own = ownedBy(params, base, true);
  const placing = (p: Param) => FRAME_PARAMS.some((key) => isOfKey(p, key));
  const moves = String(component.type).endsWith('_spheroid');
  const frame = own.filter(placing);
  const motion = moves ? own.filter((p) => !placing(p)) : [];
  const shining = moves ? [] : own.filter((p) => !placing(p));
  // (its settings, each where it acts; the others with its frame)
  const pick = (keys: string[]) => keys.filter((key) => settings.includes(key));
  const ringOptions = pick(RING_OPTIONS);
  const extent = pick(EXTENT_OPTIONS);
  const motionOptions = pick(MOTION_OPTIONS);
  const placed = new Set([...ringOptions, ...extent, ...motionOptions, 'rnodes', 'fluid']);
  const others = settings.filter((key) => !placed.has(key));
  // (a value per ring of its centre or its orientation: a switch of the
  // line of its frame that has them, not an option of its rings)
  const switches = FRAME_LINES.filter((line) => ringOptions.includes(line.option) && frame.some((p) => line.keys.some((key) => isOfKey(p, key)))).map((line) => line.option);
  const ringLines = ringOptions.filter((key) => !switches.includes(key));
  // (the problems of the component itself, not of one of its parts)
  const partOptions = new Set(parts.flatMap((p) => [p.option, ...(p.heights ? [p.heights.option] : [])]));
  const general = problems.filter((p) => startsWith(p.path, base) && !(typeof p.path[2] === 'string' && partOptions.has(p.path[2])));
  // (its groups that fold, as they are made: what folds or opens them all)
  const name = componentName(component, index);
  const foldKeys: string[] = [];
  const foldOf = (key: string): Fold => {
    foldKeys.push(key);
    const open = ctx.folds.isOpen(name, key);
    return {
      open,
      toggle: (row, all) => {
        ctx.keep(row);
        ctx.folds.set(name, all ? foldKeys : [key], !open);
      },
      show: () => ctx.folds.set(name, [key], true),
    };
  };

  const groups: Record<Section, ReactNode[]> = { geometry: [], traits: [], light: [], physics: [] };
  const feet: Record<Section, ReactNode[]> = { geometry: [], traits: [], light: [], physics: [] };
  const nodesPart = parts.find((p) => p.def === 'Nodes');
  if (nodesPart) groups.geometry.push(<RingsGroup key="rings" ctx={ctx} component={component} base={base} option={nodesPart.option} title={nodesPart.title} keys={ringLines} fold={foldOf(nodesPart.option)} />);
  for (const key of extent) {
    groups.geometry.push(
      <OptionLine key={key} ctx={ctx} owner={component} base={base} option={key}>
        {typeof component[key] === 'number' && ctx.kpcPerArcsec !== null && <span className="gm-note">{significant(component[key] * ctx.kpcPerArcsec, 3)} kpc</span>}
      </OptionLine>,
    );
  }
  if (frame.length > 0) {
    groups.geometry.push(
      <Group
        key="frame"
        title="Frame"
        description="Its centre, orientation and systemic velocity, and its other parameters of its own"
        fold={foldOf('frame')}
        summary={frame.map((p) => paramText(ctx, p)).join(' · ')}
        free={freeCount(ctx, frame)}
      >
        <FrameLines ctx={ctx} component={component} base={base} params={frame} rings={rings} switches={switches} />
      </Group>,
    );
  }
  // (the nodes of the traits of a spheroid: a setting, not a part, with
  // its traits)
  if (!nodesPart && settings.includes('rnodes') && parts.some((p) => sectionOf(p) === 'traits' && has(p))) {
    groups.traits.push(<RingsGroup key="nodes" ctx={ctx} component={component} base={base} option="rnodes" title={optionTitle(defs, 'component', component.type, 'rnodes')} keys={ringLines} fold={foldOf('rnodes')} />);
  }
  if (motion.length > 0 || motionOptions.length > 0) {
    groups.physics.push(
      <Group
        key="motion"
        title="Motion"
        description="The parameters of its motion (e.g. of the Jeans equations) and how it is made"
        fold={foldOf('motion')}
        summary={[...motion.map((p) => paramText(ctx, p)), ...motionOptions.filter((key) => component[key] !== undefined).map((key) => optionText(component, key))].join(' · ')}
        free={freeCount(ctx, motion)}
      >
        <NumberRows ctx={ctx} owned={motion} rings={rings} depth={0} />
        {motionOptions.map((key) => (
          <OptionLine key={key} ctx={ctx} owner={component} base={base} option={key} />
        ))}
      </Group>,
    );
  }
  for (const part of parts) {
    if (part === nodesPart) continue;
    const section = sectionOf(part);
    const title = subtitleOf(part, section);
    const path = [...base, part.option];
    const own = problemsOf([part.option, ...(part.heights ? [part.heights.option] : [])]);
    // (added, it opens)
    const shown = () => ctx.folds.set(name, [part.option], true);
    if (section === 'traits') {
      if (has(part) || !part.optional) groups.traits.push(<TraitGroup key={part.option} ctx={ctx} part={part} base={base} rings={rings} problems={own} fold={foldOf(part.option)} />);
      else {
        feet.traits.push(
          <AddPart key={part.option} ctx={ctx} part={part} base={base} onAdded={shown}>
            + {title.toLowerCase()} <code className="gm-key">{part.option}</code>
          </AddPart>,
        );
      }
    } else if (part.family === 'emission' || section === 'geometry') {
      // (the light, and the profile and shape of a spheroid)
      const removable = part.optional && has(part);
      groups[section].push(
        <Group
          key={part.option}
          title={title}
          keys={keysOf(title, [part.option])}
          description={part.description}
          problems={own}
          add={has(part) ? undefined : <AddPart ctx={ctx} part={part} base={base} onAdded={shown} />}
          fold={foldOf(part.option)}
          summary={itemText(component[part.option])}
          free={freeCount(ctx, ownedBy(params, path))}
        >
          {part.family !== null && has(part) && <OneItem ctx={ctx} part={part} path={path} rings={rings} onRemove={removable ? () => ctx.set(path, undefined) : undefined} />}
          {part.family === null && has(part) && <ObjectItem ctx={ctx} part={part} path={path} onRemove={removable ? () => ctx.set(path, undefined) : undefined} />}
          {!has(part) && <span className="gm-note">none</span>}
        </Group>,
      );
    } else groups[section].push(<SingleLine key={part.option} ctx={ctx} part={part} path={path} rings={rings} problems={own} />);
  }
  if (shining.length > 0) groups.light.unshift(<NumberRows key="own" ctx={ctx} owned={shining} rings={rings} depth={0} />);
  if (settings.includes('fluid')) groups.physics.push(<FluidLine key="fluid" ctx={ctx} component={component} base={base} />);
  if (others.length > 0) {
    groups.geometry.push(
      <Group key="others" title="Options" keys={others.join(' · ')} fold={foldOf('others')} summary={others.filter((key) => component[key] !== undefined).map((key) => optionText(component, key)).join(' · ') || 'as by default'}>
        <OtherOptions ctx={ctx} component={component} base={base} keys={others} />
      </Group>,
    );
  }

  return (
    <div className="gm-cbody">
      {general.length > 0 && (
        <div className="gm-why-top">
          <Problems list={general} />
        </div>
      )}
      {SECTIONS.filter((s) => groups[s.key].length > 0 || feet[s.key].length > 0).map((s) => (
        <section key={s.key} className="gm-csec" aria-label={s.title}>
          <div className="gm-rule">{s.title}</div>
          {groups[s.key]}
          {feet[s.key].length > 0 && <div className="gm-csec-foot">{feet[s.key]}</div>}
        </section>
      ))}
    </div>
  );
}

// An option of a component in a few words: its key and its value
function optionText(component: Options, key: string): string {
  const value = component[key];
  return `${key} ${isObject(value) || Array.isArray(value) ? JSON.stringify(value) : String(value)}`;
}

// The rings of a component (or its shells, or the nodes of the traits of
// a spheroid): a line of how many, to what radius and how far that is,
// which opens how they are given, and a line for each of their options.
// Nodes a component can be without (a spheroid's: by default, 0 and its
// largest radius) can be added and removed.
function RingsGroup({ ctx, component, base, option, title, keys, fold }: { ctx: Ctx; component: Options; base: Path; option: string; title: string; keys: string[]; fold: Fold }) {
  const { defs, set, problems } = ctx;
  const path = [...base, option];
  const value = component[option];
  const given = value !== undefined && value !== null;
  const required = !!defs && propertiesOf(memberOf(defs, 'component', component.type), defs).required.includes(option);
  const own = problems.filter((p) => [option, ...keys].some((key) => startsWith(p.path, [...base, key])));
  const unset = 'at 0 and its largest radius';
  // (added: to the largest radius of the component, in a few)
  const start = () => {
    set(path, { max: typeof component.rmax === 'number' ? component.rmax : 10, len: 5 });
    fold.show();
  };
  return (
    <Group
      title={title}
      keys={option}
      problems={own}
      fold={fold}
      summary={given ? ringsText(component, value) : unset}
      add={
        !given && !required ? (
          <button className="gm-plus" onClick={start}>
            + add
          </button>
        ) : undefined
      }
    >
      <div className="gm-oline">
        <span className="gm-cell-head">
          <span className="gm-oline-label">Radii</span>
        </span>
        <span className="gm-cell-nums">
          {given ? <RingsRow ctx={ctx} component={component} value={value} onChange={(next) => set(path, next)} /> : <span className="gm-note">{unset}</span>}
          <span className="gm-cell-tools">{given && !required && <Close label={`Remove the ${title.toLowerCase()}`} onClick={() => set(path, undefined)} />}</span>
        </span>
      </div>
      {keys.map((key) => (
        <OptionLine key={key} ctx={ctx} owner={component} base={base} option={key} />
      ))}
    </Group>
  );
}

// The frame of a component (its own parameters: those that place it, and
// others of its own, e.g. the flux of a point): its centre and its
// orientation, each a line with the switch of a value per ring if it has
// one and its parameters under it, and its others after them
function FrameLines({ ctx, component, base, params, rings, switches }: { ctx: Ctx; component: Options; base: Path; params: Param[]; rings: number[] | null; switches: string[] }) {
  // (in the order of their keys, e.g. xpos, ypos, vsys)
  const lines = FRAME_LINES.map((line) => ({ ...line, params: line.keys.flatMap((key) => params.filter((p) => isOfKey(p, key))) }));
  const rest = params.filter((p) => !lines.some((line) => line.params.includes(p)));
  return (
    <>
      {lines
        .filter((line) => line.params.length > 0)
        .map((line) => (
          <Fragment key={line.title}>
            <div className="gm-oline">
              <span className="gm-cell-head">
                <span className="gm-oline-label">{line.title}</span>
                {switches.includes(line.option) && <RingSwitch ctx={ctx} component={component} base={base} option={line.option} />}
              </span>
            </div>
            <NumberRows ctx={ctx} owned={line.params} rings={rings} />
          </Fragment>
        ))}
      <NumberRows ctx={ctx} owned={rest} rings={rings} depth={0} />
    </>
  );
}

// The switch of a value per ring of what a line of the frame holds (of
// its centre, loose; of its orientation, tilted)
function RingSwitch({ ctx, component, base, option }: { ctx: Ctx; component: Options; base: Path; option: string }) {
  const node = propertiesOfType(ctx.defs, component.type)[option] ?? {};
  const fallback = node.default === true;
  const on = (component[option] ?? fallback) === true;
  const title = typeof node.title === 'string' ? node.title : option;
  return (
    <button className="gm-rs" aria-pressed={on} title={`${title} (${option})`} onClick={() => ctx.set([...base, option], !on === fallback ? undefined : !on, { moved: inPlace })}>
      <RingMark />
      per ring
    </button>
  );
}

// Options of a component that no section takes (e.g. the file of
// particles): a line each those that can be, the others as a form
function OtherOptions({ ctx, component, base, keys }: { ctx: Ctx; component: Options; base: Path; keys: string[] }) {
  const { defs, set, project } = ctx;
  const properties = propertiesOfType(defs, component.type);
  const simple = keys.filter((key) => lineable(properties[key] ?? {}));
  const rest = keys.filter((key) => !simple.includes(key));
  const others = Object.keys(properties).filter((key) => !rest.includes(key));
  const node = rest.length > 0 ? restOf(defs, memberOf(defs, 'component', component.type), new Set(others)) : null;
  return (
    <>
      {simple.map((key) => (
        <OptionLine key={key} ctx={ctx} owner={component} base={base} option={key} />
      ))}
      {node && defs && (
        <div className="gm-oform">
          <SchemaForm
            defs={defs}
            node={node}
            project={project}
            value={Object.fromEntries(rest.filter((key) => component[key] !== undefined).map((key) => [key, component[key]]))}
            onChange={(next) => {
              const kept = Object.fromEntries(Object.entries(component).filter(([key]) => !rest.includes(key)));
              set(base, { ...kept, ...next }, { moved: inPlace });
            }}
          />
        </div>
      )}
    </>
  );
}

// Whether an option can be set in a line of its own (see OptionLine): a
// yes or no, a choice of a few, or a number
function lineable(node: Schema): boolean {
  if (node.type === 'boolean' || choicesOf(node)) return true;
  const branches = ((node.anyOf as Schema[] | undefined) ?? [node]).filter((b) => b.type !== 'null');
  return branches.length === 1 && (branches[0].type === 'number' || branches[0].type === 'integer');
}

// A part a component does not have, to add (by its "+ ..." or what is
// given): of one type, added at once; else its types to choose from
function AddPart({ ctx, part, base, onAdded, children }: { ctx: Ctx; part: PartInfo; base: Path; onAdded?: () => void; children?: ReactNode }) {
  const { defs } = ctx;
  const path = [...base, part.option];
  const label = children ?? '+ add';
  const set = (value: unknown) => {
    ctx.set(path, value);
    onAdded?.();
  };
  if (part.family === null) {
    const start = part.def && defs?.[part.def] ? startOfDef(defs[part.def]) : {};
    return (
      <button className="gm-plus" title={part.description} onClick={() => set(start)}>
        {label}
      </button>
    );
  }
  const types = typesOf(defs, part.family, part.types);
  const add = (type: string) => set(withDirection(startOf(defs, part.family!, type), part));
  if (types.length === 1) {
    return (
      <button className="gm-plus" title={part.description} onClick={() => add(types[0].value)}>
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
        <button className="gm-plus" title={part.description}>
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

// The options of a type of a family (by default, of components), by key
function propertiesOfType(defs: Defs | undefined, type: unknown, family = 'component'): Record<string, Schema> {
  const member = memberOf(defs, family, type);
  return (member?.properties ?? {}) as Record<string, Schema>;
}

// ---------------------------------------------------------------------
// A part of a component

// An option of a component (or of what else of a family is given, e.g. a
// part of the mass), a line of its own in the columns of the card: its
// title (and its key), and its value as its schema has it: a switch of a
// yes or no, a choice of a few, or a number (in its unit, its default or
// "auto" if it can be none, when not given), and what is given after it
// (e.g. a note). An option set to its default is left out of the
// configuration.
function OptionLine({ ctx, owner, family = 'component', base, option, children }: { ctx: Ctx; owner: Options; family?: string; base: Path; option: string; children?: ReactNode }) {
  const node = propertiesOfType(ctx.defs, owner.type, family)[option] ?? {};
  const title = typeof node.title === 'string' ? node.title : option;
  const value = owner[option];
  const fallback = node.default;
  const setOption = (next: unknown) => ctx.set([...base, option], next === fallback ? undefined : next, { moved: inPlace });
  const choices = choicesOf(node);
  let control: ReactNode;
  if (node.type === 'boolean') {
    const on = (value ?? fallback) === true;
    control = (
      <button className="gm-toggle" aria-pressed={on} aria-label={title} onClick={() => setOption(!on)}>
        {on ? 'on' : 'off'}
      </button>
    );
  } else if (choices) {
    const current = String(value ?? fallback ?? '');
    control = (
      <DropMenu
        label={title}
        width={220}
        entries={choices.map((c) => ({ label: c.title, current: String(c.value) === current, onClick: () => setOption(c.value) }))}
        target={<button className="gm-ty gm-ty-quiet">{choices.find((c) => String(c.value) === current)?.title ?? current}</button>}
      />
    );
  } else {
    // (a number: in its unit, within its bounds, a whole one if it must
    // be; emptied, none or its default)
    const branches = (node.anyOf as Schema[] | undefined) ?? [node];
    const nullable = branches.some((b) => b.type === 'null');
    const number = branches.find((b) => b.type !== 'null') ?? node;
    const fits = (n: number) => (typeof number.exclusiveMinimum === 'number' ? n > number.exclusiveMinimum : typeof number.minimum === 'number' ? n >= number.minimum : true);
    control = (
      <NumberText
        value={typeof value === 'number' ? value : NaN}
        label={title}
        unit={(node.unit ?? number.unit) === 'arcsec' ? '″' : undefined}
        placeholder={nullable ? 'auto' : typeof fallback === 'number' ? String(fallback) : undefined}
        onCommit={(typed) => {
          const next = number.type === 'integer' ? Math.round(typed) : typed;
          if (fits(next)) setOption(next);
        }}
        onClear={nullable || fallback !== undefined ? () => setOption(undefined) : undefined}
      />
    );
  }
  const keys = keysOf(title, [option]);
  return (
    <div className="gm-oline">
      <span className="gm-cell-head" title={typeof node.description === 'string' ? node.description : undefined}>
        <span className="gm-oline-label">{title}</span>
        {keys && <code className="gm-key">{keys}</code>}
      </span>
      <span className="gm-cell-nums">
        {control}
        {children}
      </span>
    </div>
  );
}

// A single setting of a component (e.g. its pressure support, its
// opacity), a line of its own: its title (and its key), and beside it its
// type (or none), its other options and its parameters under it; of
// options of their own (e.g. clouds), on or off and those options
function SingleLine({ ctx, part, path, rings, problems }: { ctx: Ctx; part: PartInfo; path: Path; rings: number[] | null; problems: Problem[] }) {
  const { defs, options, params, set, project } = ctx;
  const [more, setMore] = useState(false);
  const value = getAt(options, path);
  const item = isObject(value) ? value : null;
  let pick: ReactNode;
  let form: ReactNode = null;
  let count = 0;
  if (part.family === null) {
    const start = part.def && defs?.[part.def] ? startOfDef(defs[part.def]) : {};
    pick = (
      <DropMenu
        label={part.title}
        width={220}
        entries={[
          { label: 'Off', current: item === null, onClick: () => set(path, undefined) },
          { label: 'On', description: part.description, current: item !== null, onClick: () => item === null && set(path, start) },
        ]}
        target={<button className="gm-ty gm-ty-quiet">{item ? 'on' : 'off'}</button>}
      />
    );
    if (item && part.def && defs?.[part.def]) {
      count = Object.keys(item).length;
      form = <SchemaForm defs={defs} def={part.def} project={project} value={item} onChange={(next) => set(path, next)} />;
    }
  } else {
    const family = part.family;
    pick = (
      <TypePick
        items={typesOf(defs, family, part.types)}
        value={String(item?.type ?? '')}
        text={item ? String(item.type) : 'none'}
        none={part.optional ? 'None' : undefined}
        label={part.title}
        onChange={(type) => set(path, type === '' ? undefined : retyped(defs, family, item, type))}
        quiet
      />
    );
    const rest = item ? restOf(defs, memberOf(defs, family, item.type), new Set(['type'])) : null;
    if (rest && defs && item) {
      count = Object.keys(item).filter((key) => key !== 'type').length;
      form = <SchemaForm defs={defs} node={rest} project={project} value={omit(item, 'type')} onChange={(next) => set(path, { type: item.type, ...next })} />;
    }
  }
  const keys = keysOf(part.title, [String(path[path.length - 1])]);
  return (
    <>
      <div className="gm-single" data-bad={problems.length > 0 || undefined}>
        <span className="gm-cell-head" title={part.description}>
          <span className="gm-oline-label">{part.title}</span>
          {keys && <code className="gm-key">{keys}</code>}
        </span>
        <span className="gm-cell-nums">
          {pick}
          {form && <MoreButton on={more} count={count} onClick={() => setMore(!more)} />}
        </span>
      </div>
      {form && (
        <Expand open={more}>
          <div className="gm-more-form">{form}</div>
        </Expand>
      )}
      {item && part.family !== null && <NumberRows ctx={ctx} owned={ownedBy(params, path)} rings={rings} />}
      <Problems list={problems} />
    </>
  );
}

// The gas disks of the mass whose densities the pressure support of a
// disk takes (by default, the disks of the mass that follow it): a line of
// its own, a menu of the parts of the mass, each in or out
function FluidLine({ ctx, component, base }: { ctx: Ctx; component: Options; base: Path }) {
  const { defs, options, problems, set } = ctx;
  const path = [...base, 'fluid'];
  const node = propertiesOfType(defs, component.type).fluid ?? {};
  const title = typeof node.title === 'string' ? node.title : 'fluid';
  const value = component.fluid;
  const chosen = typeof value === 'string' ? [value] : Array.isArray(value) ? value.filter((name): name is string => typeof name === 'string') : [];
  const names = asList(options.mass)
    .map((m) => m.name)
    .filter((name): name is string => typeof name === 'string' && name !== '');
  const choose = (next: string[]) => set(path, next.length === 0 ? undefined : next.length === 1 ? next[0] : next, { moved: inPlace });
  const own = problems.filter((p) => startsWith(p.path, path));
  return (
    <>
      <div className="gm-single" data-bad={own.length > 0 || undefined}>
        <span className="gm-cell-head" title={typeof node.description === 'string' ? node.description : undefined}>
          <span className="gm-oline-label">{title}</span>
          <code className="gm-key">fluid</code>
        </span>
        <span className="gm-cell-nums">
          <DropMenu
            label={title}
            width={260}
            entries={[
              { label: 'The disks that follow it', current: chosen.length === 0, onClick: () => choose([]) },
              ...names.map((name) => ({ label: name, current: chosen.includes(name), onClick: () => choose(chosen.includes(name) ? chosen.filter((n) => n !== name) : [...chosen, name]) })),
            ]}
            target={<button className="gm-ty gm-ty-quiet">{chosen.length > 0 ? chosen.join(', ') : 'the disks that follow it'}</button>}
          />
        </span>
      </div>
      <Problems list={own} />
    </>
  );
}

// A list of traits (e.g. the velocity of a disk), a group under its
// title and its keys (of its traits and their heights), with what adds
// one: a line for each, with the height paired with it (or one that they
// share, after them); folded, their types, directions and heights
function TraitGroup({ ctx, part, base, rings, problems, fold }: { ctx: Ctx; part: PartInfo; base: Path; rings: number[] | null; problems: Problem[]; fold: Fold }) {
  const { defs, draft, options, params } = ctx;
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
    fold.show();
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
  const types = typesOf(defs, family, part.types);
  // (heights that can be none show where they are given; none given, what
  // gives one to all)
  const hlist = Array.isArray(given) ? given : given === undefined || given === null ? [] : [given];
  const noHeights = !!heights && heights.nullable && !hlist.some(isObject);
  const shown = (i: number) => !!heights && !shared && (!heights.nullable || isObject(hlist[i]));
  const heightActions = heights !== null && hpath !== null && (noHeights || (items.length > 1 && (shared || given !== undefined)));
  const typeOf = (value: unknown) => (isObject(value) ? String(value.type) : '');
  const summary =
    items.length === 0
      ? 'none'
      : [
          ...items.map((item, i) => (
            <span key={i}>
              {i > 0 && ' · '}
              {typeOf(item)}
              {typeof item.direction === 'string' && (
                <span className="gm-dir">
                  <DirectionMark direction={item.direction} />
                  {item.direction}
                </span>
              )}
              {shown(i) && ` ↳ ${typeOf(hlist[i])}`}
            </span>
          )),
          ...(shared ? [<span key="all"> · ↳ all {typeOf(hlist[0])}</span>] : []),
        ];
  return (
    <Group
      title={part.title}
      keys={keysOf(part.title, [part.option, ...(heights ? [heights.option] : [])])}
      description={part.description}
      problems={problems}
      fold={fold}
      summary={summary}
      free={freeCount(ctx, [...ownedBy(params, listPath), ...(hpath ? ownedBy(params, hpath) : [])])}
      add={
        <>
          {heightActions &&
            (noHeights ? (
              <button className="gm-plus gm-hover" title={heights.description} onClick={() => ctx.set(hpath, newHeight())}>
                + height
              </button>
            ) : (
              <button className="gm-plus gm-hover" onClick={() => share(!shared)}>
                {shared ? 'a height for each' : 'one height for all'}
              </button>
            ))}
          {types.length === 1 ? (
            <button className="gm-plus" title={types[0].description} onClick={() => add(types[0].value)}>
              + add
            </button>
          ) : (
            <DropMenu
              label={`Add to the ${part.title.toLowerCase()}`}
              width={300}
              entries={types.map((t) => ({ label: t.label, description: t.description, onClick: () => add(t.value) }))}
              target={<button className="gm-plus">+ add</button>}
            />
          )}
        </>
      }
    >
      <div className="gm-list">
        {items.map((item, i) => (
          <div key={i} className="gm-item">
            <ItemLine ctx={ctx} part={part} path={[...listPath, i]} item={item} rings={rings} onRemove={() => remove(i)} />
            {shown(i) && <HeightLine ctx={ctx} part={heights!} path={[...hpath!, i]} count={items.length} rings={rings} />}
          </div>
        ))}
        {heights && shared && <HeightLine ctx={ctx} part={heights} path={[...hpath!, 0]} count={1} rings={rings} all />}
        {items.length === 0 && <span className="gm-note">none</span>}
      </div>
    </Group>
  );
}

// One item of a family (e.g. the light of a component)
function OneItem({ ctx, part, path, rings, onRemove }: { ctx: Ctx; part: PartInfo; path: Path; rings: number[] | null; onRemove?: () => void }) {
  const item = getAt(ctx.options, path);
  return <ItemLine ctx={ctx} part={part} path={path} item={isObject(item) ? item : {}} rings={rings} onRemove={onRemove} />;
}

// An item's line: its type (and direction), the lines of an emission of
// lines; its other options behind "options", and what removes it, as the
// line is pointed at; its parameters a line each under it
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
        <span className="gm-cell-type">
          <TypePick items={typesOf(defs, family, part.types)} value={String(item.type ?? '')} text={String(item.type ?? '')} label={`The type of the ${part.title.toLowerCase()}`} onChange={(type) => set(path, retyped(defs, family, item, type))} />
        </span>
        <span className="gm-cell-dir">
          {directions && directions.length > 0 && (
            <DropMenu
              label="Its direction"
              width={240}
              entries={directions.map((d) => ({ label: d.title, current: String(d.value) === direction, onClick: () => set(path, { ...item, direction: d.value }) }))}
              target={
                <button className="gm-dir-pick" title={`Its direction: ${directions.find((d) => String(d.value) === direction)?.title ?? direction}`}>
                  <DirectionMark direction={direction} />
                  {direction}
                </button>
              }
            />
          )}
        </span>
        <span className="gm-cell-nums">
          {lines && <LinesLine emission={item} onChange={(next) => set(path, next)} />}
          <span className="gm-cell-tools">
            {rest && <MoreButton on={more} count={given} onClick={() => setMore(!more)} />}
            {onRemove && <Close label={`Remove it`} onClick={onRemove} />}
          </span>
        </span>
      </div>
      {rest && defs && (
        <Expand open={more}>
          <div className="gm-more-form">
            <SchemaForm
              defs={defs}
              node={rest}
              project={project}
              value={omitKeys(item, without)}
              onChange={(next) => set(path, { ...Object.fromEntries(Object.entries(item).filter(([key]) => without.has(key))), ...next }, { moved: inPlace })}
            />
          </div>
        </Expand>
      )}
      <NumberRows ctx={ctx} owned={ownedBy(params, path)} rings={rings} />
    </>
  );
}

// The height trait paired with a trait (or the one they share): its type,
// or none if it can be, and its parameters a line each under it
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
        <span className="gm-cell-head">
          <span className="gm-times" title={all ? 'The height of them all' : 'Its height'}>
            {all ? '↳ all' : '↳'}
          </span>
          <TypePick
            items={typesOf(defs, family, part.types)}
            value={String(item?.type ?? '')}
            text={item ? String(item.type) : undefined}
            none={part.nullable ? 'the same at every height' : undefined}
            label={part.title}
            onChange={(type) => set(type === '' ? null : retyped(defs, family, item, type))}
          />
        </span>
        <span className="gm-cell-nums">
          <span className="gm-cell-tools">{rest && <MoreButton on={more} count={Object.keys(item ?? {}).filter((key) => key !== 'type').length} onClick={() => setMore(!more)} />}</span>
        </span>
      </div>
      {rest && defs && item && (
        <Expand open={more}>
          <div className="gm-more-form">
            <SchemaForm defs={defs} node={rest} project={project} value={omit(item, 'type')} onChange={(next) => set({ type: item.type, ...next })} />
          </div>
        </Expand>
      )}
      {item && <NumberRows ctx={ctx} owned={ownedBy(params, path)} rings={rings} limit={typeof item.zmax === 'number' ? item.zmax : undefined} depth={2} />}
    </>
  );
}

// Options of their own (e.g. the clouds of a disk), as a form
function ObjectItem({ ctx, part, path, onRemove }: { ctx: Ctx; part: PartInfo; path: Path; onRemove?: () => void }) {
  const { defs, options, set, project } = ctx;
  const value = getAt(options, path);
  if (!defs || !part.def || !defs[part.def]) return null;
  return (
    <div className="gm-object gm-span">
      <SchemaForm defs={defs} def={part.def} project={project} value={isObject(value) ? value : {}} onChange={(next) => set(path, next)} />
      {onRemove && <Close label={`Remove the ${part.title.toLowerCase()}`} onClick={onRemove} />}
    </div>
  );
}

// ---------------------------------------------------------------------
// A part of the mass

// The components a part of the mass can follow (e.g. a traced disk, a
// disk; a traced spheroid, a spheroid): those of its kind, the last word
// of their types
function followable(components: Options[], type: unknown): Options[] {
  return components.filter((c) => nameBase(String(c.type)) === nameBase(String(type)));
}

// A part of the mass: a row as that of a component (its name and type, and
// under them, closed, what it follows and its numbers in a few words),
// which opens it; open, a line each, as in the card of a component: what
// it follows, its parts of a type (e.g. the profile of a halo), and its
// parameters and its options, each a group that folds
function MassRow({
  ctx,
  index,
  item,
  colour,
  components,
  open,
  onToggle,
  onRemoved,
}: {
  ctx: Ctx;
  index: number;
  item: Options;
  // Its colour (a token: of the component it follows, if it does)
  colour: string;
  components: Options[];
  open: boolean;
  onToggle: (row: HTMLElement) => void;
  onRemoved: () => void;
}) {
  const { defs, params, problems, set, project } = ctx;
  const [renaming, setRenaming] = useState(false);
  const path: Path = ['mass', index];
  const refs = referencesOf(defs, 'mass_component', item.type);
  const follows = Object.entries(refs).filter(([, what]) => what === 'components').map(([key]) => key);
  const followed = follows.map((key) => item[key]).find((name) => typeof name === 'string');
  const own = problems.filter((p) => startsWith(p.path, path));
  const member = memberOf(defs, 'mass_component', item.type);
  // (its options, each where it goes: the components it follows; its parts
  // of a type, e.g. the profile of a halo, a line each as a single setting
  // of a component; those that can be a line; the others as a form)
  const { properties, required } = defs ? propertiesOf(member, defs) : { properties: {} as Record<string, Schema>, required: [] as string[] };
  const keys = Object.keys(properties).filter((key) => !HEAD_OPTIONS.has(key) && !follows.includes(key));
  const familyOf = (key: string) => familyOfOption(defs, 'mass_component', item.type, key);
  const typed = keys.filter((key) => {
    const family = familyOf(key);
    return family !== null && typesOf(defs, family).length > 0;
  });
  const simple = keys.filter((key) => !typed.includes(key) && lineable(properties[key]));
  const others = keys.filter((key) => !typed.includes(key) && !simple.includes(key));
  const form = others.length > 0 ? restOf(defs, member, new Set(Object.keys(properties).filter((key) => !others.includes(key)))) : null;
  // (its own parameters: not those of its parts of a type, under them)
  const owned = ownedBy(params, path).filter((p) => !typed.some((key) => startsWith(p.owner, [...path, key])));
  const partOf = (key: string): PartInfo => ({
    option: key,
    title: typeof properties[key].title === 'string' ? properties[key].title : key,
    description: typeof properties[key].description === 'string' ? properties[key].description : undefined,
    family: familyOf(key),
    def: null,
    many: false,
    optional: !required.includes(key),
    nullable: false,
    types: null,
    directions: null,
    heights: null,
  });
  const within = (key: string) => own.filter((p) => startsWith(p.path, [...path, key]));
  const general = own.filter((p) => !typed.some((key) => startsWith(p.path, [...path, key])));
  const typeTitle = titled(defs, 'mass_component', String(item.type)).label;
  const name = String(item.name ?? '');
  // (in a few words: the types of what it is made of, e.g. the profile of a
  // halo; what it follows; its numbers)
  const inner = Object.entries(item)
    .filter(([key, value]) => key !== 'type' && isObject(value) && typeof value.type === 'string')
    .map(([key, value]) => titled(defs, familyOfOption(defs, 'mass_component', item.type, key) ?? '', String((value as Options).type)).label);
  const words = [...inner, followed ? `follows ${String(followed)}` : null, ...owned.slice(0, 3).map((p) => paramText(ctx, p))].filter(Boolean);
  // (the components it can follow; the one it follows, if not of them)
  const choices = followable(components, item.type).map((c) => componentName(c, components.indexOf(c)));
  // (its groups that fold, as they are made: what folds or opens them all)
  const owner = massFolds(name);
  const foldKeys: string[] = [];
  const foldOf = (key: string): Fold => {
    foldKeys.push(key);
    const isOpen = ctx.folds.isOpen(owner, key);
    return {
      open: isOpen,
      toggle: (row, all) => {
        ctx.keep(row);
        ctx.folds.set(owner, all ? foldKeys : [key], !isOpen);
      },
      show: () => ctx.folds.set(owner, [key], true),
    };
  };
  const setOptions = simple.filter((key) => item[key] !== undefined).map((key) => optionText(item, key));
  const formOptions = others.filter((key) => item[key] !== undefined).map((key) => optionText(item, key));
  // (what refers to it by name, e.g. a halo that responds to it, with it)
  const rename = (next: string) => {
    setRenaming(false);
    if (!next) return;
    const old = item.name;
    // (its open groups with it)
    ctx.folds.rename(owner, massFolds(next));
    ctx.draft.edit(
      (s) => {
        const named = setAt(s.options, [...path, 'name'], next);
        return { ...s, options: typeof old === 'string' ? withReferencesRenamed(defs, named, 'mass', old, next) : named };
      },
      { moved: inPlace },
    );
  };
  return (
    <div className="gm-row gm-mrow" data-open={open || undefined} data-bad={own.length > 0 || undefined} data-at={`mass/${index}`}>
      <div className="gm-crow" role="button" tabIndex={0} aria-expanded={open} onClick={(event) => !renaming && onToggle(event.currentTarget)} onKeyDown={(event) => event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ') && (event.preventDefault(), onToggle(event.currentTarget))}>
        <span className="gm-dot" style={{ background: `var(--${colour})` }} />
        <span className="gm-cname">
          {renaming ? (
            <span onClick={(event) => event.stopPropagation()}>
              <NameText value={name} required onCommit={rename} onCancel={() => setRenaming(false)} autoFocus />
            </span>
          ) : (
            name
          )}
          <span>{typeTitle}</span>
        </span>
        <span className="gm-cext" />
        <span className="gm-cmenu" onClick={(event) => event.stopPropagation()}>
          <DropMenu
            label={`The mass ${name}`}
            width={240}
            at="end"
            // (its rename gives the focus to the field of its name)
            returnFocus={false}
            entries={[
              { label: 'Rename', onClick: () => setRenaming(true) },
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
        {!open && words.length > 0 && <span className="gm-csum">{words.join(' · ')}</span>}
      </div>
      <Expand open={open}>
        <div className="gm-cbody gm-mbody">
          {follows.map((key) => (
            <div key={key} className="gm-single">
              <span className="gm-cell-head">
                <span className="gm-oline-label">Follows</span>
                <code className="gm-key">{key}</code>
              </span>
              <span className="gm-cell-nums">
                {choices.length > 0 || typeof item[key] === 'string' ? (
                  <TypePick
                    items={[...(typeof item[key] === 'string' && !choices.includes(item[key]) ? [item[key]] : []), ...choices].map((choice) => ({ value: choice, label: choice }))}
                    value={String(item[key] ?? '')}
                    text={typeof item[key] === 'string' ? item[key] : undefined}
                    label="The component it follows"
                    onChange={(choice) => set([...path, key], choice)}
                    quiet
                  />
                ) : (
                  <span className="gm-note">no component of its kind to follow</span>
                )}
              </span>
            </div>
          ))}
          {typed.map((key) => (
            <SingleLine key={key} ctx={ctx} part={partOf(key)} path={[...path, key]} rings={null} problems={within(key)} />
          ))}
          {owned.length > 0 && (
            <Group title="Parameters" fold={foldOf('parameters')} summary={owned.map((p) => paramText(ctx, p)).join(' · ')} free={freeCount(ctx, owned)}>
              <NumberRows ctx={ctx} owned={owned} rings={null} depth={0} />
            </Group>
          )}
          {(simple.length > 0 || form) && (
            <Group title="Options" keys={[...simple, ...others].join(' · ')} fold={foldOf('options')} summary={[...setOptions, ...formOptions].join(' · ') || 'as by default'}>
              {simple.map((key) => (
                <OptionLine key={key} ctx={ctx} owner={item} family="mass_component" base={path} option={key} />
              ))}
              {form && defs && (
                <div className="gm-oform">
                  <SchemaForm
                    defs={defs}
                    node={form}
                    project={project}
                    value={Object.fromEntries(others.filter((key) => item[key] !== undefined).map((key) => [key, item[key]]))}
                    onChange={(next) => set(path, { ...omitKeys(item, new Set(others)), ...next }, { moved: inPlace })}
                  />
                </div>
              )}
            </Group>
          )}
          {general.length > 0 && (
            <div className="gm-why-top">
              <Problems list={general} />
            </div>
          )}
        </div>
      </Expand>
    </div>
  );
}

// ---------------------------------------------------------------------
// Small parts

// A type of a family to choose (e.g. of a velocity trait): its title (or
// the text given, e.g. its type), a menu of the types, each with what it
// is; none, if it can be
function TypePick({ items, value, onChange, label, none, text, quiet }: { items: TypeChoice[]; value: string; onChange: (type: string) => void; label: string; none?: string; text?: string; quiet?: boolean }) {
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
        <button className={quiet ? 'gm-ty gm-ty-quiet' : 'gm-ty'} title={current?.label}>
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
export function NameText({ value, onCommit, required, onCancel, autoFocus }: { value: string; onCommit: (name: string) => void; required: boolean; onCancel?: () => void; autoFocus?: boolean }) {
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
function NumberText({ value, onCommit, label, unit, bare, placeholder, onClear }: { value: number; onCommit: (value: number) => void; label: string; unit?: string; bare?: boolean; placeholder?: string; onClear?: () => void }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <label className={bare ? 'gm-numtext gm-numtext-bare' : 'gm-numtext'}>
      <input
        value={text ?? (Number.isFinite(value) ? String(value) : '')}
        placeholder={placeholder}
        onChange={(event) => {
          const next = event.currentTarget.value;
          setText(next);
          const number = Number(next);
          if (next.trim() !== '' && Number.isFinite(number) && number >= 0) onCommit(number);
        }}
        // (emptied, none, if it can be)
        onBlur={() => {
          if (text !== null && text.trim() === '' && onClear) onClear();
          setText(null);
        }}
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
        <span className="gm-note" title="One line, at the velocity of the spectral axis of the data, whatever it is">
          the line of the data
        </span>
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

// A problem as people read it: without the place the library prefixes
// (e.g. '[pieces.0] ') nor the names of the errors of Python (e.g.
// 'ValueError: ')
export function problemText(message: string): string {
  return message.replace(/^\[[^\]]*\] /, '').replace(/\b[A-Za-z]+Error: /g, '');
}

function omit(options: Options, key: string): Options {
  const { [key]: _, ...rest } = options;
  return rest;
}

function omitKeys(options: Options, keys: Set<string>): Options {
  return Object.fromEntries(Object.entries(options).filter(([key]) => !keys.has(key)));
}

// The rings (or shells) of a component in a few words: how many and to
// what radius
function ringsText(component: Options, value: unknown): string {
  const radii = ringRadii(value);
  const max = radii.length ? radii[radii.length - 1] : 0;
  return `${radii.length} ${component.type === 'cone' ? 'shells' : 'rings'} · 0–${Number(max.toPrecision(4))}″`;
}

// The rings (or shells) of a component in a few words (see ringsText) and
// how far their largest radius is; a click opens how they are given
function RingsRow({ ctx, component, value, onChange }: { ctx: Ctx; component: Options; value: unknown; onChange: (value: unknown) => void }) {
  const unit = component.type === 'cone' ? 'shells' : 'rings';
  const radii = ringRadii(value);
  const max = radii.length ? radii[radii.length - 1] : 0;
  return (
    <>
      <Popover position="bottom-start" offset={4} width={340} shadow="md" withinPortal trapFocus={false} classNames={{ dropdown: 'gm-pop' }}>
        <Popover.Target>
          <button className="gm-num gm-text" title={`How its ${unit} are given`}>
            {ringsText(component, value)}
          </button>
        </Popover.Target>
        <Popover.Dropdown>
          <RingsLine value={value} onChange={onChange} unit={unit} />
        </Popover.Dropdown>
      </Popover>
      {ctx.kpcPerArcsec !== null && max > 0 && <span className="gm-note">{significant(max * ctx.kpcPerArcsec, 3)} kpc</span>}
    </>
  );
}
