// The model of a galaxy, as its sheet shows it (see ui/pieces and
// ui/rows): the galaxy as a whole (its distance and its redshift, rows),
// its mass (under its gravity, a piece each part) and its components (a
// piece each). A piece is a line (its name and type, and folded, what it
// is in a few words) that unfolds, as many as wanted, kept as the page is
// reloaded; a click on its line selects it. An unfolded component is in
// sections in the order of its configuration (its geometry, its traits,
// its light, its physics), each under a quiet caption, of items (its
// parts: e.g. its frame, its velocity) that fold, each a line (its title,
// and folded, what it has in a few words and how many of its numbers a
// fit frees; which are open is remembered), and rows of single settings.
// Every row is on the grid of the sheet: what it is, its value (one
// column for the whole sheet), its unit, how a fit takes it. A problem
// shows where it is, and the count of them at the foot goes to each in
// turn.
import { Fragment, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useComponentTypes, useLines } from '../../api/queries';
import { choicesOf, SchemaForm, type Defs, type Schema } from '../../components/SchemaForm';
import { significant } from '../../components/format';
import { Close } from '../../components/ui/basics';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import { AddMenu, FoldAll, PieceHead } from '../../components/ui/pieces';
import { NumberInput, Row } from '../../components/ui/rows';
import { useRemembered } from '../../shell/remember';
import type { EditHow, GalaxyDraft, Problem } from './draft';
import { DirectionMark, RingMark } from './Marks';
import { freeCount, ParamLine, paramText, type NumbersContext } from './Numbers';
import { asList, asOneOrList, componentName, componentsOf, freshName, getAt, isObject, itemText, lineList, lineName, ringRadii, roundStep, setAt, startsWith, type Options, type Path } from './options';
import { unitText } from './ParamField';
import { ownedBy, settingOf, type Param } from './params';
import { memberOf, propertiesOf, restOf, retyped, startOf, titled, typesOf, type TypeChoice } from './schema';
import { extentOf, familyOfOption, HEAD_OPTIONS, nameBase, optionTitle, partsOfType, referencesOf, settingsOfType, withReferencesRenamed, type ComponentTypes, type PartInfo } from './types';

interface Ctx extends NumbersContext {
  defs: Defs | undefined;
  types: ComponentTypes | undefined;
  project: string;
  problems: Problem[];
  set: (path: Path, value: unknown, how?: EditHow) => void;
  // The length of an arcsec at the distance of the galaxy (kpc), if known
  kpcPerArcsec: number | null;
  folds: Folds;
  // What keeps a line where it is in its pane as what is around it opens
  // or closes (see useKeptInPlace)
  keep: (line: HTMLElement) => void;
  // The row open (of a parameter, by its name: its slider, or what sets
  // it), one at a time; what opens a row, or closes it if it is open
  row: string | null;
  onRow: (name: string) => void;
}

// The items of the pieces that are open, by the key of their owner (a
// component, or a part of the mass: see massKey) and the key of the item
// (e.g. 'vptraits'): all folded at first, and kept as the pieces fold and
// unfold, and as the page is reloaded (in this browser, for the galaxy)
interface Folds {
  isOpen: (owner: string, item: string) => boolean;
  set: (owner: string, items: string[], open: boolean) => void;
  rename: (from: string, to: string) => void;
}

// The key of a part of the mass, among the pieces and the folds (apart
// from a component of the same name, whose key is its name)
const massKey = (name: string) => `mass:${name}`;

function useFolds(project: string, galaxy: string): Folds {
  const [open, setOpen] = useRemembered<Record<string, string[]>>(`galaxy-folds:${project}/${galaxy}`, {});
  return {
    isOpen: (owner, item) => open[owner]?.includes(item) ?? false,
    set: (owner, items, on) =>
      setOpen((all) => {
        const others = (all[owner] ?? []).filter((item) => !items.includes(item));
        return { ...all, [owner]: on ? [...others, ...items] : others };
      }),
    rename: (from, to) => setOpen(({ [from]: moved, ...rest }) => (moved ? { ...rest, [to]: moved } : rest)),
  };
}

// An item of a piece that folds (see Folds): whether it is open; what
// opens or folds it, kept where it is (all: every item of its piece, as
// it goes); what opens it (e.g. as something is added to it)
interface Fold {
  open: boolean;
  toggle: (line: HTMLElement, all: boolean) => void;
  show: () => void;
}

// The folds of the items of a piece (by its key), as they are made: each
// item's, which folds or opens it, or (all) every item made before
function foldsOf(ctx: Ctx, owner: string): (item: string) => Fold {
  const items: string[] = [];
  return (item) => {
    items.push(item);
    const open = ctx.folds.isOpen(owner, item);
    return {
      open,
      toggle: (line, all) => {
        ctx.keep(line);
        ctx.folds.set(owner, all ? items : [item], !open);
      },
      show: () => ctx.folds.set(owner, [item], true),
    };
  };
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
  // The pieces unfolded, by their keys (a component's name, see massKey),
  // in the order they were; the piece selected
  open: string[];
  onOpen: (change: (open: string[]) => string[]) => void;
  selected: string | null;
  onSelect: (key: string | null) => void;
  isolated: string | null;
  onIsolate: (name: string | null) => void;
}

export function Model({ draft, project, galaxy, field, colours, massColours, open, onOpen, selected, onSelect, isolated, onIsolate }: ModelProps) {
  const types = useComponentTypes().data as ComponentTypes | undefined;
  const options = draft.state?.options ?? {};
  const components = componentsOf(options);
  const mass = asList(options.mass);
  const set = (path: Path, value: unknown, how?: EditHow) => draft.edit((s) => ({ ...s, options: setAt(s.options, path, value) }), how);
  const folds = useFolds(project, galaxy);
  const { room, keep } = useKeptInPlace();
  const [row, setRow] = useState<string | null>(null);
  const ctx: Ctx = {
    draft,
    options,
    params: draft.params,
    field,
    defs: draft.defs,
    types,
    project,
    problems: draft.problems,
    set,
    kpcPerArcsec: kpcPerArcsecOf(draft),
    folds,
    keep,
    row,
    onRow: (name) => setRow((r) => (r === name ? null : name)),
  };
  // (Escape selects none)
  useEffect(() => {
    if (selected === null) return;
    const key = (event: KeyboardEvent) => event.key === 'Escape' && !(event.target instanceof HTMLInputElement) && onSelect(null);
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [selected, onSelect]);

  // (a piece folded or unfolded by its chevron; selected by its line, and
  // unfolded, or, if it was, selected no longer; kept where it is)
  const fold = (key: string, line: HTMLElement) => {
    keep(line);
    onOpen((o) => (o.includes(key) ? o.filter((k) => k !== key) : [...o, key]));
  };
  const select = (key: string, line: HTMLElement) => {
    keep(line);
    if (selected === key) return onSelect(null);
    onSelect(key);
    onOpen((o) => (o.includes(key) ? o : [...o, key]));
  };
  // (what the sheet shows of a piece, as its key is replaced, or gone)
  const rekey = (from: string, to: string | null) => {
    onOpen((o) => o.flatMap((k) => (k !== from ? [k] : to === null ? [] : [to])));
    if (selected === from) onSelect(to);
  };
  // (a piece added or copied: unfolded and selected)
  const show = (key: string) => {
    onOpen((o) => [...o.filter((k) => k !== key), key]);
    onSelect(key);
  };

  const addComponent = (type: string) => {
    // (several components are all named: one without a name is named
    // after its type, and its parameters with it)
    const named = components.map((c, i) => (typeof c.name === 'string' && c.name ? c : { ...c, name: freshName(components.map((o) => o.name), c.type ? nameBase(String(c.type)) : `component${i + 1}`) }));
    const added = { ...startOf(draft.defs, 'component', type), name: freshName(named.map((c) => c.name), nameBase(type)) };
    draft.edit((s) => ({ ...s, options: { ...s.options, components: [...named, added] } }), { moved: inPlace });
    named.forEach((c, i) => {
      if (c === components[i]) return;
      folds.rename(componentName(components[i], i), String(c.name));
      rekey(componentName(components[i], i), String(c.name));
    });
    show(String(added.name));
  };

  const addMass = (type: string) => {
    const refs = referencesOf(draft.defs, 'mass_component', type);
    // (one that follows a component: the first it can follow; else the
    // first)
    const follow = followable(components, type)[0] ?? components[0];
    const extra = Object.fromEntries(Object.entries(refs).filter(([, what]) => what === 'components').map(([key]) => [key, follow ? componentName(follow, components.indexOf(follow)) : undefined]));
    const names = mass.map((m) => m.name);
    const name = freshName(names, typeof extra.component === 'string' && !names.includes(extra.component) ? extra.component : nameBase(type));
    draft.edit((s) => {
      let next = s.options;
      // (mass needs the distance of its galaxy, to turn arcsec into kpc)
      if (next.distance === undefined) next = setAt(next, ['distance'], { type: 'parameter' });
      const item = startOf(draft.defs, 'mass_component', type, [], { ...extra, name });
      return { ...s, options: setAt(next, ['mass'], [...asList(next.mass), item]) };
    });
    show(massKey(name));
  };

  const componentTypes = Object.keys(types ?? {}).map((type) => ({ value: type, ...titled(draft.defs, 'component', type) }));
  const massTypes = typesOf(draft.defs, 'mass_component');
  const componentKeys = components.map((c, i) => componentName(c, i));
  const massKeys = mass.map((m) => massKey(String(m.name ?? '')));
  const foldAll = (what: string, keys: string[]) => (
    <FoldAll
      what={what}
      canUnfold={keys.some((k) => !open.includes(k))}
      canFold={keys.some((k) => open.includes(k))}
      onUnfold={() => onOpen((o) => [...o.filter((k) => !keys.includes(k)), ...keys])}
      onFold={() => onOpen((o) => o.filter((k) => !keys.includes(k)))}
    />
  );
  const general = draft.problems.filter((p) => !startsWith(p.path, ['components']) && !startsWith(p.path, ['mass']));
  // (the problems of a component or a part of the mass: the count goes to
  // each in turn, unfolding it and bringing its first problem in sight)
  const placed = draft.problems.filter((p) => !general.includes(p) && typeof p.path[1] === 'number');
  const [turn, setTurn] = useState(0);
  const goToProblem = () => {
    const problem = placed[turn % placed.length];
    setTurn(turn + 1);
    const [section, index] = problem.path as [string, number];
    const key = section === 'mass' ? massKeys[index] : componentKeys[index];
    if (key !== undefined) onOpen((o) => (o.includes(key) ? o : [...o, key]));
    // (once it is unfolded)
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        const piece = document.querySelector(`[data-at="${section}/${index}"]`);
        (piece?.querySelector('.gm-why') ?? piece)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }),
    );
  };
  const problemCount = `${draft.problems.length} ${draft.problems.length === 1 ? 'problem' : 'problems'}`;

  return (
    <div className="gm">
      <GalaxyBlock ctx={ctx} />

      <section className="sh-block">
        <h3 className="sh-head">
          Mass <span className="sh-count">{mass.length}</span>
          <Gravity ctx={ctx} hasMass={mass.length > 0} />
          <span className="sh-acts">
            {foldAll('part of the mass', massKeys)}
            <AddMenu label="Add a part of the mass" width={320} entries={massTypes.map((t) => ({ label: t.label, description: t.description, onClick: () => addMass(t.value) }))} />
          </span>
        </h3>
        <GravityRows ctx={ctx} />
        {mass.length === 0 ? (
          <p className="sh-note">No mass: the motions are as the traits give them.</p>
        ) : (
          <div className="sh-pieces">
            {mass.map((item, i) => (
              <MassPiece
                key={i}
                ctx={ctx}
                index={i}
                item={item}
                colour={massColours[i]}
                components={components}
                open={open.includes(massKeys[i])}
                selected={selected === massKeys[i]}
                onFold={(line) => fold(massKeys[i], line)}
                onSelect={(line) => select(massKeys[i], line)}
                onRekey={rekey}
              />
            ))}
          </div>
        )}
      </section>

      <section className="sh-block">
        <h3 className="sh-head">
          Components <span className="sh-count">{components.length}</span>
          <span className="sh-acts">
            {foldAll('component', componentKeys)}
            <AddMenu label="Add a component" width={320} entries={componentTypes.map((t) => ({ label: t.label, description: t.description, onClick: () => addComponent(t.value) }))} />
          </span>
        </h3>
        {components.length === 0 ? (
          <p className="sh-note">No components: nothing emits light.</p>
        ) : (
          <div className="sh-pieces">
            {components.map((component, i) => (
              <ComponentPiece
                key={i}
                ctx={ctx}
                index={i}
                component={component}
                colour={colours[i]}
                open={open.includes(componentKeys[i])}
                selected={selected === componentKeys[i]}
                onFold={(line) => fold(componentKeys[i], line)}
                onSelect={(line) => select(componentKeys[i], line)}
                onRekey={rekey}
                onShow={show}
                isolated={isolated === componentKeys[i]}
                onIsolate={onIsolate}
              />
            ))}
          </div>
        )}
      </section>

      {(draft.problems.length > 0 || draft.describeError) && (
        <div className="gm-problems">
          {draft.problems.length > 0 && (
            <span>
              {placed.length > 0 ? (
                <button className="gm-link" onClick={goToProblem} title="Unfold where the next one is">
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

// The line of a piece or an item that folds or unfolds kept where it is in
// its pane, under the pointer (else a piece above it that folds moves it
// up, and a pane that gets shorter scrolls back): for a moment, the pane
// follows the line, with room under the pieces if it needs it, which goes
// as the pane is scrolled back up
function useKeptInPlace() {
  const room = useRef<HTMLDivElement>(null);
  const shrinking = useRef<(() => void) | null>(null);
  const keep = useCallback((line: HTMLElement) => {
    const pane = scrollParent(line);
    const spacer = room.current;
    if (!pane || !spacer) return;
    const top = line.getBoundingClientRect().top;
    // (longer than what folds takes to be laid out again)
    const end = performance.now() + 400;
    // (the room the pane needs under the pieces to be scrolled so far: what
    // they lack of it; none at its top)
    const roomFor = (scroll: number) => (scroll > 0 ? Math.max(0, scroll + pane.clientHeight - (pane.scrollHeight - spacer.offsetHeight)) : 0);
    const follow = () => {
      const scroll = pane.scrollTop + line.getBoundingClientRect().top - top;
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

// The galaxy as a whole: its distance (its kind a menu, at its title; of
// a parameter, its row, the length of an arcsec there under its title)
// and its redshift
function GalaxyBlock({ ctx }: { ctx: Ctx }) {
  const { defs, options, params, set } = ctx;
  const distance = isObject(options.distance) ? options.distance : null;
  const distanceParam = params.find((p) => p.owner.length === 1 && p.owner[0] === 'distance');
  const distanceType = String(distance?.type ?? '');
  const z = typeof options.redshift === 'number' ? options.redshift : undefined;
  const kind = (
    <DropMenu
      label="The distance of the galaxy"
      width={300}
      entries={typeEntries(defs, 'distance', distanceType, (type) => set(['distance'], type === '' ? undefined : retyped(defs, 'distance', distance, type)), 'None')}
      target={
        <button type="button" className="gm-title-menu" title="The distance of the galaxy: click to change its kind">
          Distance
        </button>
      }
    />
  );
  return (
    <section className="sh-block">
      <h3 className="sh-head">Galaxy</h3>
      <div className="sh-rows">
        {distance && distanceParam ? (
          <ParamLine
            ctx={ctx}
            param={distanceParam}
            rings={null}
            title={kind}
            note={ctx.kpcPerArcsec === null ? undefined : `1″ = ${significant(ctx.kpcPerArcsec * 1000, 3)} pc`}
            open={ctx.row === distanceParam.name}
            onOpen={() => ctx.onRow(distanceParam.name)}
          />
        ) : (
          <Row title={kind} sub={distance ? titled(defs, 'distance', distanceType).label.toLowerCase() : 'none: angles only'} />
        )}
        <Row
          title="Redshift"
          titleProps={{ title: 'The redshift of the galaxy: of its lines, and the dimming of its light' }}
          box={<NumberInput value={z} label="The redshift of the galaxy" placeholder="0" valid={(v) => Number.isFinite(v) && v >= 0} onChange={(v) => set(['redshift'], v > 0 ? v : undefined)} onClear={() => set(['redshift'], undefined)} />}
        />
      </div>
    </section>
  );
}

// The gravity of the galaxy, in the head of its mass (how its mass moves
// what it holds): its kind, a menu (faint, of a galaxy of no mass yet)
function Gravity({ ctx, hasMass }: { ctx: Ctx; hasMass: boolean }) {
  const { defs, options, set } = ctx;
  const gravity = isObject(options.gravity) ? options.gravity : null;
  const gravityType = String(gravity?.type ?? 'newton');
  return (
    <DropMenu
      label="The gravity of the galaxy"
      width={300}
      entries={typeEntries(defs, 'gravity', gravityType, (type) => set(['gravity'], type === 'newton' ? undefined : retyped(defs, 'gravity', gravity, type)))}
      target={
        <button type="button" className="gm-gravity" data-idle={!hasMass || undefined} title={hasMass ? 'The gravity of the mass' : 'The gravity of the mass (it has none yet)'}>
          {titled(defs, 'gravity', gravityType).label}
        </button>
      }
    />
  );
}

// The parameters and the options of the gravity (Newton's has none), rows
// under the head of the mass
function GravityRows({ ctx }: { ctx: Ctx }) {
  const { defs, options, params } = ctx;
  const gravity = isObject(options.gravity) ? options.gravity : null;
  if (!gravity) return null;
  const keys = Object.keys(propertiesOfType(defs, gravity.type, 'gravity')).filter((key) => key !== 'type');
  return (
    <div className="sh-rows">
      <ParamRows ctx={ctx} owned={ownedBy(params, ['gravity'])} rings={null} />
      <OtherOptions ctx={ctx} owner={gravity} family="gravity" base={['gravity']} keys={keys} />
    </div>
  );
}

// ---------------------------------------------------------------------
// A component

function ComponentPiece({
  ctx,
  index,
  component,
  colour,
  open,
  selected,
  onFold,
  onSelect,
  onRekey,
  onShow,
  isolated,
  onIsolate,
}: {
  ctx: Ctx;
  index: number;
  component: Options;
  colour: string;
  open: boolean;
  selected: boolean;
  onFold: (line: HTMLElement) => void;
  onSelect: (line: HTMLElement) => void;
  // (what the sheet does as a piece is renamed or removed, or copied)
  onRekey: (from: string, to: string | null) => void;
  onShow: (key: string) => void;
  isolated: boolean;
  onIsolate: (name: string | null) => void;
}) {
  const { defs, types, problems, draft, options, params, set } = ctx;
  const base: Path = ['components', index];
  const components = componentsOf(options);
  const name = componentName(component, index);
  const parts = partsOfType(defs, types, component.type);
  const radii = ringRadii(component.rnodes);
  const own = problems.filter((p) => startsWith(p.path, base));

  // (what refers to it by name, e.g. the mass that follows it, with it;
  // its open items, and what the sheet shows of it)
  const rename = (to: string) => {
    if (components.some((c, i) => i !== index && c.name === to)) return;
    const old = component.name;
    ctx.folds.rename(name, to);
    onRekey(name, to);
    if (isolated) onIsolate(to);
    draft.edit(
      (s) => {
        const named = setAt(s.options, [...base, 'name'], to);
        return { ...s, options: typeof old === 'string' ? withReferencesRenamed(defs, named, 'components', old, to) : named };
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
    onShow(copyName);
  };
  const remove = () => {
    if (isolated) onIsolate(null);
    set(base, undefined, { moved: removedFrom(['components'], index) });
    onRekey(name, null);
  };
  return (
    <div className="sh-piece" data-open={open || undefined} data-selected={selected || undefined} data-bad={own.length > 0 || undefined} data-at={`components/${index}`} style={{ ['--piece' as string]: `var(--${colour})` }}>
      <PieceHead
        kind="component"
        name={name}
        type={titled(defs, 'component', String(component.type)).label}
        summary={glanceOf(ctx, component, parts)}
        badge={extentOf(component, radii)}
        open={open}
        selected={selected}
        onFold={onFold}
        onSelect={onSelect}
        onRename={rename}
        menu={[
          { label: isolated ? 'Show all in the views' : 'Show it alone in the views', onClick: () => onIsolate(isolated ? null : name) },
          { label: 'Duplicate', onClick: duplicate },
          { label: 'Remove', onClick: remove },
        ]}
        menuTitle="Rename, show alone, duplicate, remove"
      />
      {open && (
        <div className="sh-piece-body">
          <ComponentBody ctx={ctx} index={index} component={component} parts={parts} rings={radii.length ? radii : null} />
        </div>
      )}
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
      <span key={part.option}>
        {words.length > 0 && ' · '}
        {/* (the light says what it is: e.g. "emission lines") */}
        {part.family !== 'emission' && <span className="gm-pw">{part.title.toLowerCase()} </span>}
        {text}
      </span>,
    );
  }
  return words;
}

// ---------------------------------------------------------------------
// An unfolded component: sections in the order of its configuration (its
// geometry, its traits, its light, its physics), each under its caption,
// of items (a part, with its keys in the configuration and what adds to
// it, its rows under it) and rows of single settings

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

// The keys of a part in the configuration, as the line of its item shows
// them unfolded; none if they only say its title again (e.g. 'opacity',
// or 'equilibrium' of 'Equilibrium dispersion')
function keysOf(title: string, keys: string[]): string | null {
  const plain = (text: string) => ` ${text.toLowerCase().replace(/_/g, ' ')} `;
  return keys.length === 1 && plain(title).includes(plain(keys[0])) ? null : keys.join(' · ');
}

// What a setting is, for its tooltip: its description, and its key in the
// configuration
function about(description: unknown, key: string): string {
  return typeof description === 'string' ? `${description} (${key})` : key;
}

// An item of a piece: its line (its chevron and title; folded, what it
// has in a few words, unfolded, its keys; how many of its numbers a fit
// frees), which folds or unfolds it; unfolded, its rows under it, and at
// their foot what adds to it; its problems under it
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
    <div className="gm-item" data-open={fold.open || undefined} data-bad={problems.length > 0 || undefined}>
      <button type="button" className="gm-item-head" aria-expanded={fold.open} title={`${fold.open ? 'Fold' : 'Unfold'} it (Alt: every item)`} onClick={(event) => fold.toggle(event.currentTarget, event.altKey)}>
        <svg className="sh-chev" viewBox="0 0 16 16" aria-hidden="true">
          <path d="M6 4l4 4-4 4" />
        </svg>
        <b title={description}>{title}</b>
        <span className="gm-item-sum">{fold.open ? keys : summary}</span>
        <span className="gm-free">{free > 0 && `${free} free`}</span>
      </button>
      {fold.open && (
        <div className="gm-item-body">
          {children}
          {add && <div className="gm-adds">{add}</div>}
        </div>
      )}
      <Problems list={problems} />
    </div>
  );
}

// The problems of a part or an item, under it
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

// Parameters of a part, a row each (see ParamLine)
function ParamRows({ ctx, owned, rings, limit }: { ctx: Ctx; owned: Param[]; rings: number[] | null; limit?: number }) {
  return (
    <>
      {owned.map((param) => (
        <ParamLine key={param.name} ctx={ctx} param={param} rings={rings} limit={limit} open={ctx.row === param.name} onOpen={() => ctx.onRow(param.name)} />
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
  const foldOf = foldsOf(ctx, componentName(component, index));

  const groups: Record<Section, ReactNode[]> = { geometry: [], traits: [], light: [], physics: [] };
  const feet: Record<Section, ReactNode[]> = { geometry: [], traits: [], light: [], physics: [] };
  const nodesPart = parts.find((p) => p.def === 'Nodes');
  if (nodesPart) groups.geometry.push(<RingsGroup key="rings" ctx={ctx} component={component} base={base} option={nodesPart.option} title={nodesPart.title} keys={ringLines} fold={foldOf(nodesPart.option)} />);
  for (const key of extent) {
    groups.geometry.push(<OptionLine key={key} ctx={ctx} owner={component} base={base} option={key} note={typeof component[key] === 'number' && ctx.kpcPerArcsec !== null && `${significant(component[key] * ctx.kpcPerArcsec, 3)} kpc`} />);
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
        <ParamRows ctx={ctx} owned={motion} rings={rings} />
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
    const shown = () => ctx.folds.set(componentName(component, index), [part.option], true);
    if (section === 'traits') {
      if (has(part) || !part.optional) groups.traits.push(<TraitGroup key={part.option} ctx={ctx} part={part} base={base} rings={rings} problems={own} fold={foldOf(part.option)} />);
      else {
        feet.traits.push(
          <AddPart key={part.option} ctx={ctx} part={part} base={base} onAdded={shown}>
            + {title.toLowerCase()}
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
          {!has(part) && <Row title={title} sub="none" />}
        </Group>,
      );
    } else groups[section].push(<SingleLine key={part.option} ctx={ctx} part={part} path={path} rings={rings} problems={own} />);
  }
  if (shining.length > 0) groups.light.unshift(<ParamRows key="own" ctx={ctx} owned={shining} rings={rings} />);
  if (settings.includes('fluid')) groups.physics.push(<FluidLine key="fluid" ctx={ctx} component={component} base={base} />);
  if (others.length > 0) {
    groups.geometry.push(
      <Group key="others" title="Options" keys={others.join(' · ')} fold={foldOf('others')} summary={others.filter((key) => component[key] !== undefined).map((key) => optionText(component, key)).join(' · ') || 'as by default'}>
        <OtherOptions ctx={ctx} owner={component} base={base} keys={others} />
      </Group>,
    );
  }

  return (
    <>
      {general.length > 0 && (
        <div className="gm-why-top">
          <Problems list={general} />
        </div>
      )}
      {SECTIONS.filter((s) => groups[s.key].length > 0 || feet[s.key].length > 0).map((s) => (
        <section key={s.key} className="gm-sec" aria-label={s.title}>
          <div className="gm-cap">{s.title}</div>
          {groups[s.key]}
          {feet[s.key].length > 0 && <div className="gm-adds">{feet[s.key]}</div>}
        </section>
      ))}
    </>
  );
}

// An option of a component in a few words: its key and its value
function optionText(component: Options, key: string): string {
  const value = component[key];
  return `${key} ${isObject(value) || Array.isArray(value) ? JSON.stringify(value) : String(value)}`;
}

// The rings of a component (or its shells, or the nodes of the traits of
// a spheroid): an item of how many, to what radius, which unfolds into
// how they are given, and a row for each of their options. Nodes a
// component can be without (a spheroid's: by default, 0 and its largest
// radius) can be added and removed.
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
          <button type="button" className="gm-plus" onClick={start}>
            + add
          </button>
        ) : undefined
      }
    >
      {given ? (
        <RingsRows ctx={ctx} unit={component.type === 'cone' ? 'shells' : 'rings'} value={value} onChange={(next) => set(path, next)} onRemove={required ? undefined : () => set(path, undefined)} />
      ) : (
        <Row title="Radii" sub={unset} />
      )}
      {keys.map((key) => (
        <OptionLine key={key} ctx={ctx} owner={component} base={base} option={key} />
      ))}
    </Group>
  );
}

// The frame of a component (its own parameters: those that place it, and
// others of its own, e.g. the flux of a point): its centre and its
// orientation, each under a subtitle with the switch of a value per ring
// if it has one, and its others after them
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
            <div className="gm-sub">
              {line.title}
              {switches.includes(line.option) && <RingSwitch ctx={ctx} component={component} base={base} option={line.option} />}
            </div>
            <ParamRows ctx={ctx} owned={line.params} rings={rings} />
          </Fragment>
        ))}
      <ParamRows ctx={ctx} owned={rest} rings={rings} />
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
    <button type="button" className="gm-rs" aria-pressed={on} title={`${title} (${option})`} onClick={() => ctx.set([...base, option], !on === fallback ? undefined : !on, { moved: inPlace })}>
      <RingMark />
      per ring
    </button>
  );
}

// Options of an item (e.g. of a component, of the gravity) that no part
// takes (e.g. the file of particles): a row each those that can be, the
// others as a form
function OtherOptions({ ctx, owner, family = 'component', base, keys }: { ctx: Ctx; owner: Options; family?: string; base: Path; keys: string[] }) {
  const { defs, set, project } = ctx;
  const properties = propertiesOfType(defs, owner.type, family);
  const simple = keys.filter((key) => lineable(properties[key] ?? {}));
  const rest = keys.filter((key) => !simple.includes(key));
  const node = rest.length > 0 ? restOf(defs, memberOf(defs, family, owner.type), new Set(Object.keys(properties).filter((key) => !rest.includes(key)))) : null;
  return (
    <>
      {simple.map((key) => (
        <OptionLine key={key} ctx={ctx} owner={owner} family={family} base={base} option={key} />
      ))}
      {node && defs && (
        <div className="gm-form">
          <SchemaForm
            defs={defs}
            node={node}
            project={project}
            value={Object.fromEntries(rest.filter((key) => owner[key] !== undefined).map((key) => [key, owner[key]]))}
            onChange={(next) => set(base, { ...omitKeys(owner, new Set(rest)), ...next }, { moved: inPlace })}
          />
        </div>
      )}
    </>
  );
}

// Whether an option can be set in a row of its own (see OptionLine): a
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
  const title = about(part.description, part.option);
  const set = (value: unknown) => {
    ctx.set(path, value);
    onAdded?.();
  };
  if (part.family === null) {
    const start = part.def && defs?.[part.def] ? startOfDef(defs[part.def]) : {};
    return (
      <button type="button" className="gm-plus" title={title} onClick={() => set(start)}>
        {label}
      </button>
    );
  }
  const types = typesOf(defs, part.family, part.types);
  const add = (type: string) => set(withDirection(startOf(defs, part.family!, type), part));
  if (types.length === 1) {
    return (
      <button type="button" className="gm-plus" title={title} onClick={() => add(types[0].value)}>
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
        <button type="button" className="gm-plus" title={title}>
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
// part of the mass), a row of its own: its title, and in its box its value
// as its schema has it: a yes or no, a choice of a few, or a number (in
// its unit; its default, or "auto" if it can be none, when not given);
// under its title, a note given. An option set to its default is left out
// of the configuration.
function OptionLine({ ctx, owner, family = 'component', base, option, note }: { ctx: Ctx; owner: Options; family?: string; base: Path; option: string; note?: ReactNode }) {
  const node = propertiesOfType(ctx.defs, owner.type, family)[option] ?? {};
  const title = typeof node.title === 'string' ? node.title : option;
  const value = owner[option];
  const fallback = node.default;
  const setOption = (next: unknown) => ctx.set([...base, option], next === fallback ? undefined : next, { moved: inPlace });
  const choices = choicesOf(node);
  let box: ReactNode;
  let unit = '';
  if (node.type === 'boolean') {
    const on = (value ?? fallback) === true;
    box = (
      <ChoiceBox
        label={title}
        text={on ? 'on' : 'off'}
        entries={[
          { label: 'Off', current: !on, onClick: () => setOption(false) },
          { label: 'On', current: on, onClick: () => setOption(true) },
        ]}
      />
    );
  } else if (choices) {
    const current = String(value ?? fallback ?? '');
    box = <ChoiceBox label={title} text={choices.find((c) => String(c.value) === current)?.title ?? current} entries={choices.map((c) => ({ label: c.title, current: String(c.value) === current, onClick: () => setOption(c.value) }))} />;
  } else {
    // (a number: in its unit, within its bounds, a whole one if it must
    // be; emptied, none or its default)
    const branches = (node.anyOf as Schema[] | undefined) ?? [node];
    const nullable = branches.some((b) => b.type === 'null');
    const number = branches.find((b) => b.type !== 'null') ?? node;
    const whole = (n: number) => (number.type === 'integer' ? Math.round(n) : n);
    const fits = (n: number) => n >= 0 && (typeof number.exclusiveMinimum === 'number' ? n > number.exclusiveMinimum : typeof number.minimum === 'number' ? n >= number.minimum : true);
    const given = node.unit ?? number.unit;
    unit = unitText(typeof given === 'string' ? given : null);
    box = (
      <NumberInput
        value={typeof value === 'number' ? value : undefined}
        label={title}
        placeholder={nullable ? 'auto' : typeof fallback === 'number' ? String(fallback) : undefined}
        valid={(n) => Number.isFinite(n) && fits(whole(n))}
        onChange={(n) => setOption(whole(n))}
        onClear={nullable || fallback !== undefined ? () => setOption(undefined) : undefined}
      />
    );
  }
  return <Row title={title} titleProps={{ title: about(node.description, option) }} sub={note} box={box} unit={unit} />;
}

// A single setting of a component (e.g. its pressure support, its
// opacity), a row of its own: its title, and in its box its type (or
// none), its other options and its parameters under it; of options of
// their own (e.g. clouds), on or off and those options
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
      <ChoiceBox
        label={part.title}
        text={item ? 'on' : 'off'}
        entries={[
          { label: 'Off', current: item === null, onClick: () => set(path, undefined) },
          { label: 'On', description: part.description, current: item !== null, onClick: () => item === null && set(path, start) },
        ]}
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
      />
    );
    const rest = item ? restOf(defs, memberOf(defs, family, item.type), new Set(['type'])) : null;
    if (rest && defs && item) {
      count = Object.keys(item).filter((key) => key !== 'type').length;
      form = <SchemaForm defs={defs} node={rest} project={project} value={omit(item, 'type')} onChange={(next) => set(path, { type: item.type, ...next })} />;
    }
  }
  return (
    <>
      <Row title={part.title} titleProps={{ title: about(part.description, String(path[path.length - 1])) }} bad={problems.length > 0} box={pick} tools={form ? <MoreButton on={more} count={count} onClick={() => setMore(!more)} /> : undefined} />
      {form && more && <div className="gm-more-form">{form}</div>}
      {item && part.family !== null && <ParamRows ctx={ctx} owned={ownedBy(params, path)} rings={rings} />}
      <Problems list={problems} />
    </>
  );
}

// The gas disks of the mass whose densities the pressure support of a
// disk takes (by default, the disks of the mass that follow it): a row of
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
  const followers = 'The disks that follow it';
  return (
    <>
      <Row
        title={title}
        titleProps={{ title: about(node.description, 'fluid') }}
        bad={own.length > 0}
        box={
          <ChoiceBox
            label={title}
            width={260}
            text={chosen.length > 0 ? chosen.join(', ') : 'auto'}
            title={chosen.length > 0 ? chosen.join(', ') : followers}
            entries={[
              { label: followers, current: chosen.length === 0, onClick: () => choose([]) },
              ...names.map((name) => ({ label: name, current: chosen.includes(name), onClick: () => choose(chosen.includes(name) ? chosen.filter((n) => n !== name) : [...chosen, name]) })),
            ]}
          />
        }
      />
      <Problems list={own} />
    </>
  );
}

// A list of traits (e.g. the velocity of a disk), an item under its title
// and its keys (of its traits and their heights): the rows of each trait,
// with the height paired with it (or one that they share, after them),
// and at their foot what adds one; folded, their types, directions and
// heights
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
          {types.length === 1 ? (
            <button type="button" className="gm-plus" title={types[0].description} onClick={() => add(types[0].value)}>
              + add
            </button>
          ) : (
            <DropMenu
              label={`Add to the ${part.title.toLowerCase()}`}
              width={300}
              entries={types.map((t) => ({ label: t.label, description: t.description, onClick: () => add(t.value) }))}
              target={
                <button type="button" className="gm-plus">
                  + add
                </button>
              }
            />
          )}
          {heightActions &&
            (noHeights ? (
              <button type="button" className="gm-plus" title={heights.description} onClick={() => ctx.set(hpath, newHeight())}>
                + height
              </button>
            ) : (
              <button type="button" className="gm-plus" onClick={() => share(!shared)}>
                {shared ? 'a height for each' : 'one height for all'}
              </button>
            ))}
        </>
      }
    >
      {items.map((item, i) => (
        <div key={i} className="gm-trait">
          <ItemLine ctx={ctx} part={part} path={[...listPath, i]} item={item} rings={rings} onRemove={() => remove(i)} />
          {shown(i) && <HeightLine ctx={ctx} part={heights!} path={[...hpath!, i]} count={items.length} rings={rings} />}
        </div>
      ))}
      {heights && shared && (
        <div className="gm-trait">
          <HeightLine ctx={ctx} part={heights} path={[...hpath!, 0]} count={1} rings={rings} all />
        </div>
      )}
      {items.length === 0 && <Row title={part.title} sub="none" />}
    </Group>
  );
}

// One item of a family (e.g. the light of a component)
function OneItem({ ctx, part, path, rings, onRemove }: { ctx: Ctx; part: PartInfo; path: Path; rings: number[] | null; onRemove?: () => void }) {
  const item = getAt(ctx.options, path);
  return (
    <div className="gm-trait">
      <ItemLine ctx={ctx} part={part} path={path} item={isObject(item) ? item : {}} rings={rings} onRemove={onRemove} />
    </div>
  );
}

// The title of the row of the type of an item: of a trait, its profile;
// of another (e.g. a selection, an emission), its type
function typeTitle(family: string): string {
  return /^[rvdz]pt$/.test(family) ? 'Profile' : 'Type';
}

// An item's rows: its type, its direction, the lines of an emission of
// lines, a row each; its other options behind "options", and what removes
// it, as it is pointed at; its parameters a row each under them
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
      <Row
        title={typeTitle(family)}
        box={<TypePick items={typesOf(defs, family, part.types)} value={String(item.type ?? '')} text={String(item.type ?? '')} label={`The type of the ${part.title.toLowerCase()}`} onChange={(type) => set(path, retyped(defs, family, item, type))} />}
        tools={
          <>
            {rest && <MoreButton on={more} count={given} onClick={() => setMore(!more)} />}
            {onRemove && <Close label="Remove it" onClick={onRemove} />}
          </>
        }
      />
      {directions && directions.length > 0 && (
        <Row
          title="Direction"
          box={
            <ChoiceBox
              label="Its direction"
              text={
                <>
                  <DirectionMark direction={direction} />
                  {direction}
                </>
              }
              title={directions.find((d) => String(d.value) === direction)?.title ?? direction}
              entries={directions.map((d) => ({ label: d.title, current: String(d.value) === direction, onClick: () => set(path, { ...item, direction: d.value }) }))}
            />
          }
        />
      )}
      {lines && <Row title="Lines" wide box={<LinesLine emission={item} onChange={(next) => set(path, next)} />} />}
      {rest && defs && more && (
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
      <ParamRows ctx={ctx} owned={ownedBy(params, path)} rings={rings} />
    </>
  );
}

// The height trait paired with a trait (or the one they share): its type,
// or none if it can be, and its parameters a row each under it
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
      <Row
        title="Height profile"
        sub={all && 'of them all'}
        titleProps={{ title: part.title }}
        box={
          <TypePick
            items={typesOf(defs, family, part.types)}
            value={String(item?.type ?? '')}
            text={item ? String(item.type) : 'none'}
            none={part.nullable ? 'The same at every height' : undefined}
            label={part.title}
            onChange={(type) => set(type === '' ? null : retyped(defs, family, item, type))}
          />
        }
        tools={rest ? <MoreButton on={more} count={Object.keys(item ?? {}).filter((key) => key !== 'type').length} onClick={() => setMore(!more)} /> : undefined}
      />
      {rest && defs && item && more && (
        <div className="gm-more-form">
          <SchemaForm defs={defs} node={rest} project={project} value={omit(item, 'type')} onChange={(next) => set({ type: item.type, ...next })} />
        </div>
      )}
      {item && <ParamRows ctx={ctx} owned={ownedBy(params, path)} rings={rings} limit={typeof item.zmax === 'number' ? item.zmax : undefined} />}
    </>
  );
}

// Options of their own (e.g. the clouds of a disk), as a form
function ObjectItem({ ctx, part, path, onRemove }: { ctx: Ctx; part: PartInfo; path: Path; onRemove?: () => void }) {
  const { defs, options, set, project } = ctx;
  const value = getAt(options, path);
  if (!defs || !part.def || !defs[part.def]) return null;
  return (
    <div className="gm-form gm-object">
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

// A part of the mass: a piece as a component is (its name and type, and
// folded, its parts, what it follows and its numbers in a few words);
// unfolded, a row each of what it follows and of its parts of a type
// (e.g. the profile of a halo), and its parameters and its options, each
// an item that folds
function MassPiece({
  ctx,
  index,
  item,
  colour,
  components,
  open,
  selected,
  onFold,
  onSelect,
  onRekey,
}: {
  ctx: Ctx;
  index: number;
  item: Options;
  // Its colour (a token: of the component it follows, if it does)
  colour: string;
  components: Options[];
  open: boolean;
  selected: boolean;
  onFold: (line: HTMLElement) => void;
  onSelect: (line: HTMLElement) => void;
  onRekey: (from: string, to: string | null) => void;
}) {
  const { defs, options, params, problems, set } = ctx;
  const path: Path = ['mass', index];
  const refs = referencesOf(defs, 'mass_component', item.type);
  const follows = Object.entries(refs).filter(([, what]) => what === 'components').map(([key]) => key);
  const followed = follows.map((key) => item[key]).find((name) => typeof name === 'string');
  const own = problems.filter((p) => startsWith(p.path, path));
  // (its options, each where it goes: the components it follows; its parts
  // of a type, e.g. the profile of a halo, a row each as a single setting
  // of a component; the others with its options)
  const properties = defs ? propertiesOf(memberOf(defs, 'mass_component', item.type), defs) : { properties: {} as Record<string, Schema>, required: [] as string[] };
  const keys = Object.keys(properties.properties).filter((key) => !HEAD_OPTIONS.has(key) && !follows.includes(key));
  const familyOf = (key: string) => familyOfOption(defs, 'mass_component', item.type, key);
  const typed = keys.filter((key) => {
    const family = familyOf(key);
    return family !== null && typesOf(defs, family).length > 0;
  });
  const others = keys.filter((key) => !typed.includes(key));
  // (its own parameters: not those of its parts of a type, under them)
  const owned = ownedBy(params, path).filter((p) => !typed.some((key) => startsWith(p.owner, [...path, key])));
  const partOf = (key: string): PartInfo => ({
    option: key,
    title: typeof properties.properties[key].title === 'string' ? properties.properties[key].title : key,
    description: typeof properties.properties[key].description === 'string' ? properties.properties[key].description : undefined,
    family: familyOf(key),
    def: null,
    many: false,
    optional: !properties.required.includes(key),
    nullable: false,
    types: null,
    directions: null,
    heights: null,
  });
  const within = (key: string) => own.filter((p) => startsWith(p.path, [...path, key]));
  const general = own.filter((p) => !typed.some((key) => startsWith(p.path, [...path, key])));
  const name = String(item.name ?? '');
  const key = massKey(name);
  // (in a few words: the types of what it is made of, e.g. the profile of a
  // halo; what it follows; its numbers)
  const inner = Object.entries(item)
    .filter(([option, value]) => option !== 'type' && isObject(value) && typeof value.type === 'string')
    .map(([option, value]) => titled(defs, familyOfOption(defs, 'mass_component', item.type, option) ?? '', String((value as Options).type)).label);
  const words = [...inner, followed ? `follows ${String(followed)}` : null, ...owned.slice(0, 3).map((p) => paramText(ctx, p))].filter(Boolean);
  // (the components it can follow; the one it follows, if not of them)
  const choices = followable(components, item.type).map((c) => componentName(c, components.indexOf(c)));
  const foldOf = foldsOf(ctx, key);
  // (what refers to it by name, e.g. a halo that responds to it, with it;
  // its open items, and what the sheet shows of it)
  const rename = (to: string) => {
    if (asList(options.mass).some((m, i) => i !== index && m.name === to)) return;
    const old = item.name;
    ctx.folds.rename(key, massKey(to));
    onRekey(key, massKey(to));
    ctx.draft.edit(
      (s) => {
        const named = setAt(s.options, [...path, 'name'], to);
        return { ...s, options: typeof old === 'string' ? withReferencesRenamed(defs, named, 'mass', old, to) : named };
      },
      { moved: inPlace },
    );
  };
  const remove = () => {
    set(path, undefined, { moved: removedFrom(['mass'], index) });
    onRekey(key, null);
  };
  return (
    <div className="sh-piece" data-open={open || undefined} data-selected={selected || undefined} data-bad={own.length > 0 || undefined} data-at={`mass/${index}`} style={{ ['--piece' as string]: `var(--${colour})` }}>
      <PieceHead
        kind="part of the mass"
        name={name}
        type={titled(defs, 'mass_component', String(item.type)).label}
        summary={words.join(' · ')}
        open={open}
        selected={selected}
        onFold={onFold}
        onSelect={onSelect}
        onRename={rename}
        menu={[{ label: 'Remove', onClick: remove }]}
        menuTitle="Rename, remove"
      />
      {open && (
        <div className="sh-piece-body">
          {follows.map((option) => (
            <Row
              key={option}
              title="Follows"
              titleProps={{ title: option }}
              sub={choices.length === 0 && typeof item[option] !== 'string' && 'no component of its kind to follow'}
              box={
                (choices.length > 0 || typeof item[option] === 'string') && (
                  <TypePick
                    items={[...(typeof item[option] === 'string' && !choices.includes(item[option]) ? [item[option]] : []), ...choices].map((choice) => ({ value: choice, label: choice }))}
                    value={String(item[option] ?? '')}
                    text={typeof item[option] === 'string' ? item[option] : undefined}
                    label="The component it follows"
                    onChange={(choice) => set([...path, option], choice)}
                  />
                )
              }
            />
          ))}
          {typed.map((option) => (
            <SingleLine key={option} ctx={ctx} part={partOf(option)} path={[...path, option]} rings={null} problems={within(option)} />
          ))}
          {owned.length > 0 && (
            <Group title="Parameters" fold={foldOf('parameters')} summary={owned.map((p) => paramText(ctx, p)).join(' · ')} free={freeCount(ctx, owned)}>
              <ParamRows ctx={ctx} owned={owned} rings={null} />
            </Group>
          )}
          {others.length > 0 && (
            <Group title="Options" keys={others.join(' · ')} fold={foldOf('options')} summary={others.filter((option) => item[option] !== undefined).map((option) => optionText(item, option)).join(' · ') || 'as by default'}>
              <OtherOptions ctx={ctx} owner={item} family="mass_component" base={path} keys={others} />
            </Group>
          )}
          {general.length > 0 && (
            <div className="gm-why-top">
              <Problems list={general} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------
// Small parts

// A choice, in the box of a row: what is chosen, and a menu of the
// choices
function ChoiceBox({ label, text, title, entries, width = 240 }: { label: string; text: ReactNode; title?: string; entries: MenuEntry[]; width?: number }) {
  return (
    <DropMenu
      label={label}
      width={width}
      entries={entries}
      target={
        <button type="button" className="sh-num gm-choice" title={title ?? (typeof text === 'string' ? text : undefined)}>
          <span>{text}</span>
        </button>
      }
    />
  );
}

// A type of a family to choose (e.g. of a velocity trait), in the box of a
// row: its title (or the text given, e.g. its type), a menu of the types,
// each with what it is; none, if it can be
function TypePick({ items, value, onChange, label, none, text }: { items: TypeChoice[]; value: string; onChange: (type: string) => void; label: string; none?: string; text?: string }) {
  const current = items.find((t) => t.value === value);
  return (
    <ChoiceBox
      label={label}
      width={300}
      text={text ?? current?.label ?? (value === '' ? (none ?? 'Choose…') : value)}
      title={current?.label}
      entries={[
        ...(none ? [{ label: none, current: value === '', onClick: () => onChange('') }] : []),
        ...items.map((t) => ({ label: t.label, description: t.description, current: t.value === value, onClick: () => onChange(t.value) })),
      ]}
    />
  );
}

// What opens the other options of an item (how many are given)
function MoreButton({ on, count = 0, onClick }: { on: boolean; count?: number; onClick: () => void }) {
  return (
    <button type="button" className="gm-opt" data-on={on || undefined} onClick={onClick} aria-expanded={on} title="Its other options">
      {count > 0 ? `options · ${count}` : 'options'}
    </button>
  );
}

type RingsForm = 'sep' | 'len' | 'list';

// The radii of the rings (or shells): {max, sep}, {max, len} or a list of
// them; a row of how they are given (how many, and how far they reach,
// under it; a change of form keeps the reach of the rings and about their
// number), and of their largest radius and their separation or number,
// or the list of them; what removes them, if they can be; and
// where they are, marks along a line
function RingsRows({ ctx, unit, value, onChange, onRemove }: { ctx: Ctx; unit: string; value: unknown; onChange: (value: unknown) => void; onRemove?: () => void }) {
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
  const positive = (v: number) => Number.isFinite(v) && v > 0;
  const reach = ctx.kpcPerArcsec !== null && radii.length > 0 ? ` · to ${significant(max * ctx.kpcPerArcsec, 3)} kpc` : '';
  return (
    <div className="gm-rings">
      <Row
        title="Radii"
        sub={`${radii.length} ${unit}${reach}`}
        box={
          <ChoiceBox
            label={`How the ${unit} are given`}
            width={220}
            text={{ sep: 'every', len: 'count', list: 'list' }[form]}
            entries={[
              { label: 'Evenly, every so far', current: form === 'sep', onClick: () => toForm('sep') },
              { label: 'Evenly, so many', current: form === 'len', onClick: () => toForm('len') },
              { label: 'At the radii given', current: form === 'list', onClick: () => toForm('list') },
            ]}
          />
        }
        tools={onRemove && <Close label={`Remove the ${unit}`} onClick={onRemove} />}
      />
      {form === 'list' ? (
        <RadiiText radii={radii} onCommit={onChange} />
      ) : (
        <>
          <Row
            title="Largest radius"
            box={<NumberInput value={numberOr(options.max)} label="The largest radius" valid={positive} onChange={(m) => onChange({ ...options, max: m })} />}
            unit="″"
          />
          {form === 'sep' ? (
            <Row title="Separation" box={<NumberInput value={numberOr(options.sep)} label="The separation" valid={positive} onChange={(sep) => onChange({ ...options, sep })} />} unit="″" />
          ) : (
            <Row title="Number" box={<NumberInput value={numberOr(options.len)} label="The number" valid={(n) => Number.isInteger(n) && n >= 2} onChange={(len) => onChange({ ...options, len })} />} />
          )}
        </>
      )}
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

// The lines of an emission of lines, in its row: by default one line at
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
    <div className="gm-lines">
      {lines.length === 0 ? (
        <span className="gm-note" title="One line, at the velocity of the spectral axis of the data, whatever it is">
          the line of the data
        </span>
      ) : (
        lines.map((line, i) => (
          <span key={i} className="gm-line" title={`${rests.get(lineName(line)) ?? 'A line of the configuration'}${i === 0 ? ' · its flux from the density' : ' · a ratio of flux to the first'}`}>
            {lineName(line)}
            <button type="button" className="gm-x" aria-label={`Remove the line ${lineName(line)}`} onClick={() => set(lines.filter((_, j) => j !== i))}>
              ×
            </button>
          </span>
        ))
      )}
      <DropMenu
        label="Add a line"
        width={260}
        entries={catalogue.filter((l) => !named.includes(l.name)).map((l) => ({ label: l.name, description: l.rest, onClick: () => set([...lines, l.name]) }))}
        target={
          <button type="button" className="gm-plus">
            + line
          </button>
        }
      />
    </div>
  );
}

// A problem as people read it: without the place the library prefixes
// (e.g. '[pieces.0] ') nor the names of the errors of Python (e.g.
// 'ValueError: ')
export function problemText(message: string): string {
  return message.replace(/^\[[^\]]*\] /, '').replace(/\b[A-Za-z]+Error: /g, '');
}

function numberOr(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined;
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
