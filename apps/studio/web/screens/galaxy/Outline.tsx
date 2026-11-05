// The outline of a galaxy: the galaxy as a whole (its distance, mass and
// gravity, those it has; the others to add), and its components, each in
// its colour, with its parts (each at a glance: e.g. the types of its
// traits) and the parts it can have too. A part chosen is set in the
// inspector; a part with a problem is marked. A component is renamed in
// place (a double click), and its menu duplicates, shows alone or removes
// it.
import { useState } from 'react';
import type { Defs } from '../../components/SchemaForm';
import { DropMenu } from '../../components/ui/overlays';
import { COMPONENT_ORDER, COMPONENT_STARTS, GALAXY_PARTS, partsOf, typeLabel, type Part } from './catalogue';
import type { Problem } from './draft';
import { asList, componentName, getAt, isObject, lineList, lineName, ringRadii, startsWith, type Options, type Path } from './options';
import { typesOf } from './schema';

export interface Selected {
  component: number | null;
  part: string;
}

export interface OutlineProps {
  options: Options;
  defs: Defs | undefined;
  problems: Problem[];
  colours: string[];
  selected: Selected | null;
  isolated: string | null;
  onSelect: (selected: Selected) => void;
  onAddPart: (component: number | null, part: Part) => void;
  onAddComponent: (type: string) => void;
  onRenameComponent: (index: number, name: string) => void;
  onDuplicateComponent: (index: number) => void;
  onRemoveComponent: (index: number) => void;
  onIsolate: (name: string | null) => void;
}

export function Outline(props: OutlineProps) {
  const { options, defs, problems, colours, selected, onSelect, onAddPart, onAddComponent } = props;
  const components = asList(options.components);
  const isOn = (component: number | null, part: string) => selected?.component === component && selected.part === part;
  const present = GALAXY_PARTS.filter((p) => getAt(options, p.option) !== undefined);
  const missing = GALAXY_PARTS.filter((p) => getAt(options, p.option) === undefined);
  const flagged = (path: Path) => problems.some((p) => startsWith(p.path, path));
  return (
    <nav className="gx-outline" aria-label="What the galaxy is made of">
      <section className="gx-section">
        <header className="gx-section-head">
          <span>Galaxy</span>
          {missing.length > 0 && (
            <DropMenu
              label="Add to the galaxy"
              width={300}
              at="end"
              entries={missing.map((p) => {
                const why = p.needs?.(options, options) ?? null;
                return { label: p.title, description: why ?? p.hint, disabled: why !== null, onClick: () => onAddPart(null, p) };
              })}
              target={
                <button className="gx-plus" aria-label="Add to the galaxy" title="Add its distance, mass or gravity">
                  +
                </button>
              }
            />
          )}
        </header>
        {present.length === 0 && <p className="gx-outline-note">Its distance, mass and gravity, when it needs them.</p>}
        {present.map((p) => (
          <PartRow key={p.key} part={p} glance={glanceOf(p, options)} on={isOn(null, p.key)} problem={flagged(p.option)} onClick={() => onSelect({ component: null, part: p.key })} />
        ))}
      </section>
      <section className="gx-section">
        <header className="gx-section-head">
          <span>Components</span>
          <AddComponent defs={defs} onAdd={onAddComponent} />
        </header>
        {components.length === 0 && <p className="gx-outline-note">None yet: add a disk, a cone, a spheroid or a point.</p>}
        {components.map((c, i) => (
          <ComponentGroup key={i} {...props} component={c} index={i} colour={colours[i]} isOn={isOn} flagged={flagged} />
        ))}
      </section>
    </nav>
  );
}

// The menu that adds a component of a type (the common ones first)
function AddComponent({ defs, onAdd }: { defs: Defs | undefined; onAdd: (type: string) => void }) {
  const rank = (type: string) => (COMPONENT_ORDER.includes(type) ? COMPONENT_ORDER.indexOf(type) : COMPONENT_ORDER.length);
  const types = typesOf(defs, 'component').sort((a, b) => rank(a.value) - rank(b.value));
  return (
    <DropMenu
      label="Add a component"
      width={320}
      at="end"
      entries={types.map(({ value, label, description }) => {
        const ready = COMPONENT_STARTS[value] !== undefined;
        return {
          label,
          description: ready ? description : 'Made of a file of particles: give it in the configuration.',
          disabled: !ready,
          onClick: () => onAdd(value),
        };
      })}
      target={
        <button className="gx-plus" aria-label="Add a component" title="Add a component">
          +
        </button>
      }
    />
  );
}

function ComponentGroup({
  component,
  index,
  colour,
  isOn,
  flagged,
  options,
  isolated,
  onSelect,
  onAddPart,
  onRenameComponent,
  onDuplicateComponent,
  onRemoveComponent,
  onIsolate,
}: OutlineProps & {
  component: Options;
  index: number;
  colour: string;
  isOn: (component: number | null, part: string) => boolean;
  flagged: (path: Path) => boolean;
}) {
  const [renaming, setRenaming] = useState(false);
  const name = componentName(component, index);
  const parts = partsOf(component);
  const present = parts.filter((p) => p.key === 'geometry' || getAt(component, p.option) !== undefined);
  const absent = parts.filter((p) => p.optional && getAt(component, p.option) === undefined);
  const base: Path = ['components', index];
  const alone = isolated === name;
  return (
    <div className="gx-component" data-dim={(isolated !== null && !alone) || undefined}>
      <div className="gx-component-head">
        <i className="gx-dot" style={{ background: `var(--${colour})` }} />
        {renaming ? (
          <RenameField
            value={typeof component.name === 'string' ? component.name : ''}
            onDone={(next) => {
              setRenaming(false);
              if (next !== null && next !== component.name) onRenameComponent(index, next);
            }}
          />
        ) : (
          <span className="gx-component-name" onDoubleClick={() => setRenaming(true)} title="Double-click to rename">
            {name}
          </span>
        )}
        <span className="gx-type">{typeLabel(component.type)}</span>
        <DropMenu
          label={`The actions of ${name}`}
          width={220}
          at="end"
          // (Rename gives the focus to the field of the name)
          returnFocus={false}
          entries={[
            { label: 'Rename', onClick: () => setRenaming(true) },
            { label: 'Duplicate', onClick: () => onDuplicateComponent(index) },
            { label: alone ? 'Show all components' : 'Show it alone', onClick: () => onIsolate(alone ? null : name) },
            ...absent.map((p) => {
              const why = p.needs?.(component, options) ?? null;
              return { label: `Add ${p.title.toLowerCase()}`, description: why ?? undefined, disabled: why !== null, onClick: () => onAddPart(index, p) };
            }),
            { label: 'Remove', onClick: () => onRemoveComponent(index) },
          ]}
          target={
            <button className="gx-more-button" aria-label={`The actions of ${name}`}>
              ⋯
            </button>
          }
        />
      </div>
      {present.map((p) => (
        <PartRow
          key={p.key}
          part={p}
          glance={glanceOf(p, component)}
          on={isOn(index, p.key)}
          problem={p.key === 'geometry' ? false : flagged([...base, ...p.option])}
          onClick={() => onSelect({ component: index, part: p.key })}
        />
      ))}
      {absent.length > 0 && (
        <DropMenu
          label={`Add to ${name}`}
          width={300}
          entries={absent.map((p) => {
            const why = p.needs?.(component, options) ?? null;
            return { label: p.title, description: why ?? p.hint, disabled: why !== null, onClick: () => onAddPart(index, p) };
          })}
          target={<button className="gx-add-part">+ {absent.map((p) => p.title.toLowerCase()).join(', ')}</button>}
        />
      )}
    </div>
  );
}

function PartRow({ part, glance, on, problem, onClick }: { part: Part; glance: string; on: boolean; problem: boolean; onClick: () => void }) {
  return (
    <button className="gx-part" data-on={on || undefined} onClick={onClick} aria-current={on || undefined}>
      <span className="gx-part-title">
        {part.title}
        {problem && <i className="gx-flag" title="It has a problem" />}
      </span>
      <span className="gx-glance">{glance}</span>
    </button>
  );
}

// The name of a component as it is typed (Enter or leaving: set; Escape:
// as it was); a name of letters, digits and underscores
function RenameField({ value, onDone }: { value: string; onDone: (name: string | null) => void }) {
  const [text, setText] = useState(value);
  const valid = /^[A-Za-z_][A-Za-z0-9_]*$/.test(text);
  return (
    <input
      className="gx-rename"
      value={text}
      autoFocus
      data-invalid={!valid || undefined}
      onFocus={(event) => event.currentTarget.select()}
      onChange={(event) => setText(event.currentTarget.value)}
      onBlur={() => onDone(valid ? text : null)}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') onDone(null);
      }}
      aria-label="The name of the component"
      spellCheck={false}
    />
  );
}

// A part at a glance: the types of its items (or of its option), and the
// like (e.g. the rings of a disk, the lines of its light)
export function glanceOf(part: Part, owner: Options): string {
  const value = getAt(owner, part.option);
  if (part.key === 'geometry') {
    if (owner.type === 'spheroid') return `to ${owner.rmax}″`;
    const radii = ringRadii(owner.rnodes);
    if (radii.length === 0) return '';
    const flags = [owner.tilted === true && 'tilted', owner.loose === true && 'loose'].filter(Boolean);
    return `${radii.length} rings · ${Number(radii[radii.length - 1].toPrecision(3))}″${flags.length ? ` · ${flags.join(', ')}` : ''}`;
  }
  if (part.key === 'light' && isObject(value) && value.type === 'lines') {
    const lines = lineList(value.lines).map(lineName);
    return lines.length ? lines.join(', ') : 'a line';
  }
  if (part.key === 'mass') return asList(value).map((m) => String(m.name ?? m.type)).join(', ');
  const items = asList(value);
  return items
    .map((item) => [item.type, typeof item.direction === 'string' ? item.direction : null].filter(Boolean).join(' '))
    .join(', ')
    .replaceAll('_', ' ');
}
