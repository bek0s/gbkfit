// The parameters of a galaxy as one list, which the keyboard moves
// through: each a row (see ParamLine: its title and name, its value, its
// unit, how a fit takes it; opened, its slider or what sets it), grouped
// by what owns it (a part of a component, the mass, the galaxy). A search
// finds them by title or name, and filters keep those free, changed since
// the save, tied, or with a value per ring.
//
// Keys, the list focused: Up and Down move; a digit (or a minus) starts
// typing the value, Enter sets it; F frees or fixes it; T ties it to an
// expression; Enter opens it (and closes it); Escape closes it, or leaves
// what is being typed; / finds.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { Defs } from '../../components/SchemaForm';
import type { GalaxyDraft } from './draft';
import { ParamLine, type NumbersContext } from './Numbers';
import { componentName, componentsOf, getAt, ringRadii, type Options } from './options';
import { naturalSpan, settingOf, withFreedom, type Param } from './params';
import { partsOfType, type ComponentTypes } from './types';

type Filter = 'all' | 'free' | 'changed' | 'tied' | 'ring';

const FILTERS: { value: Filter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'free', label: 'Free' },
  { value: 'changed', label: 'Changed' },
  { value: 'tied', label: 'Tied' },
  { value: 'ring', label: 'Per ring' },
];

interface Group {
  key: string;
  label: string;
  colour: string;
  params: Param[];
}

export interface ParamsProps {
  draft: GalaxyDraft;
  field: number;
  colours: string[];
  types: ComponentTypes | undefined;
  defs: Defs | undefined;
}

// The group of a parameter: its component and the part of it (its frame,
// for those of the component itself), the mass, or the galaxy
function groupOf(param: Param, options: Options, colours: string[], titleOf: (component: Options, option: string) => string): { key: string; label: string; colour: string } {
  const [section, index, option] = param.owner;
  if (section === 'components' && typeof index === 'number') {
    const component = componentsOf(options)[index] ?? {};
    const name = componentName(component, index);
    const part = typeof option === 'string' ? titleOf(component, option).toLowerCase() : 'frame';
    return { key: `c${index}/${String(option ?? '')}`, label: `${name} · ${part}`, colour: `var(--${colours[index] ?? 'faint'})` };
  }
  if (section === 'mass') return { key: 'mass', label: 'mass', colour: 'var(--faint)' };
  return { key: 'galaxy', label: 'galaxy', colour: 'var(--faint)' };
}

export function Params({ draft, field, colours, types, defs }: ParamsProps) {
  const options = draft.state?.options ?? {};
  const properties = draft.state?.properties ?? {};
  const saved = draft.saved?.properties ?? {};
  const params = draft.params;
  const ctx: NumbersContext = { draft, options, params, field };
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<Filter>('all');
  const [selected, setSelected] = useState<string | null>(null);
  // (the row open, and the tie of a parameter as it is typed)
  const [opened, setOpened] = useState<string | null>(null);
  const [tie, setTie] = useState<{ name: string; text: string } | null>(null);
  const list = useRef<HTMLDivElement>(null);
  const find = useRef<HTMLInputElement>(null);

  const titleOf = (component: Options, option: string) => partsOfType(defs, types, component.type).flatMap((p) => [p, ...(p.heights ? [p.heights] : [])]).find((p) => p.option === option)?.title ?? option;
  const settings = useMemo(() => new Map(params.map((p) => [p.name, settingOf(properties, p)])), [params, properties]);
  const matches = (p: Param, f: Filter) => {
    const setting = settings.get(p.name)!;
    if (f === 'free') return setting.kind === 'free';
    if (f === 'tied') return setting.kind === 'tied';
    if (f === 'ring') return p.size > 1;
    if (f === 'changed') return JSON.stringify(properties[p.name]) !== JSON.stringify(saved[p.name]);
    return true;
  };
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const found = params.filter((p) => words.every((w) => `${p.title ?? ''} ${p.name}`.toLowerCase().includes(w)));
  const shown = found.filter((p) => matches(p, filter));
  // (in the order of their first parameters, each with all of its own)
  const byKey = new Map<string, Group>();
  for (const p of shown) {
    const g = groupOf(p, options, colours, titleOf);
    const group = byKey.get(g.key);
    if (group) group.params.push(p);
    else byKey.set(g.key, { ...g, params: [p] });
  }
  const groups = [...byKey.values()];
  const order = groups.flatMap((g) => g.params.map((p) => p.name));
  const current = selected !== null && order.includes(selected) ? selected : (order[0] ?? null);

  // (the row selected in sight)
  useEffect(() => {
    if (current) list.current?.querySelector(`[data-name="${current}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [current]);
  // (/ finds, from anywhere on the page but a field)
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key !== '/' || (event.target instanceof HTMLElement && event.target.closest('input, textarea, [contenteditable]'))) return;
      event.preventDefault();
      find.current?.focus();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const setProperty = (name: string, next: unknown) =>
    draft.edit((s) => {
      const { [name]: _, ...rest } = s.properties;
      return { ...s, properties: next === undefined ? rest : { ...rest, [name]: next } };
    });
  const param = (name: string) => params.find((p) => p.name === name)!;
  const commitTie = () => {
    if (!tie) return;
    const expression = tie.text.trim();
    if (expression) setProperty(tie.name, expression);
    setTie(null);
    list.current?.focus();
  };
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    // (a value typed in a row, set or left: the keys to the list again)
    if (event.target !== list.current) {
      if ((event.key === 'Enter' || event.key === 'Escape') && (event.target as HTMLElement).matches('.gp-list input.sh-num')) requestAnimationFrame(() => list.current?.focus());
      return;
    }
    if (tie || current === null) return;
    const at = order.indexOf(current);
    const p = param(current);
    const setting = settings.get(current)!;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      setSelected(order[Math.max(0, Math.min(order.length - 1, at + (event.key === 'ArrowDown' ? 1 : -1)))]);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      setOpened(opened === current ? null : current);
    } else if (event.key === 'f' || event.key === 'F') {
      event.preventDefault();
      if (setting.kind === 'fixed' || setting.kind === 'free') setProperty(p.name, withFreedom(properties[p.name], p, setting, setting.kind !== 'free', naturalSpan(p, setting.values, field)));
    } else if (event.key === 't' || event.key === 'T') {
      event.preventDefault();
      setTie({ name: current, text: setting.kind === 'tied' ? setting.expression : '' });
    } else if (/^[0-9.-]$/.test(event.key)) {
      // (into the box of the row, the key typed over its value)
      list.current?.querySelector<HTMLInputElement>(`[data-name="${current}"] input.sh-num`)?.focus();
    } else if (event.key === 'Escape') setOpened(null);
  };

  const count = (f: Filter) => found.filter((p) => matches(p, f)).length;
  return (
    <div className="gp">
      <label className="gp-find">
        <input ref={find} value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="Find a parameter" aria-label="Find a parameter" onKeyDown={(event) => event.key === 'Escape' && (setQuery(''), list.current?.focus())} />
        <span className="gp-kbd">/</span>
      </label>
      <div className="gp-filters" role="group" aria-label="Which parameters">
        {FILTERS.map((f) => (
          <button key={f.value} type="button" aria-pressed={filter === f.value} onClick={() => setFilter(f.value)}>
            {f.label}
            <em>{count(f.value)}</em>
          </button>
        ))}
      </div>
      <div
        className="gp-list"
        ref={list}
        tabIndex={0}
        onKeyDown={onKey}
        // (a click on a row, but on what takes the keys itself: the keys to
        // the list; not one in what opens from a row, out of it)
        onClick={(event) => {
          const target = event.target as HTMLElement;
          if (list.current?.contains(target) && !target.closest('input, button, textarea')) list.current.focus();
        }}
        aria-label="The parameters"
        aria-activedescendant={current ? `gp-${current}` : undefined}
      >
        {groups.length === 0 && <p className="sh-note">{params.length ? 'None matches.' : 'No parameters: the library has not described the galaxy.'}</p>}
        {groups.map((g) => (
          <div key={g.key} className="gp-group">
            <div className="gp-ghead">
              <i style={{ background: g.colour }} />
              {g.label}
            </div>
            {g.params.map((p) => {
              const rings = p.owner[0] === 'components' && typeof p.owner[1] === 'number' ? ringRadii(getAt(options, ['components', p.owner[1], 'rnodes'])) : [];
              return (
                <div key={p.name} id={`gp-${p.name}`} data-name={p.name}>
                  <ParamLine
                    ctx={ctx}
                    param={p}
                    rings={rings.length ? rings : null}
                    open={opened === p.name}
                    onOpen={() => {
                      setSelected(p.name);
                      setOpened((o) => (o === p.name ? null : p.name));
                    }}
                    named
                    current={current === p.name}
                    box={
                      tie?.name === p.name ? (
                        <input
                          className="sh-num gp-tie"
                          autoFocus
                          value={tie.text}
                          placeholder="e.g. gas_incl"
                          onChange={(event) => setTie({ ...tie, text: event.currentTarget.value })}
                          onBlur={commitTie}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') commitTie();
                            if (event.key === 'Escape') {
                              setTie(null);
                              list.current?.focus();
                            }
                          }}
                          aria-label="Its tie"
                        />
                      ) : undefined
                    }
                  />
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="gp-foot">
        <span>
          <span className="gp-kbd">↑</span>
          <span className="gp-kbd">↓</span>move
        </span>
        <span>
          <span className="gp-kbd">0–9</span>type the value
        </span>
        <span>
          <span className="gp-kbd">F</span>free / fixed
        </span>
        <span>
          <span className="gp-kbd">T</span>tie
        </span>
        <span>
          <span className="gp-kbd">Enter</span>open
        </span>
      </div>
    </div>
  );
}
