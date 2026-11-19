// The parameters of a galaxy as one list, which the keyboard moves
// through: each its title and name, its value, how a fit takes it (fixed,
// free or tied) and its range (or its values along its rings), grouped by
// what owns it (a part of a component, the mass, the galaxy). A search
// finds them by title or name, and filters keep those free, changed since
// the save, tied, or with a value per ring.
//
// Keys, the list focused: Up and Down move; a digit (or a minus) starts
// typing the value, Enter sets it; F frees or fixes it; T ties it to an
// expression; Enter opens its range (and closes it); Escape leaves what is
// being typed; / finds.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { Defs } from '../../components/SchemaForm';
import type { GalaxyDraft } from './draft';
import { ParamEditor, valueText, type NumbersContext } from './Numbers';
import { componentName, componentsOf, getAt, ringRadii, type Options } from './options';
import { unitText } from './ParamField';
import { freeRange, settingOf, withFreedom, withValues, type Param, type Setting } from './params';
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
  const [typing, setTyping] = useState<{ name: string; kind: 'value' | 'tie'; text: string } | null>(null);
  const [opened, setOpened] = useState<string | null>(null);
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
  const commit = () => {
    if (!typing) return;
    const p = param(typing.name);
    if (typing.kind === 'tie') {
      const expression = typing.text.trim();
      if (expression) setProperty(p.name, expression);
    } else {
      const value = Number(typing.text);
      if (typing.text.trim() !== '' && Number.isFinite(value)) setProperty(p.name, withValues(properties[p.name], p, [value]));
    }
    setTyping(null);
    list.current?.focus();
  };
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (typing || current === null || event.target !== list.current) return;
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
      if (setting.kind === 'fixed' || setting.kind === 'free') setProperty(p.name, withFreedom(properties[p.name], p, setting, setting.kind !== 'free', freeRange(p, field)));
    } else if (event.key === 't' || event.key === 'T') {
      event.preventDefault();
      setTyping({ name: current, kind: 'tie', text: setting.kind === 'tied' ? setting.expression : '' });
    } else if (/^[0-9.-]$/.test(event.key) && p.size === 1) {
      event.preventDefault();
      setTyping({ name: current, kind: 'value', text: event.key });
    } else if (event.key === 'Escape') setOpened(null);
  };

  const count = (f: Filter) => found.filter((p) => matches(p, f)).length;
  return (
    <div className="gp">
      <div className="gp-tools">
        <label className="gp-find">
          <input ref={find} value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="Find a parameter" aria-label="Find a parameter" onKeyDown={(event) => event.key === 'Escape' && (setQuery(''), list.current?.focus())} />
          <span className="gp-kbd">/</span>
        </label>
      </div>
      <div className="gp-filters" role="group" aria-label="Which parameters">
        {FILTERS.map((f) => (
          <button key={f.value} data-on={filter === f.value || undefined} onClick={() => setFilter(f.value)}>
            {f.label}
            <em>{count(f.value)}</em>
          </button>
        ))}
      </div>
      <div className="gp-list" ref={list} tabIndex={0} onKeyDown={onKey} aria-label="The parameters" aria-activedescendant={current ? `gp-${current}` : undefined}>
        {groups.length === 0 && <p className="gm-empty">{params.length ? 'None matches.' : 'No parameters: the library has not described the galaxy.'}</p>}
        {groups.map((g) => (
          <div key={g.key} className="gp-group">
            <div className="gp-ghead">
              <i style={{ background: g.colour }} />
              {g.label}
            </div>
            {g.params.map((p) => {
              const setting = settings.get(p.name)!;
              const rings = p.owner[0] === 'components' && typeof p.owner[1] === 'number' ? ringRadii(getAt(options, ['components', p.owner[1], 'rnodes'])) : [];
              return (
                <div key={p.name} id={`gp-${p.name}`} data-name={p.name}>
                  <div
                    className="gp-row"
                    data-on={current === p.name || undefined}
                    onClick={() => {
                      setSelected(p.name);
                      list.current?.focus();
                    }}
                    onDoubleClick={() => setOpened(opened === p.name ? null : p.name)}
                  >
                    <span className="gp-t">
                      <b>{p.title ?? p.name}</b>
                      <small>{p.name}</small>
                    </span>
                    {typing?.name === p.name ? (
                      <input
                        className="gp-type"
                        autoFocus
                        value={typing.text}
                        placeholder={typing.kind === 'tie' ? 'an expression, e.g. gas_incl' : ''}
                        onChange={(event) => setTyping({ ...typing, text: event.currentTarget.value })}
                        onBlur={commit}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') commit();
                          if (event.key === 'Escape') {
                            setTyping(null);
                            list.current?.focus();
                          }
                        }}
                        aria-label={typing.kind === 'tie' ? 'Its tie' : 'Its value'}
                      />
                    ) : (
                      <span className="gp-v">
                        {valueText(setting)}
                        {setting.kind !== 'tied' && setting.kind !== 'other' && p.unit && <span> {unitText(p.unit)}</span>}
                      </span>
                    )}
                    {/* (how a fit takes it, said of those it does not keep
                    fixed: most are) */}
                    <span className="gp-st" data-kind={setting.kind} title={setting.kind === 'fixed' ? 'Fixed' : undefined}>
                      {setting.kind === 'fixed' ? '' : setting.kind === 'other' ? 'set' : setting.kind}
                    </span>
                    <Reach param={p} setting={setting} />
                  </div>
                  {opened === p.name && (
                    <div className="gp-open">
                      <ParamEditor ctx={ctx} param={p} rings={rings.length ? rings : null} />
                    </div>
                  )}
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
          <span className="gp-kbd">Enter</span>range
        </span>
      </div>
    </div>
  );
}

// Where a parameter stands: its values along its rings (one per ring), or
// of a free one, its value in its range; nothing of a fixed or tied one
function Reach({ param, setting }: { param: Param; setting: Setting }) {
  if ((setting.kind === 'fixed' || setting.kind === 'free') && setting.values.length > 1) {
    const values = setting.values;
    const low = Math.min(...values);
    const high = Math.max(...values);
    const y = (v: number) => 16 - ((v - low) / (high - low || 1)) * 13;
    const x = (i: number) => (i / (values.length - 1)) * 60 + 1;
    return (
      <svg className="gp-spark" viewBox="0 0 62 18" aria-label={`${values.length} values`}>
        <polyline points={values.map((v, i) => `${x(i)},${y(v)}`).join(' ')} />
      </svg>
    );
  }
  if (setting.kind !== 'free' || setting.min === null || setting.max === null || setting.max <= setting.min) return <span className="gp-range" data-none />;
  const at = Math.max(0, Math.min(1, ((setting.values[0] ?? setting.min) - setting.min) / (setting.max - setting.min)));
  return (
    <span className="gp-range" title={`${setting.min} to ${setting.max}${param.unit ? ` ${unitText(param.unit)}` : ''}`}>
      <i style={{ left: `${at * 100}%` }} />
    </span>
  );
}
