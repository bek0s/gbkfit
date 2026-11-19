// The recipe of a fit, its sheet: what the next run fits, without the
// words of the setups (of:, use:; see recipe.ts). The galaxies and the
// lenses it puts in place, each a piece of the project (by a name in the
// fit when there are several: its parameters are then named after it);
// its observations, each with its data and the noise of its mock data,
// and their likelihood; its parameters, in one table by item, each with
// what gives it its properties (the values of a piece, shared by the fits
// that use it, or this fit's own, where the edits go: a fit changes no
// piece); and its fitter.
import { useQueries } from '@tanstack/react-query';
import { useState, type ReactNode } from 'react';
import { api, data } from '../../api/client';
import { keys, useProject, useSchema } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { SchemaForm, type Defs } from '../../components/SchemaForm';
import { Dock, DockSection } from '../../components/ui/dock';
import { Choice, NumberField } from '../../components/ui/inputs';
import { DropMenu } from '../../components/ui/overlays';
import { useStudio } from '../../studio';
import { isObject } from '../galaxy/options';
import { typesOf } from '../galaxy/schema';
import type { SetupDraft } from './FitScreen';
import { plain } from './Start';
import { itemsOf, nameFor, ownProperties, sourceOf, withItemName, withItems, withOwnProperty, type Item, type Options, type Section } from './recipe';

// A parameter of the setup, as the server describes it (see SetupInfo)
export interface SetupParam {
  name: string;
  title: string | null;
  size: number;
  default: number | null;
  unit: string | null;
  item: string;
  properties: Record<string, unknown>;
  source: string | null;
}

interface Problem {
  severity: string;
  path: (string | number)[];
  message: string;
}

export function RecipeSheet({
  project,
  path,
  setup,
  draft,
  params,
  problems,
  described,
}: {
  project: string;
  path: string;
  setup: Options;
  draft: SetupDraft;
  params: SetupParam[];
  problems: Problem[];
  described: boolean;
}) {
  const pieces = useProject(project).data?.pieces;
  const defs = (useSchema().data as { $defs?: Defs } | undefined)?.$defs;
  // (the pieces that have values, params/<piece>, by the Studio's
  // convention)
  const valued = new Set(pieces?.params ?? []);
  const setItems = (section: Section, items: Item[]) => draft.edit((s) => withItems(s, section, items, valued));
  const free = params.filter((p) => isObject(p.properties[p.name])).length;
  const tied = params.filter((p) => typeof p.properties[p.name] === 'string').length;
  const fixed = params.length - free - tied;
  const setupFile = path.slice(project.length + 1);
  return (
    <Dock label="The recipe of the fit">
      <DockSection id="fx-galaxies" title="Galaxies" note={itemsOf(setup, 'galaxies').map((g) => g.name ?? g.piece ?? 'in place').join(' · ')} needed={itemsOf(setup, 'galaxies').length === 0}>
        <Items section="galaxies" setup={setup} draft={draft} all={pieces?.galaxies ?? []} params={params} valued={valued} onChange={(items) => setItems('galaxies', items)} />
      </DockSection>
      <DockSection id="fx-lenses" title="Lenses" note={itemsOf(setup, 'lenses').length ? itemsOf(setup, 'lenses').map((l) => l.name ?? l.piece).join(' · ') : 'none'} folded={itemsOf(setup, 'lenses').length === 0}>
        <Items section="lenses" setup={setup} draft={draft} all={pieces?.lenses ?? []} params={params} valued={valued} onChange={(items) => setItems('lenses', items)} />
      </DockSection>
      <DockSection id="fx-observations" title="Observations" note={String(itemsOf(setup, 'observations').length)} needed={itemsOf(setup, 'observations').length === 0}>
        <Observations project={project} setup={setup} draft={draft} all={pieces?.observations ?? []} defs={defs} onChange={(items) => setItems('observations', items)} />
      </DockSection>
      <DockSection id="fx-params" title="Parameters" note={params.length ? [`${free} free`, fixed && `${fixed} fixed`, tied && `${tied} tied`].filter(Boolean).join(' · ') : described ? 'none yet' : '…'}>
        <Params setup={setup} draft={draft} params={params} setupFile={setupFile} problems={problems} />
      </DockSection>
      <DockSection id="fx-fitter" title="Fitter" note={typeof setup.fitter === 'string' ? setup.fitter : isObject(setup.fitter) ? String(setup.fitter.type ?? '') : 'none'} needed={setup.fitter === undefined}>
        <Fitter project={project} setup={setup} draft={draft} all={pieces?.fitters ?? []} defs={defs} />
      </DockSection>
    </Dock>
  );
}

// The galaxies (or the lenses) of the fit: each its piece, under a name of
// the fit with several galaxies (or always, of a lens: its own name);
// what adds one (naming the first, as the second comes), changes one, or
// takes one out
function Items({
  section,
  setup,
  draft,
  all,
  params,
  valued,
  onChange,
}: {
  section: 'galaxies' | 'lenses';
  setup: Options;
  draft: SetupDraft;
  all: string[];
  params: SetupParam[];
  valued: Set<string>;
  onChange: (items: Item[]) => void;
}) {
  const { go } = useStudio();
  const items = itemsOf(setup, section);
  const kind = section === 'galaxies' ? 'galaxy' : 'lens';
  const names = items.map((item) => item.name).filter((n): n is string => n !== null);
  const parametersOf = (name: string | null) => params.filter((p) => p.item === (name ?? '')).map((p) => p.name);
  const add = (piece: string) => {
    if (section === 'galaxies' && items.length === 1 && items[0].name === null) {
      // (the first is named as the second comes: each needs a name)
      const first = nameFor(items[0].piece ?? 'galaxy', []);
      draft.edit((s) => {
        const named = withItemName(s, 'galaxies', 0, first, parametersOf(null), valued);
        return withItems(named, 'galaxies', [...itemsOf(named, 'galaxies'), { name: nameFor(piece, [first]), piece, options: null }], valued);
      });
      return;
    }
    const name = section === 'lenses' || items.length > 0 ? nameFor(piece, names) : null;
    onChange([...items, { name, piece, options: null }]);
  };
  return (
    <div className="fx-placed-list">
      {items.map((item, i) => (
        <div key={i} className="fx-placed">
          {item.name !== null && (
            <>
              <NameField value={item.name} label={`The name of the ${kind} in the fit: its parameters are named after it`} onChange={(name) => draft.edit((s) => withItemName(s, section, i, name, parametersOf(item.name), valued))} />
              <span className="fx-faint fx-small">uses</span>
            </>
          )}
          <span className="fx-placed-piece">
            {item.piece !== null ? (
              <Choice label={`The ${kind}`} value={item.piece} onChange={(piece) => onChange(items.map((it, j) => (j === i ? { ...it, piece } : it)))} items={all.map((name) => ({ value: name, label: name }))} />
            ) : (
              <span className="fx-faint fx-small">written in the fit</span>
            )}
          </span>
          {item.piece !== null && (
            <button className="fx-link" onClick={() => go({ kind: 'block', folder: section, name: item.piece! })}>
              Open →
            </button>
          )}
          <button className="fx-x" aria-label={`Take the ${kind} out`} title={`Take the ${kind} out of the fit`} onClick={() => onChange(items.filter((_, j) => j !== i))}>
            ×
          </button>
        </div>
      ))}
      <DropMenu
        label={`Add a ${kind}`}
        entries={all.length ? all.map((name) => ({ label: name, onClick: () => add(name) })) : [{ label: `The project has no ${kind} yet`, onClick: () => {} }]}
        target={<button className="gm-add">+ Add a {kind}</button>}
      />
      {section === 'galaxies' && items.length === 1 && items[0].name === null && <p className="ob-text">One galaxy needs no name here. With two or more, each gets a name in the fit, and its parameters are named after it.</p>}
      {section === 'galaxies' && Array.isArray(setup.depth_order) && (
        <p className="ob-text">
          In front first: <code>{setup.depth_order.join(' › ')}</code>
        </p>
      )}
    </div>
  );
}

// A name, as typed: taken once it is one (letters, digits and _)
function NameField({ value, label, onChange }: { value: string; label: string; onChange: (name: string) => void }) {
  const [typing, setTyping] = useState<string | null>(null);
  const text = typing ?? value;
  const valid = /^[A-Za-z_][A-Za-z0-9_]*$/.test(text);
  return (
    <input
      className="fx-role"
      data-invalid={!valid || undefined}
      value={text}
      aria-label={label}
      title={label}
      onChange={(event) => setTyping(event.target.value)}
      onBlur={() => {
        if (valid && text !== value) onChange(text);
        setTyping(null);
      }}
      onKeyDown={(event) => event.key === 'Enter' && (event.target as HTMLInputElement).blur()}
    />
  );
}

// The observations of the fit: each its piece, with its data and the noise
// of its mock data; and their likelihood (one for all)
function Observations({ project, setup, draft, all, defs, onChange }: { project: string; setup: Options; draft: SetupDraft; all: string[]; defs: Defs | undefined; onChange: (items: Item[]) => void }) {
  const { go } = useStudio();
  const items = itemsOf(setup, 'observations');
  const seen = useQueries({
    queries: items.map((item) => {
      const path = joinPath(project, `observations/${item.piece ?? ''}.yaml`);
      return { queryKey: keys.piece(path), enabled: item.piece !== null, queryFn: () => data(api.GET('/api/v1/pieces/{path}', { params: { path: { path } } })) };
    }),
  });
  const likelihoods = setup.likelihoods;
  const likelihood = likelihoods === undefined ? 'gauss' : isObject(likelihoods) && typeof likelihoods.type === 'string' ? likelihoods.type : null;
  return (
    <div className="fx-placed-list">
      {items.map((item, i) => {
        const options = item.options ?? (isObject(seen[i]?.data) ? (seen[i].data as Options) : null);
        const dataset = options?.dataset;
        const noise = options?.noise;
        return (
          <div key={i} className="fx-seen">
            <div className="fx-placed">
              <span className="fx-placed-piece">
                {item.piece !== null ? (
                  <Choice label="The observation" value={item.piece} onChange={(piece) => onChange(items.map((it, j) => (j === i ? { ...it, piece } : it)))} items={all.map((name) => ({ value: name, label: name }))} />
                ) : (
                  <span className="fx-faint fx-small">written in the fit</span>
                )}
              </span>
              {item.piece !== null && (
                <button className="fx-link" onClick={() => go({ kind: 'block', folder: 'observations', name: item.piece! })}>
                  Open →
                </button>
              )}
              <button className="fx-x" aria-label="Take the observation out" title="Take the observation out of the fit" onClick={() => onChange(items.filter((_, j) => j !== i))}>
                ×
              </button>
            </div>
            <div className="fx-seen-lines">
              <span className="ob-label">Data</span>
              <span className="fx-small">{typeof dataset === 'string' ? dataset : isObject(dataset) ? 'written in the observation' : <span className="fx-faint">none: a model alone, fitted to nothing</span>}</span>
              <span className="ob-label">Noise</span>
              <span className="fx-small" title="The noise of its mock data (its option noise)">
                {typeof noise === 'string' ? noise : isObject(noise) ? `${String(noise.type)}, its own` : dataset !== undefined ? 'Gaussian, of the errors of its data' : <span className="fx-faint">none</span>}
              </span>
            </div>
          </div>
        );
      })}
      <DropMenu
        label="Add an observation"
        entries={all.length ? all.map((name) => ({ label: name, onClick: () => onChange([...items, { name: null, piece: name, options: null }]) })) : [{ label: 'The project has no observation yet', onClick: () => {} }]}
        target={<button className="gm-add">{items.length ? '+ Another observation (a joint fit)' : '+ Add an observation'}</button>}
      />
      <div className="fx-seen-lines">
        <span className="ob-label">Likelihood</span>
        {likelihood === null ? (
          <span className="fx-small">by observation (in the file)</span>
        ) : (
          <span className="fx-likelihood">
            <Choice label="The likelihood of the observations" value={likelihood} onChange={(type) => draft.edit((s) => (type === 'gauss' ? withoutKey(s, 'likelihoods') : { ...s, likelihoods: { type } }))} items={typesOf(defs, 'likelihood')} />
          </span>
        )}
      </div>
    </div>
  );
}

// The parameters, by item: their starts and ranges (free), values
// (fixed) or ties, each with what gives it its properties; an edit goes to
// the fit's own layer (in its names: a property replaced whole), and a
// parameter of it can go back to what the layers before give it
function Params({ setup, draft, params, setupFile, problems }: { setup: Options; draft: SetupDraft; params: SetupParam[]; setupFile: string; problems: Problem[] }) {
  const own = ownProperties(setup);
  const items = [...new Set(params.map((p) => p.item))];
  const set = (key: string, property: unknown) => draft.edit((s) => withOwnProperty(s, key, property));
  const paramProblems = problems.filter((p) => p.path[0] === 'params');
  return (
    <div className="fx-params-sheet">
      {params.length === 0 ? (
        <p className="ob-text">The parameters show once the fit has galaxies and observations the library can set up.</p>
      ) : (
        <table className="fx-param-editor">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Free</th>
              <th className="fx-num">Start</th>
              <th className="fx-num">Min</th>
              <th className="fx-num">Max</th>
              <th>From</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <ItemRows key={item} item={item} params={params.filter((p) => p.item === item)} own={own} setupFile={setupFile} onSet={set} />
            ))}
          </tbody>
        </table>
      )}
      {paramProblems.map((p, i) => (
        <p key={i} className="fx-problem">
          {plain(p.message)}
        </p>
      ))}
      <p className="ob-text">Edits go to this fit alone: the values of a galaxy (its own screen sets them) stay as they are for its other fits. × takes a value of this fit out, back to that of the galaxy.</p>
    </div>
  );
}

function ItemRows({ item, params, own, setupFile, onSet }: { item: string; params: SetupParam[]; own: Options; setupFile: string; onSet: (key: string, property: unknown) => void }) {
  return (
    <>
      {item && (
        <tr className="fx-item-row">
          <td colSpan={6}>
            <code>{item}</code>
          </td>
        </tr>
      )}
      {params.map((p) => (
        <ParamRow key={p.name} param={p} item={item} own={own} setupFile={setupFile} onSet={onSet} />
      ))}
    </>
  );
}

function ParamRow({ param, item, own, setupFile, onSet }: { param: SetupParam; item: string; own: Options; setupFile: string; onSet: (key: string, property: unknown) => void }) {
  const key = param.name;
  const short = item ? key.slice(item.length + 1) : key;
  const property = param.properties[key];
  const from = sourceOf(param.source, setupFile);
  const unit = param.unit === 'arcsec' ? '″' : param.unit === 'deg' ? '°' : (param.unit ?? '');
  const title = `${param.title ?? key}${unit ? ` (${unit})` : ''}${item ? ` · ${key}` : ''}`;
  // (a vector set element by element, or a list: shown, not edited here)
  const keys = Object.keys(param.properties);
  const vector = param.size > 1 || keys.some((k) => k !== key) || Array.isArray(property);
  const free = isObject(property);
  const tie = typeof property === 'string' ? property : null;
  const value = free ? property.value : typeof property === 'number' ? property : (param.default ?? undefined);
  const setFree = (on: boolean) => onSet(key, on ? { value: typeof value === 'number' ? value : 0 } : typeof value === 'number' ? value : (param.default ?? 0));
  const setField = (field: 'value' | 'min' | 'max', v: number | undefined) => {
    if (!free) return onSet(key, v ?? param.default ?? 0);
    const next: Options = { ...property };
    if (v === undefined) delete next[field];
    else next[field] = v;
    onSet(key, next);
  };
  return (
    <tr data-fixed={(!free && tie === null) || undefined} data-own={from?.own || undefined}>
      <td className="fx-name" title={title}>
        {short}
      </td>
      {vector ? (
        <td colSpan={4} className="fx-tie">
          {param.size} elements{keys.length > 1 ? `, set ${keys.length} ways` : ''} (in the file)
        </td>
      ) : tie !== null ? (
        <>
          <td />
          <td colSpan={3} className="fx-tie">
            = <code>{tie}</code>
          </td>
        </>
      ) : (
        <>
          <td>
            <input type="checkbox" checked={free} aria-label={`${key} free in the fit`} onChange={(event) => setFree(event.target.checked)} />
          </td>
          <td>
            <Num value={value} placeholder={param.default === null ? 'needed' : String(param.default)} label={free ? `The start of ${key}` : `The value of ${key}`} onChange={(v) => setField('value', v)} />
          </td>
          <td>{free && <Num value={property.min} placeholder="none" label={`The least value of ${key}`} onChange={(v) => setField('min', v)} />}</td>
          <td>{free && <Num value={property.max} placeholder="none" label={`The greatest value of ${key}`} onChange={(v) => setField('max', v)} />}</td>
        </>
      )}
      <td>
        {from ? (
          <span className="fx-from" data-own={from.own || undefined} title={param.source ?? ''}>
            {from.label}
            {from.own && key in own && (
              <button className="fx-from-x" aria-label={`Take the value of this fit of ${key} out`} title="Take it out: back to the value before" onClick={() => onSet(key, undefined)}>
                ×
              </button>
            )}
          </span>
        ) : (
          <span className="fx-from fx-faint" title="No layer gives it: its default">
            default
          </span>
        )}
      </td>
    </tr>
  );
}

// A number, as typed: taken once it reads as one (blank: none)
function Num({ value, placeholder, label, onChange }: { value: unknown; placeholder: string; label: string; onChange: (value: number | undefined) => void }) {
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <span className="fx-num-field" onBlur={() => setTyping(null)}>
      <NumberField
        value={typing ?? (typeof value === 'number' ? String(value) : '')}
        placeholder={placeholder}
        label={label}
        onChange={(text) => {
          setTyping(text);
          const parsed = text.trim() === '' ? undefined : Number(text);
          if (parsed === undefined || Number.isFinite(parsed)) onChange(parsed);
        }}
      />
    </span>
  );
}

// The fitter: a piece of the project, or written in the fit (its type and
// options)
function Fitter({ project, setup, draft, all, defs }: { project: string; setup: Options; draft: SetupDraft; all: string[]; defs: Defs | undefined }) {
  const { go } = useStudio();
  const fitter = setup.fitter;
  const here = '(written here)';
  const choice = typeof fitter === 'string' ? fitter : isObject(fitter) ? here : '';
  const choose = (value: string) => draft.edit((s) => ({ ...s, fitter: value === here ? { type: typesOf(defs, 'fitter').some((t) => t.value === 'scipy.least_squares') ? 'scipy.least_squares' : (typesOf(defs, 'fitter')[0]?.value ?? 'scipy.least_squares') } : value }));
  return (
    <div className="ob-form">
      <Row label="Fitter">
        <span className="fx-placed">
          <span className="fx-placed-piece">
            <Choice label="The fitter" placeholder="none: choose one" value={choice} onChange={choose} items={[...all.map((name) => ({ value: name, label: name, group: 'Of the project' })), { value: here, label: 'Written in the fit', group: 'Here' }]} />
          </span>
          {typeof fitter === 'string' && (
            <button className="fx-link" onClick={() => go({ kind: 'block', folder: 'fitters', name: fitter })}>
              Open →
            </button>
          )}
        </span>
      </Row>
      {isObject(fitter) && defs && <SchemaForm defs={defs} def="fitter" project={project} value={fitter} onChange={(value) => draft.edit((s) => ({ ...s, fitter: value }))} />}
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="ob-row">
      <span className="ob-label">{label}</span>
      <div className="ob-field">{children}</div>
    </div>
  );
}

function withoutKey(options: Options, key: string): Options {
  const { [key]: _, ...rest } = options;
  return rest;
}
