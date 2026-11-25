// The sheet of an observation (see ui/rows and ui/pieces): its kind first
// (a menu of every kind, each with what it is in a line), which decides
// the rest; its sampling, the options of its kind (e.g. its pixels and
// channels, its regions, its UV coverage), with a menu that copies those
// of a dataset; its instrument (see Instrument); what it sees; the noise
// of its mock data; and more, folded: how its model is matched to data,
// and how accurately and fast it is made. Every option of the observation
// is in one of them, a row of the schema (see OptionRows); what each row
// makes (e.g. the field of its pixels) is said under its title. The
// library's problems are at their rows, and counted in the head.
import { useCallback, useState, type ReactNode } from 'react';
import { useLines } from '../../api/queries';
import type { ObservationSampling } from '../../api/queries';
import { ProblemField, useSeverityWithin } from '../../components/Problems';
import { propertiesOf, resolve, type Options, type Schema } from '../../components/SchemaForm';
import { FoldScope } from '../../components/ui/fold';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import { Row } from '../../components/ui/rows';
import { typesOf } from '../galaxy/schema';
import { ACCURACY_KEYS, CHANNEL_KEYS, formsOf, givenCount, hasOption, kindsOf, MATCHING_KEYS, noiseOf, objectAt, observationMember, PLACE_KEYS, retypedObservation, samplingKeys, SEES_KEYS, short, withKey, withPath } from './model';
import { OptionRow, OptionRows, type RowsContext } from './OptionRows';

// ---------------------------------------------------------------------------
// The kind

export function KindBlock({ ctx, options, member, onChange }: { ctx: RowsContext; options: Options; member: Schema | undefined; onChange: (options: Options) => void }) {
  const kinds = kindsOf(ctx.defs);
  const current = kinds.find((k) => k.type === options.type);
  return (
    <section className="sh-block" id="ob-kind">
      <h3 className="sh-head">Kind</h3>
      <ProblemField path={[]} label="The observation">
        <DropMenu
          label="Its kind"
          width={368}
          entries={kinds.map((k) => ({
            label: k.title,
            description: k.line,
            current: k.type === options.type,
            onClick: () => k.type !== options.type && onChange(retypedObservation(options, observationMember(ctx.defs, k.type), k.type)),
          }))}
          target={
            <button type="button" className="ob-kind" title={typeof member?.description === 'string' ? member.description : undefined}>
              <span>
                <b>{current?.title ?? String(options.type ?? 'Choose its kind')}</b>
                <small>{current?.line ?? ' '}</small>
              </span>
            </button>
          }
        />
      </ProblemField>
    </section>
  );
}

// ---------------------------------------------------------------------------
// The sampling

// The options of a sampling of pixels shown first: the number and the
// size of its pixels
const FIRST_KEYS = ['size', 'step'];

export function SamplingBlock({
  ctx,
  options,
  member,
  sampling,
  onChange,
  datasets,
  onCopy,
}: {
  ctx: RowsContext;
  options: Options;
  member: Schema;
  sampling: ObservationSampling | undefined;
  onChange: (options: Options) => void;
  datasets: string[];
  onCopy: (dataset: string) => void;
}) {
  const keys = samplingKeys(member);
  const properties = propertiesOf(member);
  // (its pixels and their size first, as the library's grids give them)
  const first = keys.includes('size') ? FIRST_KEYS : [];
  const shown = [...first.filter((key) => keys.includes(key)), ...keys.filter((key) => !PLACE_KEYS.includes(key) && !first.includes(key))];
  const placed = keys.filter((key) => PLACE_KEYS.includes(key));
  const set = (key: string, value: unknown) => onChange(withKey(options, key, value));
  const field = sampling?.field ?? null;
  const channels = sampling?.channels ?? null;
  // (what the rows make: the field of the pixels, the velocities of the
  // channels)
  const across = field ? (Math.abs(field.size[0] - field.size[1]) < 1e-6 * field.size[0] ? `${short(field.size[0])}″ across` : `${short(field.size[0])}″ × ${short(field.size[1])}″`) : ' ';
  const range = channels ? `${short(channels.first, 4)} … ${short(channels.last, 4)} km/s` : ' ';
  const menu: MenuEntry[] = [
    datasets.length ? { label: 'Copy from a dataset', entries: datasets.map((d) => ({ label: d, onClick: () => onCopy(d) })) } : { label: 'Copy from a dataset', onClick: () => {}, disabled: true, description: 'The project has no dataset' },
    { label: 'As by default', description: 'Its sampling options removed', onClick: () => onChange(Object.fromEntries(Object.entries(options).filter(([key]) => !keys.includes(key)))) },
  ];
  return (
    <section className="sh-block" id="ob-sampling">
      <h3 className="sh-head">
        Sampling
        <span className="sh-acts">
          <DropMenu label="Its sampling" width={260} at="end" entries={menu} target={moreButton('Copy the sampling of a dataset, or remove it')} />
        </span>
      </h3>
      <div className="sh-rows">
        {shown.map((key) =>
          key === 'spectral' ? (
            <ChannelRows key={key} ctx={ctx} node={properties.spectral} value={objectAt(options, 'spectral')} range={range} onChange={(value) => set('spectral', Object.keys(value).length ? value : undefined)} />
          ) : (
            <OptionRow key={key} ctx={ctx} name={key} node={properties[key]} value={options[key]} path={[key]} sub={key === 'step' ? across : undefined} hold={key === 'step'} onChange={(value) => set(key, value)} />
          ),
        )}
        {placed.length > 0 && (
          <FoldItem id="ob-place" title="On the sky" summary={placeText(options, placed)} count={givenCount(options, placed)}>
            <OptionRows ctx={ctx} node={member} keys={placed} value={options} path={[]} onChange={onChange} />
          </FoldItem>
        )}
      </div>
    </section>
  );
}

// The rows of a spectral axis: its channels (how many, how wide, the
// velocity of the reference one and the range of them all under it, and
// their convention), and its other options, folded
function ChannelRows({ ctx, node, value, range, onChange }: { ctx: RowsContext; node: Schema; value: Options; range: string; onChange: (value: Options) => void }) {
  const axis = (((resolve(node, ctx.defs).anyOf ?? []) as Schema[]).map((n) => resolve(n, ctx.defs)).find((n) => n.properties) ?? resolve(node, ctx.defs)) as Schema;
  const keys = Object.keys(propertiesOf(axis));
  const more = keys.filter((key) => !CHANNEL_KEYS.includes(key));
  return (
    <ProblemField path={['spectral']} label="Channels">
      <div className="sh-rows">
        <OptionRows ctx={ctx} node={axis} keys={CHANNEL_KEYS} value={value} path={['spectral']} subs={{ rval: range }} onChange={onChange} />
        <FoldItem id="ob-channels-more" title="More of the channels" summary={moreText(value, more)} count={givenCount(value, more)}>
          <OptionRows ctx={ctx} node={axis} keys={more} value={value} path={['spectral']} onChange={onChange} />
        </FoldItem>
      </div>
    </ProblemField>
  );
}

function placeText(options: Options, keys: string[]): string {
  const given = keys.filter((key) => options[key] != null);
  return given.length ? given.map((key) => `${key} ${textOf(options[key])}`).join(' · ') : 'as by default';
}

function moreText(options: Options, keys: string[]): string {
  const given = keys.filter((key) => options[key] != null);
  return given.length ? given.map((key) => `${key} ${textOf(options[key])}`).join(' · ') : 'as by default';
}

function textOf(value: unknown): string {
  if (typeof value === 'number') return short(value, 5);
  if (Array.isArray(value)) return value.map(textOf).join(', ');
  if (typeof value === 'string') return value;
  return '…';
}

// ---------------------------------------------------------------------------
// What it sees

export function SeesBlock({ ctx, options, member, onChange }: { ctx: RowsContext; options: Options; member: Schema; onChange: (options: Options) => void }) {
  const catalogue = (useLines().data ?? []) as { name: string }[];
  const properties = propertiesOf(member);
  const keys = SEES_KEYS.filter((key) => key in properties);
  const choices: Record<string, string[] | undefined> = { lines: catalogue.map((l) => l.name) };
  const nones: Record<string, string> = { lines: 'all' };
  const given = givenCount(options, keys);
  return (
    <section className="sh-block" id="ob-sees">
      <h3 className="sh-head">
        Sees <span className="sh-say">{given ? `${given} set` : 'all'}</span>
      </h3>
      <div className="sh-rows">
        {keys.map((key) => (
          <OptionRow key={key} ctx={ctx} name={key} node={properties[key]} value={options[key]} path={[key]} choices={choices[key]} none={nones[key]} onChange={(value) => onChange(withKey(options, key, value))} />
        ))}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// The noise of its mock data: none, one of the project (shared by name), or
// its own, of a type

export function NoiseBlock({ ctx, options, member, shared, onChange, titleOf }: { ctx: RowsContext; options: Options; member: Schema; shared: Options | undefined; onChange: (options: Options) => void; titleOf: (family: string, type: unknown) => string }) {
  const value = options.noise;
  const set = (next: unknown) => onChange(withKey(options, 'noise', next));
  const names = ctx.blocks.noise ?? [];
  const types = typesOf(ctx.defs, 'noise');
  const named = typeof value === 'string' ? value : null;
  const own = value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Options) : null;
  const text = value == null ? 'none' : named ? `${named} · shared` : titleOf('noise', own?.type);
  const sub = value == null ? 'a setup draws no mock data' : named ? (shared ? noiseOf(shared, titleOf) : ' ') : noiseOf(own ?? {}, titleOf);
  const form = own ? formsOf(ctx.defs, 'noise', own).form : undefined;
  if (!hasOption(member, 'noise')) return null;
  return (
    <section className="sh-block" id="ob-noise">
      <h3 className="sh-head">Mock noise</h3>
      <div className="sh-rows">
        <ProblemField path={['noise']} label="Noise">
          <Row
            title="Noise"
            sub={sub}
            wideBox
            box={
              <DropMenu
                label="The noise of its mock data"
                width={280}
                entries={[
                  { label: 'None', current: value == null, onClick: () => set(undefined) },
                  ...(names.length ? [{ section: 'Shared' }, ...names.map((n) => ({ label: n, current: n === named, onClick: () => set(n) }))] : []),
                  ...(named && shared ? [{ label: 'Make it its own (a copy)', onClick: () => set({ ...shared }) }] : []),
                  ...(named ? [{ label: `Open ${named}`, onClick: () => ctx.openBlock('noise', named) }] : []),
                  { section: 'Its own' },
                  ...types.map((t) => ({ label: t.label, current: own?.type === t.value, onClick: () => set({ type: t.value }) })),
                ]}
                target={
                  <button type="button" className="sh-num gm-choice">
                    <span data-default={value == null || undefined}>{text}</span>
                  </button>
                }
              />
            }
          />
        </ProblemField>
        {own && form && (
          <OptionRows
            ctx={ctx}
            node={form}
            keys={Object.keys(propertiesOf(form)).filter((key) => key !== 'type')}
            value={own}
            path={['noise']}
            onChange={(next) => set({ ...next, type: own.type })}
          />
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// More: how its model is matched to the data a fit pairs it with, and how
// accurately and fast it is made, each folded

export function MoreBlock({ ctx, options, member, onChange }: { ctx: RowsContext; options: Options; member: Schema; onChange: (options: Options) => void }) {
  const properties = propertiesOf(member);
  const matching = MATCHING_KEYS.filter((key) => key in properties);
  const accuracy = ACCURACY_KEYS.filter((key) => key in properties);
  const offsets = objectAt(options, 'offsets');
  if (matching.length + accuracy.length === 0) return null;
  return (
    <section className="sh-block" id="ob-more">
      <h3 className="sh-head">More</h3>
      <div className="sh-rows">
        {matching.length > 0 && (
          <FoldItem id="ob-matching" title="Matching the data" summary={givenCount(options, matching) ? givenText(options, matching, properties) : titlesText(matching, properties)} count={givenCount(options, matching)}>
            {matching.map((key) =>
              key === 'offsets' ? (
                <ProblemField key={key} path={['offsets']} label="Offsets">
                  {(['position', 'velocity'] as const).map((which) => (
                    <OptionRow
                      key={which}
                      ctx={ctx}
                      name={which}
                      title={which === 'position' ? 'Fit a position offset' : 'Fit a velocity offset'}
                      node={{ type: 'boolean', default: false, description: which === 'position' ? 'A shift of its frame on the sky, fitted ({name}_dx, {name}_dy).' : 'A shift of its velocities, fitted ({name}_dv).' }}
                      value={offsets[which]}
                      path={['offsets', which]}
                      onChange={(value) => onChange(withPath(options, ['offsets', which], value))}
                    />
                  ))}
                </ProblemField>
              ) : (
                <OptionRow key={key} ctx={ctx} name={key} node={properties[key]} value={options[key]} path={[key]} onChange={(value) => onChange(withKey(options, key, value))} />
              ),
            )}
          </FoldItem>
        )}
        {accuracy.length > 0 && (
          <FoldItem id="ob-accuracy" title="Accuracy and speed" summary={givenCount(options, accuracy) ? givenText(options, accuracy, properties) : 'as by default'} count={givenCount(options, accuracy)}>
            <OptionRows ctx={ctx} node={member} keys={accuracy} value={options} path={[]} onChange={onChange} />
          </FoldItem>
        )}
      </div>
    </section>
  );
}

// The options given of keys in a few words (their titles), or of all of
// them
function givenText(options: Options, keys: string[], properties: Record<string, Schema>): string {
  return titlesText(keys.filter((key) => options[key] != null), properties);
}

function titlesText(keys: string[], properties: Record<string, Schema>): string {
  return keys.map((key) => (typeof properties[key]?.title === 'string' ? (properties[key].title as string).toLowerCase() : key)).join(' · ');
}

// ---------------------------------------------------------------------------
// Small parts

// An item that folds (as those of a component of a galaxy): its line (a
// chevron, its title, what it holds in a few words, how many of its
// options are given) and, open, its rows; the problems in it mark its line
// while it is folded, and open it as the counter leads to them
function FoldItem({ id, title, summary, count, children }: { id: string; title: string; summary: string; count: number; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const show = useCallback(() => setOpen(true), []);
  const severity = useSeverityWithin(id);
  return (
    <div className="gm-item ob-item" data-open={open || undefined} data-severity={(!open && severity) || undefined}>
      <button type="button" className="gm-item-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <svg className="sh-chev" viewBox="0 0 16 16" aria-hidden="true">
          <path d="M6 4l4 4-4 4" />
        </svg>
        <b>{title}</b>
        <span className="gm-item-sum">{summary}</span>
        <span className="gm-free">{count > 0 ? `${count} set` : ''}</span>
      </button>
      <FoldScope id={id} open={show}>
        <div className="gm-item-body" hidden={!open}>
          {children}
        </div>
      </FoldScope>
    </div>
  );
}

// The button of the menu of a block (its actions)
function moreButton(title: string) {
  return (
    <button type="button" className="sh-icon" title={title} aria-label={title}>
      <svg viewBox="0 0 16 16">
        <circle cx="3.5" cy="8" r=".9" />
        <circle cx="8" cy="8" r=".9" />
        <circle cx="12.5" cy="8" r=".9" />
      </svg>
    </button>
  );
}
