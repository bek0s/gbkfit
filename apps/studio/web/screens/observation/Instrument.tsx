// The instrument of an observation in its sheet: its primary beam, its
// PSF and its LSF, a piece each (see ui/pieces). A piece is none (a perfect one), one of the project shared
// by name (its options shown as they are there: opened, or made its own
// to change them), or its own, of a type (and of a form of it, e.g. a
// Gaussian by its dispersion or by the axes of a beam), its options rows
// of the schema. Its head says its type, what it is in a few words while
// folded, and its width (the FWHM the server measures); its menu copies
// the beam of a dataset (a PSF), uses a shared one, makes a shared one its
// own, opens it, gives it another type, or none.
import { useState, type ReactNode } from 'react';
import { propertiesOf, type Options, type Schema } from '../../components/SchemaForm';
import { ProblemField } from '../../components/Problems';
import { FoldAll, PieceHead } from '../../components/ui/pieces';
import { ChoiceBox, Row } from '../../components/ui/rows';
import type { MenuEntry } from '../../components/ui/overlays';
import { unitText } from '../../components/units';
import { retyped, typesOf } from '../galaxy/schema';
import { formsOf, isObject, objectAt, short, withKey, type Ellipse } from './model';
import { optionTitle, OptionRow, pairText, type RowsContext } from './OptionRows';

// The parts of an instrument: the family of each in the schema, its
// title, the folder of the project its shared ones are in (none: they
// are not shared), and what none of it is
const PARTS = [
  { key: 'primary_beam', title: 'Primary beam', folder: null, none: 'no attenuation', noneEntry: 'None: no attenuation' },
  { key: 'psf', title: 'PSF', folder: 'psfs', none: 'a perfect instrument', noneEntry: 'None: a perfect instrument' },
  { key: 'lsf', title: 'LSF', folder: 'lsfs', none: 'a perfect spectrograph', noneEntry: 'None: a perfect spectrograph' },
] as const;

type PartKey = (typeof PARTS)[number]['key'];

export function InstrumentBlock({
  ctx,
  options,
  onChange,
  shared,
  psf,
  lsfFwhm,
  channel,
  datasets,
  onCopyBeam,
  hidden,
  onHide,
}: {
  ctx: RowsContext;
  options: Options;
  onChange: (options: Options) => void;
  // The options of the shared parts it names, as read (by part)
  shared: Partial<Record<PartKey, Options>>;
  // The PSF and the FWHM of the LSF (km/s) as the server measures them,
  // and the width of its channels (km/s)
  psf: Ellipse | null;
  lsfFwhm: number | null;
  channel: number | null;
  // The datasets of the project, and what copies the beam of one
  datasets: string[];
  onCopyBeam: (dataset: string) => void;
  // The parts hidden in the picture (not saved), and what hides or shows
  // one
  hidden: ReadonlySet<string>;
  onHide: (key: string) => void;
}) {
  const instrument = objectAt(options, 'instrument');
  const set = (key: PartKey, value: unknown) => {
    const next = withKey(instrument, key, value);
    onChange(withKey(options, 'instrument', Object.keys(next).length ? next : undefined));
  };
  // (the parts unfolded: at first those of their own options)
  const [open, setOpen] = useState<Set<PartKey>>(() => new Set(PARTS.filter((p) => isObject(instrument[p.key])).map((p) => p.key)));
  const toggle = (key: PartKey) => setOpen((now) => (now.has(key) ? new Set([...now].filter((k) => k !== key)) : new Set([...now, key])));
  const keys = PARTS.map((p) => p.key);
  return (
    <section className="sh-block" id="ob-instrument">
      <h3 className="sh-head">
        Instrument <span className="sh-count">{PARTS.length}</span>
        <span className="sh-acts">
          <FoldAll what="part" canUnfold={!keys.every((k) => open.has(k))} canFold={keys.some((k) => open.has(k))} onUnfold={() => setOpen(new Set(keys))} onFold={() => setOpen(new Set())} />
        </span>
      </h3>
      <div className="sh-pieces">
        {PARTS.map((part) => (
          <Part
            key={part.key}
            ctx={ctx}
            part={part}
            value={instrument[part.key]}
            onChange={(value) => set(part.key, value)}
            shared={shared[part.key]}
            open={open.has(part.key)}
            onFold={() => toggle(part.key)}
            badge={badgeOf(part.key, instrument[part.key], shared[part.key], psf, lsfFwhm)}
            subs={subsOf(part.key, psf, lsfFwhm, channel)}
            beamMenu={part.key === 'psf' && datasets.length > 0 ? { label: 'Copy the beam of a dataset', entries: datasets.map((d) => ({ label: d, onClick: () => onCopyBeam(d) })) } : null}
            hidden={hidden.has(part.key)}
            onHide={instrument[part.key] == null ? null : () => onHide(part.key)}
          />
        ))}
      </div>
    </section>
  );
}

// The eye of a part: it hides the part in the picture, or shows it again
// (as that of a piece of a lens)
function EyeButton({ hidden, what, onClick }: { hidden: boolean; what: string; onClick: () => void }) {
  return (
    <button
      type="button"
      className="sh-icon"
      title={hidden ? 'Show it in the picture' : 'Hide it in the picture (not saved)'}
      aria-label={hidden ? `Show the ${what} in the picture` : `Hide the ${what} in the picture`}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
    >
      {hidden ? (
        <svg viewBox="0 0 16 16">
          <path d="M2 2l12 12" />
          <path d="M6.2 4c.6-.3 1.2-.5 1.8-.5 4 0 6.5 4.5 6.5 4.5s-.7 1.3-2 2.5M4 5.6C2.4 6.8 1.5 8 1.5 8s2.5 4.5 6.5 4.5c.9 0 1.8-.2 2.6-.6" />
        </svg>
      ) : (
        <svg viewBox="0 0 16 16">
          <path d="M1.5 8s2.5-4.5 6.5-4.5 6.5 4.5 6.5 4.5-2.5 4.5-6.5 4.5S1.5 8 1.5 8z" />
          <circle cx="8" cy="8" r="1.8" />
        </svg>
      )}
    </button>
  );
}

function Part({
  ctx,
  part,
  value,
  onChange,
  shared,
  open,
  onFold,
  badge,
  subs,
  beamMenu,
  hidden,
  onHide,
}: {
  ctx: RowsContext;
  part: (typeof PARTS)[number];
  value: unknown;
  onChange: (value: unknown) => void;
  shared: Options | undefined;
  open: boolean;
  onFold: () => void;
  badge: string | null;
  subs: Record<string, ReactNode>;
  beamMenu: MenuEntry | null;
  // Whether it is hidden in the picture, and what hides or shows it (none:
  // there is nothing to hide)
  hidden: boolean;
  onHide: (() => void) | null;
}) {
  const { defs } = ctx;
  const named = typeof value === 'string' ? value : null;
  const own = isObject(value) ? value : null;
  const types = typesOf(defs, part.key);
  const typeTitle = (type: unknown) => types.find((t) => t.value === type)?.label ?? String(type);
  const sharedNames = part.folder ? (ctx.blocks[part.folder] ?? []) : [];
  const typeText = named ? `${named} · shared` : own ? typeTitle(own.type) : 'none';
  const summary = value == null ? part.none : named ? (shared ? `set in ${named}: ${optionsText(defs, part.key, shared)}` : `set in ${named}`) : optionsText(defs, part.key, own ?? {});
  const menu: MenuEntry[] = [
    ...(beamMenu ? [beamMenu] : []),
    ...(sharedNames.length ? [{ label: `Use a shared ${part.title}`, entries: sharedNames.map((n) => ({ label: n, current: n === named, onClick: () => onChange(n) })) }] : []),
    ...(named && shared ? [{ label: 'Make it its own (a copy)', onClick: () => onChange({ ...shared }) }] : []),
    ...(named && part.folder ? [{ label: `Open ${named}`, onClick: () => ctx.openBlock(part.folder!, named) }] : []),
    { label: own ? 'Another type' : `A ${part.title} of its own`, entries: types.map((t) => ({ label: t.label, description: t.description, current: own?.type === t.value, onClick: () => onChange(retyped(defs, part.key, own ?? {}, t.value)) })) },
    ...(value != null ? [{ label: part.noneEntry, onClick: () => onChange(undefined) }] : []),
  ];
  return (
    <div className="sh-piece ob-piece" data-open={open || undefined} data-none={value == null || undefined}>
      <ProblemField path={['instrument', part.key]} label={part.title}>
        <PieceHead
          kind="part"
          name={part.title}
          type={typeText}
          summary={summary}
          badge={badge}
          badgeTitle={badge ? 'As the server measures it' : undefined}
          open={open}
          selected={false}
          onFold={onFold}
          onSelect={onFold}
          menu={menu}
          menuTitle={`Its ${part.title}: shared, its own, of a type, or none`}
          eye={onHide ? <EyeButton hidden={hidden} what={part.title} onClick={onHide} /> : undefined}
        />
      </ProblemField>
      {open && (
        <div className="sh-piece-body">
          {value == null && <p className="sh-note ob-none-note">None: {part.none}. Its menu gives it one.</p>}
          {named && <SharedRows defs={defs} family={part.key} name={named} options={shared} />}
          {own && <OwnRows ctx={ctx} family={part.key} value={own} onChange={onChange} subs={subs} />}
        </div>
      )}
    </div>
  );
}

// The rows of a part of its own: the form of its type (if it has several,
// e.g. a Gaussian by its dispersion, its FWHM, the axes of a beam, or a
// header), and its options
function OwnRows({ ctx, family, value, onChange, subs }: { ctx: RowsContext; family: PartKey; value: Options; onChange: (value: unknown) => void; subs: Record<string, ReactNode> }) {
  const { defs } = ctx;
  const { form, forms } = formsOf(defs, family, value);
  const properties = form ? propertiesOf(form) : {};
  const keys = Object.keys(properties).filter((key) => key !== 'type');
  const base: (string | number)[] = ['instrument', family];
  return (
    <>
      {forms.length > 1 && form && (
        <Row
          title="Given by"
          wideBox
          box={
            <FormPick
              forms={forms}
              form={form}
              onPick={(next) => {
                // (the options of the new form that the old has kept; the
                // others to give)
                const kept = Object.keys(propertiesOf(next)).filter((key) => key in value);
                onChange({ type: value.type, ...Object.fromEntries(kept.map((key) => [key, value[key]])) });
              }}
            />
          }
        />
      )}
      {keys.map((key) => (
        <OptionRow
          key={key}
          ctx={ctx}
          name={key}
          node={properties[key]}
          value={value[key]}
          path={[...base, key]}
          sub={subs[key]}
          hold={key in subs}
          onChange={(next) => onChange(withKey(value, key, next))}
        />
      ))}
    </>
  );
}

// The forms of a type to choose, each by its required options (e.g.
// 'Major FWHM, Minor FWHM')
function FormPick({ forms, form, onPick }: { forms: Schema[]; form: Schema; onPick: (form: Schema) => void }) {
  const label = (m: Schema) => {
    const properties = propertiesOf(m);
    const keys = ((m.required as string[] | undefined) ?? []).filter((key) => key !== 'type');
    return keys.map((key) => optionTitle(properties[key], key)).join(', ') || 'its defaults';
  };
  return <ChoiceBox label="The form of its options" text={label(form)} width={260} entries={forms.map((m) => ({ label: label(m), current: m === form, onClick: () => m !== form && onPick(m) }))} />;
}

// The options of a shared part as they are there, a row each, read only
function SharedRows({ defs, family, name, options }: { defs: RowsContext['defs']; family: PartKey; name: string; options: Options | undefined }) {
  if (!options) return <p className="sh-note">Reading {name}…</p>;
  const properties = propertiesOf(formsOf(defs, family, options).form ?? {});
  return (
    <>
      {Object.entries(options)
        .filter(([key]) => key !== 'type')
        .map(([key, v]) => (
          <Row
            key={key}
            title={optionTitle(properties[key], key)}
            sub={`set in ${name}`}
            box={<span className="sh-num ob-ro">{valueText(v)}</span>}
            unit={typeof properties[key]?.unit === 'string' ? unitText(properties[key].unit as string) : ''}
          />
        ))}
    </>
  );
}

// The options of a part in a few words (e.g. 'Dispersion 23.5″ · Axis
// ratio 0.93')
function optionsText(defs: RowsContext['defs'], family: PartKey, options: Options): string {
  const properties = propertiesOf(formsOf(defs, family, options).form ?? {});
  const parts = Object.entries(options)
    .filter(([key]) => key !== 'type')
    .map(([key, v]) => `${optionTitle(properties[key], key)} ${valueText(v)}${typeof properties[key]?.unit === 'string' ? unitText(properties[key].unit as string) : ''}`);
  return parts.join(' · ') || 'its defaults';
}

function valueText(value: unknown): string {
  if (typeof value === 'number') return String(Number(value.toPrecision(4)));
  if (typeof value === 'string') return value;
  if (Array.isArray(value) && value.every((v) => typeof v === 'number')) return pairText(value);
  return '…';
}

// The badge of a part: its FWHM as the server measures it (of a PSF, of
// both its axes if it is not round; of an LSF, km/s), or as given (of a
// primary beam)
function badgeOf(key: PartKey, value: unknown, shared: Options | undefined, psf: Ellipse | null, lsfFwhm: number | null): string | null {
  if (value == null) return null;
  if (key === 'psf' && psf) return Math.abs(psf.major - psf.minor) > 1e-3 * psf.major ? `FWHM ${short(psf.major)}″ × ${short(psf.minor)}″` : `FWHM ${short(psf.major)}″`;
  if (key === 'lsf' && lsfFwhm !== null) return `FWHM ${short(lsfFwhm)} km/s`;
  const options = isObject(value) ? value : shared;
  if (key === 'primary_beam' && typeof options?.fwhm === 'number') return `FWHM ${short(options.fwhm)}″`;
  return null;
}

// What is said under the rows of the width of a part: of a PSF, its FWHM;
// of an LSF, its FWHM in channels
function subsOf(key: PartKey, psf: Ellipse | null, lsfFwhm: number | null, channel: number | null): Record<string, ReactNode> {
  if (key === 'psf') {
    const fwhm = psf ? `FWHM ${short(psf.major)}″` : ' ';
    return { sigma: fwhm, alpha: fwhm };
  }
  if (key === 'lsf') {
    const across = lsfFwhm !== null && channel ? `FWHM ${short(lsfFwhm / Math.abs(channel))} channels` : ' ';
    return { sigma: across, fwhm: across, alpha: across, gamma: across };
  }
  return {};
}
