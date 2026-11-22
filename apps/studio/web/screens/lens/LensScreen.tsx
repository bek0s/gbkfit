// The screen of a lens of the project: a bending of the light of the
// galaxies behind it, at a redshift, made of pieces (e.g. an SIE and a
// shear, or the mass of a galaxy). Its sheet sets the lens (its redshift,
// and that of the source its quoted strengths are of) and its pieces, a
// box each with its parameters, whose values are those of the params block
// of the same name (by the names of the lens: e.g. sie_theta_e), saved with
// it. Its picture shows what it makes of a source behind it: the image
// plane, with its critical curves and the images of a test source, and
// the source plane, with its caustics and the source, which can be moved
// (not saved).
import { useMemo, useState } from 'react';
import { useLensDescription, useLensView } from '../../api/queries';
import { titleOf, type Options } from '../../components/SchemaForm';
import { Dock, DockSection } from '../../components/ui/dock';
import { NumberField } from '../../components/ui/inputs';
import { DropMenu } from '../../components/ui/overlays';
import { SheetLayout } from '../../components/ui/sheet';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useStudio } from '../../studio';
import { typesOf } from '../galaxy/schema';
import { isObject, useLensDraft, type LensDraft } from './draft';
import { LensPicture, type Source } from './Picture';

// A parameter of the lens, as the server describes it (see LensInfo)
interface LensParam {
  name: string;
  title?: string | null;
  unit: string | null;
  default: number | null;
  owner: [string, number];
}

export function LensScreen({ name }: { name: string }) {
  const { project } = useStudio();
  const root = project ?? '.';
  const draft = useLensDraft(root, name);
  const head = useBlockHead(root, 'lenses', name);
  const options = draft.state?.options;
  const properties = useMemo(() => draft.state?.properties ?? {}, [draft.state]);
  const description = useLensDescription(root, options).data;
  const params = (description?.parameters ?? []) as unknown as LensParam[];
  const problems = (description?.problems ?? []) as { message: string }[];
  // (the values that the picture takes: of the properties of the
  // parameters, or their defaults)
  const shown = useMemo(() => Object.fromEntries(params.flatMap((p) => (properties[p.name] !== undefined ? [[p.name, properties[p.name]]] : p.default !== null ? [[p.name, p.default]] : []))), [params, properties]);
  const view = useLensView(root, problems.length === 0 && params.length > 0 ? options : undefined, shown);
  const [source, setSource] = useState<Source>({ x: 0.1, y: 0.05, size: 0.1 });
  if (project === null) return null;
  const pieces = asList(options?.pieces);
  return (
    <section className="screen ln-screen">
      <ScreenHead up="lenses" title={head.title}>
        <SaveControls saving={draft} />
        {head.menu}
      </ScreenHead>
      {draft.error && <p className="error screen-error">{draft.error.message}</p>}
      {!draft.loaded || !options ? (
        <p className="note screen-error">Reading the lens…</p>
      ) : (
        <SheetLayout
          className="ob-body"
          width={420}
          sheet={
            <Dock label="The lens">
              <DockSection id="ln-lens" title="Lens">
                <LensFields draft={draft} options={options} />
              </DockSection>
              <DockSection id="ln-pieces" title="Pieces" note={pieces.map((p) => String(p.name ?? p.type)).join(' · ')}>
                <PiecesFields draft={draft} options={options} params={params} properties={properties} />
                {problems.map((p, i) => (
                  <p key={i} className="ln-problem">
                    {p.message.replace(/^[A-Za-z]+Error: /, '')}
                  </p>
                ))}
              </DockSection>
              <DockSection id="ln-source" title="Try it" note="not saved">
                <SourceFields source={source} onChange={setSource} />
              </DockSection>
            </Dock>
          }
          picture={<LensPicture view={view.data} error={view.error?.message ?? null} source={source} onSource={setSource} />}
        />
      )}
    </section>
  );
}

// The lens itself: its redshift, and that of the source its quoted
// strengths are of (e.g. the Einstein radius of an SIE, as lens models
// quote it)
function LensFields({ draft, options }: { draft: LensDraft; options: Options }) {
  const set = (key: string, value: number | undefined) => draft.edit((s) => ({ ...s, options: withKey(s.options, key, value) }));
  return (
    <div className="ob-form">
      <Row label="Redshift">
        <Number value={options.redshift} placeholder="needed" label="The redshift of the lens" onChange={(v) => set('redshift', v)} />
      </Row>
      <Row label="Source redshift">
        <Number value={options.source_redshift} placeholder="none" label="The redshift of the source of its quoted strengths" onChange={(v) => set('source_redshift', v)} />
      </Row>
      <p className="ob-text">Its quoted strengths (e.g. an Einstein radius) are those of a source at its source redshift; the light of galaxies at other redshifts is bent in proportion.</p>
    </div>
  );
}

// Its pieces, a box each: its name and type, and its parameters, whose
// values are the properties of the params block of the lens (a number, or
// free in a range); what adds a piece of a type
function PiecesFields({ draft, options, params, properties }: { draft: LensDraft; options: Options; params: LensParam[]; properties: Options }) {
  const defs = draft.defs;
  const pieces = asList(options.pieces);
  const types = typesOf(defs, 'lens_piece');
  const setPieces = (next: Options[]) => draft.edit((s) => ({ ...s, options: { ...s.options, pieces: next } }));
  const setProperty = (key: string, value: unknown) => draft.edit((s) => ({ ...s, properties: withKey(s.properties, key, value) }));
  const add = (type: string) => {
    const names = pieces.map((p) => String(p.name ?? p.type));
    let base = type;
    for (let i = 2; names.includes(base); i++) base = `${type}${i}`;
    setPieces([...pieces, base === type ? { type } : { type, name: base }]);
  };
  return (
    <div className="ln-pieces">
      {pieces.map((piece, i) => (
        <div key={i} className="ln-piece">
          <div className="ln-piece-head">
            <b>{String(piece.name ?? piece.type)}</b>
            <span>{titleOf(defs, 'lens_piece', String(piece.type))}</span>
            <button className="ln-x" aria-label="Remove the piece" title="Remove the piece" onClick={() => setPieces(pieces.filter((_, j) => j !== i))}>
              ×
            </button>
          </div>
          {params
            .filter((p) => p.owner[1] === i)
            .map((p) => (
              <ParamLine key={p.name} param={p} value={properties[p.name]} onChange={(value) => setProperty(p.name, value)} />
            ))}
          {piece.type === 'mass' && <p className="ob-text">The mass of the galaxy {String(piece.galaxy ?? '?')}: its parameters are those of the galaxy, which its setups lay out.</p>}
        </div>
      ))}
      <DropMenu
        label="Add a piece"
        width={300}
        entries={types.map((t) => ({ label: t.label, description: t.description, onClick: () => add(t.value) }))}
        target={<button className="gm-add">+ Add a piece</button>}
      />
    </div>
  );
}

// A parameter of a piece: its short name and title, and its value (a
// number), or free (its value and its range, a fit frees it within)
function ParamLine({ param, value, onChange }: { param: LensParam; value: unknown; onChange: (value: unknown) => void }) {
  const free = isObject(value);
  const number = free ? value.value : value;
  const unit = param.unit === 'arcsec' ? '″' : param.unit === 'deg' ? '°' : (param.unit ?? '');
  const short = param.name.slice(param.name.indexOf('_') + 1);
  const setValue = (v: number | undefined) => onChange(free ? { ...value, value: v } : v);
  return (
    <div className="ln-param">
      <code className="gm-pn">{short}</code>
      <span className="gm-pt">{param.title ?? short}</span>
      <Number value={number} placeholder={param.default === null ? 'needed' : String(param.default)} label={param.title ?? param.name} onChange={setValue} />
      <span className="ob-unit">{unit}</span>
      <button className="ln-free" aria-pressed={free} title={free ? 'Free in a fit: click to fix it' : 'Fixed: click to free it in a fit'} onClick={() => onChange(free ? value.value : { value: typeof number === 'number' ? number : (param.default ?? 0), min: null, max: null })}>
        {free ? 'free' : 'fixed'}
      </button>
      {free && (
        <span className="ln-range">
          <Number value={value.min} placeholder="min" label="Its least value" onChange={(v) => onChange({ ...value, min: v })} />
          <span>to</span>
          <Number value={value.max} placeholder="max" label="Its greatest value" onChange={(v) => onChange({ ...value, max: v })} />
        </span>
      )}
    </div>
  );
}

// The test source: its place behind the lens and its size (arcsec)
function SourceFields({ source, onChange }: { source: Source; onChange: (source: Source) => void }) {
  return (
    <div className="ob-form">
      <Row label="Source at x">
        <Number value={source.x} placeholder="0" label="The x of the source" onChange={(x) => onChange({ ...source, x: x ?? 0 })} />
      </Row>
      <Row label="Source at y">
        <Number value={source.y} placeholder="0" label="The y of the source" onChange={(y) => onChange({ ...source, y: y ?? 0 })} />
      </Row>
      <Row label="Source size">
        <Number value={source.size} placeholder="0.1" label="The size of the source (its sigma)" onChange={(size) => onChange({ ...source, size: size && size > 0 ? size : 0.1 })} />
      </Row>
      <p className="ob-text">A round source, only for the picture: drag it in the source plane too.</p>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="ob-row">
      <span className="ob-label">{label}</span>
      <div className="ob-field">{children}</div>
    </div>
  );
}

// A number, as typed: taken once it reads as one (blank: none)
function Number({ value, placeholder, label, onChange }: { value: unknown; placeholder: string; label: string; onChange: (value: number | undefined) => void }) {
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <span className="ob-number" onBlur={() => setTyping(null)}>
      <NumberField
        value={typing ?? (typeof value === 'number' ? String(value) : '')}
        placeholder={placeholder}
        label={label}
        onChange={(text) => {
          setTyping(text);
          const parsed = text.trim() === '' ? undefined : globalThis.Number(text);
          if (parsed === undefined || globalThis.Number.isFinite(parsed)) onChange(parsed);
        }}
      />
    </span>
  );
}

function asList(value: unknown): Options[] {
  return (Array.isArray(value) ? value : value === undefined || value === null ? [] : [value]).filter(isObject);
}

function withKey(options: Options, key: string, value: unknown): Options {
  const { [key]: _, ...rest } = options;
  return value === undefined ? rest : { ...rest, [key]: value };
}
