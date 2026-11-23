// The screen of a lens of the project: a bending of the light of the
// galaxies behind it, at a redshift, made of pieces (e.g. an SIE and a
// shear, or the mass of a galaxy). Its sheet shows the path of the light
// (you, the lens, the source its quoted strengths are of) with both
// redshifts, and its pieces, a line each that unfolds into its parameters
// (slider rows, see SliderRow), whose values are those of the params block
// of the same name (by the names of the lens: e.g. sie_theta_e), saved
// with it, and its own source redshift, if it has one. A piece's eye hides
// it in the picture only. Its picture shows what the lens makes of a
// source behind it (see LensPicture), with handles of the piece selected,
// over a field fitted to the lens as the screen opens (or fitted again,
// or zoomed), kept as it is edited.
import { useDebouncedValue } from '@mantine/hooks';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLensDescription, useLensView, type LensFrame } from '../../api/queries';
import { membersOf, propertiesOf, titleOf, typeOf, type Defs, type Options } from '../../components/SchemaForm';
import { NeedsNote } from '../../components/ui/dock';
import { DropMenu } from '../../components/ui/overlays';
import { SheetLayout } from '../../components/ui/sheet';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useStudio } from '../../studio';
import { NameText } from '../galaxy/Model';
import { typesOf } from '../galaxy/schema';
import { isObject, useLensDraft, type LensDraft } from './draft';
import { LightPath } from './LightPath';
import { bareView, LensPicture, zoomed, type PictureNote, type Source } from './Picture';
import { fmt, NumberInput, numberOf, SliderRow, unitText, useScrub, type LensParam, type ParamValue } from './SliderRow';

// The colours of the pieces, in order (those of the components of a
// galaxy)
export const PIECE_COLOURS = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat1'];

export function LensScreen({ name }: { name: string }) {
  const { project } = useStudio();
  const root = project ?? '.';
  const draft = useLensDraft(root, name);
  const head = useBlockHead(root, 'lenses', name);
  const options = draft.state?.options;
  const properties = useMemo(() => draft.state?.properties ?? {}, [draft.state]);
  const description = useLensDescription(root, options).data;
  const problems = (description?.problems ?? []) as { message: string; path?: (string | number)[] }[];
  const pieces = asList(options?.pieces);
  // (its parameters as the library describes them; of a lens with a
  // problem, which has none, those it had last, of the pieces it still
  // has, so that its rows stay while it is mended)
  const described = (description?.parameters ?? []) as unknown as LensParam[];
  const last = useRef<LensParam[]>([]);
  if (described.length > 0) last.current = described;
  const params = described.length > 0 ? described : last.current.filter((p) => p.owner[1] < pieces.length);
  // (the pieces unfolded, the one selected, whose handles the picture
  // shows, and those hidden in the picture: none saved)
  const [open, setOpen] = useState<Set<number>>(() => new Set([0]));
  const [selected, setSelected] = useState<number | null>(null);
  // (the parameter whose row shows its slider)
  const [openRow, setOpenRow] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<number>>(() => new Set());
  const [source, setSource] = useState<Source>({ x: 0.1, y: 0.05, size: 0.1 });
  // (the field of the picture: null until the server fits one to the lens)
  const [frame, setFrame] = useState<LensFrame | null>(null);
  // (whether no piece is shown, every one hidden or none yet: the picture
  // shows the source as it is, of no call to the server)
  const bare = !!options && problems.length === 0 && pieces.every((_, i) => hidden.has(i));
  // (what the picture is of: the lens, its hidden pieces left out by the
  // server (the lens is still the whole of it, e.g. its source_redshift
  // taken by those pieces only), at the values of the properties of its
  // parameters, or their defaults; asked for once they stand a moment,
  // not at each step of a drag)
  const shown = useMemo(() => {
    if (!options || problems.length > 0 || described.length === 0 || bare) return null;
    const values = Object.fromEntries(params.map((p) => [p.name, properties[p.name] !== undefined ? properties[p.name] : p.default]));
    return { lens: options, values, hidden: pieces.map((_, i) => i).filter((i) => hidden.has(i)) };
  }, [options, problems.length, described.length, bare, params, pieces, hidden, properties]);
  const bareShown = useMemo(() => (bare ? bareView(frame) : undefined), [bare, frame]);
  const [settled] = useDebouncedValue(shown, 80);
  const view = useLensView(root, settled?.lens, settled?.values ?? {}, frame, settled?.hidden ?? []);
  // (what the picture draws: the view of what is shown, else the last it
  // drew: while the next comes, or while the lens cannot be drawn; and a
  // note of why, if it is not the lens as it is)
  const drawn = useLast(bareShown ?? (settled && !view.isError ? view.data : undefined));
  const note: PictureNote | null =
    problems.length > 0
      ? { text: 'The last picture: the lens has a problem (see the head)', bad: true }
      : view.isError && !bare
        ? { text: `The last picture: ${view.error.message.replace(/\b[A-Za-z]+Error: /g, '')}`, bad: true }
        : pieces.length === 0
          ? { text: 'No piece yet: the source as it is', bad: false }
          : bare
            ? { text: 'Every piece hidden: the source as it is', bad: false }
            : null;
  // (the field fitted, kept; the test source brought into its source
  // plane, if it is out of it)
  const fitted = frame === null && view.data ? view.data : null;
  useEffect(() => {
    if (!fitted) return;
    const { half, centre } = fitted;
    setFrame({ half, centre: [centre[0], centre[1]] });
    setSource((s) => (Math.abs(s.x - centre[0]) < half / 2 && Math.abs(s.y - centre[1]) < half / 2 ? s : { ...s, x: centre[0] + half / 30, y: centre[1] + half / 60 }));
  }, [fitted]);
  if (project === null) return null;
  // (how far the picture reaches from the origin, the span of the slider
  // of a position)
  const field = frame ? frame.half + Math.max(Math.abs(frame.centre[0]), Math.abs(frame.centre[1])) : 3;
  const setProperty = (key: string, value: unknown) => draft.edit((s) => ({ ...s, properties: withKey(s.properties, key, value) }));
  // (a value set by a handle: of a free parameter, its value in its range)
  const setValue = (key: string, value: number) =>
    draft.edit((s) => {
      const current = s.properties[key];
      return { ...s, properties: withKey(s.properties, key, isObject(current) ? { ...current, value } : value) };
    });
  const values = Object.fromEntries(params.map((p) => [p.name, numberOf(p, properties[p.name] as ParamValue)]));
  return (
    <section className="screen ln-screen">
      <ScreenHead up="lenses" title={head.title}>
        <NeedsNote needs={problems.map((p) => ({ text: p.message.replace(/^\[[^\]]*\] /, '').replace(/^[A-Za-z]+Error: /, ''), at: 'ln-pieces' }))} />
        <SaveControls saving={draft} />
        {head.menu}
      </ScreenHead>
      {draft.error && <p className="error screen-error">{draft.error.message}</p>}
      {!draft.loaded || !options ? (
        <p className="note screen-error">Reading the lens…</p>
      ) : (
        <SheetLayout
          className="ob-body"
          width={404}
          sheet={
            <aside className="lp-sheet" aria-label="The lens">
              <LensBlock draft={draft} options={options} />
              <PiecesBlock
                draft={draft}
                options={options}
                pieces={pieces}
                params={params}
                properties={properties}
                problems={problems}
                field={field}
                open={open}
                selected={selected}
                openRow={openRow}
                onOpenRow={(name) => setOpenRow(openRow === name ? null : name)}
                hidden={hidden}
                onOpen={setOpen}
                onSelect={setSelected}
                onHide={setHidden}
                onProperty={setProperty}
              />
            </aside>
          }
          picture={
            <LensPicture
              view={drawn}
              note={note}
              onZoom={(wider) => {
                // (of the field kept, else of that drawn, e.g. of no piece)
                const at = frame ?? (drawn && { half: drawn.half, centre: [drawn.centre[0], drawn.centre[1]] as [number, number] });
                if (at) setFrame({ ...at, half: zoomed(2 * at.half, wider) / 2 });
              }}
              onFit={() => setFrame(null)}
              source={source}
              onSource={setSource}
              handles={selected === null ? null : handlesOf(pieces[selected], params.filter((p) => p.owner[1] === selected), values, hidden.has(selected))}
              dots={centresOf(pieces, params, values, hidden, selected)}
              onPick={(i) => {
                setSelected(i);
                setOpen(new Set([...open, i]));
              }}
              onClearPick={() => setSelected(null)}
              onHandle={setValue}
            />
          }
        />
      )}
    </section>
  );
}

// The latest of a value that was given (e.g. the last view drawn, while
// the next comes)
function useLast<T>(value: T | undefined): T | undefined {
  const last = useRef(value);
  if (value !== undefined) last.current = value;
  return last.current;
}

// The lens itself: the path of the light, and its redshift and that of the
// source its quoted strengths are of (e.g. the Einstein radius of an SIE,
// as lens models quote it), of the pieces without one of their own
function LensBlock({ draft, options }: { draft: LensDraft; options: Options }) {
  const set = (key: string, value: number | undefined) => draft.edit((s) => ({ ...s, options: withKey(s.options, key, value) }));
  const z = typeof options.redshift === 'number' ? options.redshift : undefined;
  const zs = typeof options.source_redshift === 'number' ? options.source_redshift : undefined;
  const setZ = (v: number) => set('redshift', Math.max(0.001, v));
  const setZs = (v: number) => set('source_redshift', Math.max(0.001, v));
  const scrubZ = useScrub(z, [0, 2], setZ);
  const scrubZs = useScrub(zs, [0, 4], setZs);
  return (
    <section className="lp-block">
      <h3 className="lp-head">Lens</h3>
      <LightPath lens={z} source={zs} />
      <div className="lp-z">
        <span className="lp-z-field">
          <span {...scrubZ}>Redshift</span>
          <NumberInput value={z} label="The redshift of the lens" placeholder="needed" onChange={setZ} onClear={() => set('redshift', undefined)} />
        </span>
        <span className="lp-z-field">
          <span {...scrubZs} title="The redshift of the source its quoted strengths (e.g. an Einstein radius) are those of, of the pieces without their own (drag sideways to change it)">
            Source redshift
          </span>
          <NumberInput value={zs} label="The source redshift of the lens" placeholder="none" onChange={setZs} onClear={() => set('source_redshift', undefined)} />
        </span>
      </div>
    </section>
  );
}

// Its pieces: a line each, which a click unfolds and selects (its handles
// in the picture; folded, no longer selected); unfolded, its parameters
// and its own source redshift; its menu renames it, shows it alone in the
// picture, duplicates it or removes it, its values with it (they are named
// after it: e.g. main_theta_e); what adds a piece of a type, and unfolds
// or folds them all
function PiecesBlock({
  draft,
  options,
  pieces,
  params,
  properties,
  problems,
  field,
  open,
  selected,
  openRow,
  onOpenRow,
  hidden,
  onOpen,
  onSelect,
  onHide,
  onProperty,
}: {
  draft: LensDraft;
  options: Options;
  pieces: Options[];
  params: LensParam[];
  properties: Options;
  problems: { path?: (string | number)[] }[];
  field: number;
  open: Set<number>;
  selected: number | null;
  openRow: string | null;
  onOpenRow: (name: string) => void;
  hidden: Set<number>;
  onOpen: (open: Set<number>) => void;
  onSelect: (i: number | null) => void;
  onHide: (hidden: Set<number>) => void;
  onProperty: (key: string, value: unknown) => void;
}) {
  const defs = draft.defs;
  const types = typesOf(defs, 'lens_piece');
  const setPieces = (next: Options[]) => draft.edit((s) => ({ ...s, options: { ...s.options, pieces: next } }));
  const add = (type: string) => {
    const names = pieces.map((p) => String(p.name ?? p.type));
    let base = type;
    for (let i = 2; names.includes(base); i++) base = `${type}${i}`;
    setPieces([...pieces, base === type ? { type } : { type, name: base }]);
    onOpen(new Set([...open, pieces.length]));
    onSelect(pieces.length);
  };
  const [renaming, setRenaming] = useState<number | null>(null);
  const nameOf = (piece: Options) => String(piece.name ?? piece.type);
  // (the properties of the values of a piece of a name: those named after it)
  const ownKeys = (properties: Options, name: string) => Object.keys(properties).filter((key) => key.startsWith(`${name}_`));
  const rename = (i: number, name: string) => {
    setRenaming(null);
    const old = nameOf(pieces[i]);
    if (!name || pieces.some((p, j) => j !== i && nameOf(p) === name)) return;
    draft.edit((s) => {
      const list = asList(s.options.pieces).map((p, j) => (j === i ? { ...p, name } : p));
      const properties = Object.fromEntries(Object.entries(s.properties).map(([key, value]) => [key.startsWith(`${old}_`) ? `${name}${key.slice(old.length)}` : key, value]));
      return { options: { ...s.options, pieces: list }, properties };
    });
  };
  const duplicate = (i: number) => {
    const old = nameOf(pieces[i]);
    let name = `${old}_copy`;
    for (let k = 2; pieces.some((p) => nameOf(p) === name); k++) name = `${old}_copy${k}`;
    draft.edit((s) => {
      const list = asList(s.options.pieces);
      const copies = Object.fromEntries(ownKeys(s.properties, old).map((key) => [`${name}${key.slice(old.length)}`, s.properties[key]]));
      return { options: { ...s.options, pieces: [...list.slice(0, i + 1), { ...list[i], name }, ...list.slice(i + 1)] }, properties: { ...s.properties, ...copies } };
    });
    const after = (set: Set<number>) => new Set([...set].map((j) => (j > i ? j + 1 : j)));
    onOpen(new Set([...after(open), i + 1]));
    onHide(after(hidden));
    onSelect(i + 1);
  };
  const remove = (i: number) => {
    const old = nameOf(pieces[i]);
    draft.edit((s) => {
      const gone = new Set(ownKeys(s.properties, old));
      return {
        options: { ...s.options, pieces: asList(s.options.pieces).filter((_, j) => j !== i) },
        properties: Object.fromEntries(Object.entries(s.properties).filter(([key]) => !gone.has(key))),
      };
    });
    const shift = (set: Set<number>) => new Set([...set].filter((j) => j !== i).map((j) => (j > i ? j - 1 : j)));
    onOpen(shift(open));
    onHide(shift(hidden));
    if (selected === i) onSelect(null);
    else if (selected !== null && selected > i) onSelect(selected - 1);
  };
  // (whether a piece is alone in the picture: every other one hidden)
  const alone = (i: number) => pieces.length > 1 && !hidden.has(i) && pieces.every((_, j) => j === i || hidden.has(j));
  const toggle = (set: Set<number>, i: number) => {
    const next = new Set(set);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    return next;
  };
  const lensSource = typeof options.source_redshift === 'number' ? options.source_redshift : null;
  return (
    <section className="lp-block" id="ln-pieces">
      <h3 className="lp-head">
        Pieces <span className="lp-count">{pieces.length}</span>
        <span className="lp-acts">
          <button type="button" className="lp-icon" title="Unfold every piece" aria-label="Unfold every piece" disabled={pieces.every((_, i) => open.has(i))} onClick={() => onOpen(new Set(pieces.map((_, i) => i)))}>
            <svg viewBox="0 0 16 16"><path d="M4 6l4-4 4 4M4 10l4 4 4-4" /></svg>
          </button>
          <button type="button" className="lp-icon" title="Fold every piece" aria-label="Fold every piece" disabled={pieces.every((_, i) => !open.has(i))} onClick={() => onOpen(new Set())}>
            <svg viewBox="0 0 16 16"><path d="M4 2l4 4 4-4M4 14l4-4 4 4" /></svg>
          </button>
          <DropMenu
            label="Add a piece"
            width={300}
            at="end"
            entries={types.map((t) => ({ label: t.label, description: t.description, onClick: () => add(t.value) }))}
            target={
              <button type="button" className="lp-icon lp-add" title="Add a piece" aria-label="Add a piece">
                <svg viewBox="0 0 16 16"><path d="M8 3v10M3 8h10" /></svg>
              </button>
            }
          />
        </span>
      </h3>
      {pieces.length === 0 && <p className="lp-note">No piece yet: + adds an SIE, a power law, an NFW halo, a shear, a sheet, deflection maps or the mass of a galaxy.</p>}
      <div className="lp-pieces">
        {pieces.map((piece, i) => {
          const own = params.filter((p) => p.owner[1] === i);
          const isOpen = open.has(i);
          const colour = `var(--${PIECE_COLOURS[i % PIECE_COLOURS.length]})`;
          const bad = problems.some((p) => p.path?.[0] === 'pieces' && p.path[1] === i);
          const zs = typeof piece.source_redshift === 'number' ? piece.source_redshift : null;
          return (
            <div key={i} className="lp-piece" data-open={isOpen || undefined} data-selected={i === selected || undefined} data-hidden={hidden.has(i) || undefined} data-bad={bad || undefined} style={{ ['--piece' as string]: colour }}>
              <div
                className="lp-piece-head"
                role="button"
                tabIndex={0}
                aria-expanded={isOpen}
                onClick={() => {
                  // (unfolded, it is selected; folded, it is no longer)
                  onOpen(toggle(open, i));
                  if (isOpen) {
                    if (selected === i) onSelect(null);
                  } else onSelect(i);
                }}
                onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.click()}
              >
                <svg className="lp-chev" viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M6 4l4 4-4 4" />
                </svg>
                <span className="lp-dot" />
                <span className="lp-piece-text">
                  <span className="lp-piece-name">
                    {renaming === i ? (
                      <span onClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
                        <NameText value={nameOf(piece)} required onCommit={(name) => rename(i, name)} onCancel={() => setRenaming(null)} autoFocus />
                      </span>
                    ) : (
                      <b>{nameOf(piece)}</b>
                    )}
                    <span className="lp-piece-type">{titleOf(defs, 'lens_piece', String(piece.type))}</span>
                  </span>
                  {!isOpen && <span className="lp-sum">{piece.type === 'mass' ? `the mass of ${String(piece.galaxy ?? '?')}` : summary(own, properties)}</span>}
                </span>
                {/* (its cell kept when empty: the buttons stay in their columns) */}
                <span className="lp-badge" title={zs === null ? undefined : `Its own source redshift (not the lens's)`}>{zs !== null && `z_s ${zs}`}</span>
                <button
                  type="button"
                  className="lp-icon"
                  title={hidden.has(i) ? 'Show it in the picture' : 'Hide it in the picture (not saved)'}
                  aria-label={hidden.has(i) ? 'Show the piece in the picture' : 'Hide the piece in the picture'}
                  onClick={(event) => {
                    event.stopPropagation();
                    onHide(toggle(hidden, i));
                  }}
                >
                  {hidden.has(i) ? (
                    <svg viewBox="0 0 16 16"><path d="M2 2l12 12" /><path d="M6.2 4c.6-.3 1.2-.5 1.8-.5 4 0 6.5 4.5 6.5 4.5s-.7 1.3-2 2.5M4 5.6C2.4 6.8 1.5 8 1.5 8s2.5 4.5 6.5 4.5c.9 0 1.8-.2 2.6-.6" /></svg>
                  ) : (
                    <svg viewBox="0 0 16 16"><path d="M1.5 8s2.5-4.5 6.5-4.5 6.5 4.5 6.5 4.5-2.5 4.5-6.5 4.5S1.5 8 1.5 8z" /><circle cx="8" cy="8" r="1.8" /></svg>
                  )}
                </button>
                <span onClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
                  <DropMenu
                    label={`The piece ${nameOf(piece)}`}
                    width={240}
                    at="end"
                    // (its rename gives the focus to the field of its name)
                    returnFocus={false}
                    entries={[
                      { label: 'Rename', onClick: () => setRenaming(i) },
                      { label: alone(i) ? 'Show all in the picture' : 'Show it alone in the picture', onClick: () => onHide(alone(i) ? new Set() : new Set(pieces.map((_, j) => j).filter((j) => j !== i))) },
                      { label: 'Duplicate', onClick: () => duplicate(i) },
                      { label: 'Remove', onClick: () => remove(i) },
                    ]}
                    target={
                      <button type="button" className="lp-icon" title="Rename, show alone, duplicate, remove" aria-label="Its actions">
                        <svg viewBox="0 0 16 16"><circle cx="3.5" cy="8" r=".9" /><circle cx="8" cy="8" r=".9" /><circle cx="12.5" cy="8" r=".9" /></svg>
                      </button>
                    }
                  />
                </span>
              </div>
              {isOpen && (
                <div className="lp-piece-body">
                  {piece.type === 'mass' && <p className="lp-note">The mass of the galaxy {String(piece.galaxy ?? '?')}: its parameters are the galaxy's, which its setups lay out. Not drawn here.</p>}
                  {own.map((p) => (
                    <SliderRow key={p.name} param={p} value={properties[p.name] as ParamValue} field={field} open={openRow === p.name} onOpen={() => onOpenRow(p.name)} onChange={(value) => onProperty(p.name, value)} />
                  ))}
                  {canBeQuoted(defs, String(piece.type)) && (
                    <SourceOption
                      own={zs}
                      lens={lensSource}
                      onChange={(v) => setPieces(pieces.map((p, j) => (j === i ? withKey(p, 'source_redshift', v) : p)))}
                    />
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// The source redshift of a piece: its own, or that of the lens (shown
// faint, until one is set); undefined: the lens's again
function SourceOption({ own, lens, onChange }: { own: number | null; lens: number | null; onChange: (value: number | undefined) => void }) {
  const set = (v: number) => onChange(Math.max(0.001, v));
  return (
    <div className="lp-option">
      <div className="lp-title">
        <b>Source redshift</b>
        <small>{own === null ? 'of the lens' : 'its own'}</small>
      </div>
      <NumberInput
        value={own ?? undefined}
        label="The redshift of the source its strength is quoted for (by default, that of the lens)"
        placeholder={lens === null ? 'needed' : String(lens)}
        onChange={set}
        onClear={() => onChange(undefined)}
      />
      <span />
      {own !== null ? (
        <button type="button" className="lp-state" title={lens === null ? "Drop its own: use the lens's" : `Drop its own: use the lens's, z ${lens}`} aria-label="Use the source redshift of the lens" onClick={() => onChange(undefined)}>
          reset
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}

// The values of a piece in a few words, the free ones first and marked
function summary(params: LensParam[], properties: Options) {
  const free = (p: LensParam) => isObject(properties[p.name]);
  const sorted = [...params.filter(free), ...params.filter((p) => !free(p))];
  return sorted.map((p, k) => {
    const short = p.name.slice(p.name.indexOf('_') + 1);
    const text = `${short} ${fmt(numberOf(p, properties[p.name] as ParamValue))}${unitText(p.unit)}`;
    return (
      <span key={p.name}>
        {k > 0 && ' · '}
        {free(p) ? <em>{text}</em> : text}
      </span>
    );
  });
}

// The handles of a piece in the picture (none if hidden): of a shape (its
// centre, and of an Einstein radius, its ellipse with its position angle
// and axis ratio) or of a shear (its strength and angle), by the names of
// its parameters
function handlesOf(piece: Options | undefined, params: LensParam[], values: Record<string, number>, hidden: boolean) {
  if (!piece || hidden) return null;
  const named = (short: string) => params.find((p) => p.name.slice(p.name.indexOf('_') + 1) === short)?.name;
  const keys = { xpos: named('xpos'), ypos: named('ypos'), posa: named('posa'), q: named('q'), theta_e: named('theta_e'), gamma: named('gamma') };
  if (keys.gamma && keys.posa) return { kind: 'shear' as const, keys, values };
  if (keys.xpos && keys.ypos) return { kind: 'shape' as const, keys, values };
  return null;
}

export type Handles = ReturnType<typeof handlesOf>;

// The centres of the pieces not selected nor hidden that have one, a dot
// each in the picture (a click selects its piece)
function centresOf(pieces: Options[], params: LensParam[], values: Record<string, number>, hidden: Set<number>, selected: number | null) {
  return pieces.flatMap((_, i) => {
    if (i === selected || hidden.has(i)) return [];
    const own = params.filter((p) => p.owner[1] === i);
    const x = own.find((p) => p.name.endsWith('_xpos'))?.name;
    const y = own.find((p) => p.name.endsWith('_ypos'))?.name;
    return x && y ? [{ piece: i, x: values[x], y: values[y], colour: `var(--${PIECE_COLOURS[i % PIECE_COLOURS.length]})` }] : [];
  });
}

export type Dots = ReturnType<typeof centresOf>;

// Whether a piece of a type can be quoted for a source: whether it has
// the option source_redshift (see the schema)
function canBeQuoted(defs: Defs | undefined, type: string): boolean {
  if (!defs) return false;
  const member = membersOf(defs.lens_piece ?? {}, defs).find((m) => typeOf(m) === type);
  return member !== undefined && 'source_redshift' in propertiesOf(member);
}

function asList(value: unknown): Options[] {
  return (Array.isArray(value) ? value : value === undefined || value === null ? [] : [value]).filter(isObject);
}

function withKey(options: Options, key: string, value: unknown): Options {
  const { [key]: _, ...rest } = options;
  return value === undefined ? rest : { ...rest, [key]: value };
}
