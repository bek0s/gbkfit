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
import { fmt } from '../../components/format';
import { NeedsNote } from '../../components/ui/dock';
import { AddMenu, FoldAll, PieceHead } from '../../components/ui/pieces';
import { NumberInput, Row, useScrub } from '../../components/ui/rows';
import { SheetLayout } from '../../components/ui/sheet';
import { settled, snap } from '../../components/values';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useStudio } from '../../studio';
import { problemText } from '../galaxy/Model';
import { unitText } from '../galaxy/ParamField';
import { typesOf } from '../galaxy/schema';
import { isObject, useLensDraft, type LensDraft } from './draft';
import { LightPath } from './LightPath';
import { bareView, LensPicture, zoomed, type PictureNote, type Source } from './Picture';
import { naturalSpan, numberOf, SliderRow, type LensParam, type ParamValue } from './SliderRow';

// The colours of the pieces, in order (those of the components of a
// galaxy)
const PIECE_COLOURS = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat1'];

// What the screen shows of the pieces, by their names (not saved; kept as
// pieces are added, removed or renamed, and through undo): those unfolded
// (null: the first, as the screen opens), those hidden in the picture,
// the one selected (its handles in the picture), and the parameter whose
// row shows its slider
interface PiecesView {
  open: Set<string> | null;
  hidden: Set<string>;
  selected: string | null;
  row: string | null;
}

type Problem = { message: string; path?: (string | number)[] };

export function LensScreen({ name }: { name: string }) {
  const { project } = useStudio();
  const root = project ?? '.';
  const draft = useLensDraft(root, name);
  const head = useBlockHead(root, 'lenses', name);
  const options = draft.state?.options;
  const properties = useMemo(() => draft.state?.properties ?? {}, [draft.state]);
  const pieces = asList(options?.pieces);
  const description = useLensDescription(root, options);
  // (the description of the options as they are: none while it comes; its
  // problems, or those of the options before, while it comes)
  const current = description.data && !description.isPlaceholderData ? description.data : undefined;
  const problems = (description.data?.problems ?? []) as Problem[];
  const badPieces = new Set(problems.flatMap((p) => (p.path?.[0] === 'pieces' && typeof p.path[1] === 'number' ? [nameOf(asList(description.data?.of.pieces)[p.path[1]] ?? {})] : [])));
  // (the parameters of each piece: of the last description of no problem,
  // so that the rows stay while the next comes and while the lens has a
  // problem; see paramsByPiece)
  const sound = useLast(current && current.problems.length === 0 ? current : undefined);
  const byPiece = useMemo(() => paramsByPiece(pieces, sound), [pieces, sound]);
  const params = useMemo(() => [...byPiece.values()].flat(), [byPiece]);
  const [ui, setUi] = useState<PiecesView>({ open: null, hidden: new Set(), selected: null, row: null });
  const [source, setSource] = useState<Source>({ x: 0.1, y: 0.05, size: 0.1 });
  // (the field of the picture: null until the server fits one to the lens)
  const [frame, setFrame] = useState<LensFrame | null>(null);
  // (whether no piece is shown, none yet or every one hidden: the picture
  // shows the source as it is, of no call to the server)
  const bare = pieces.length === 0 || (problems.length === 0 && pieces.every((p) => ui.hidden.has(nameOf(p))));
  // (what the picture is of: the lens as it is described, its hidden
  // pieces left out by the server (the lens still the whole of it), at the
  // values of the properties of its parameters, or their defaults; asked
  // for once they stand a moment, not at each step of a drag)
  const shown = useMemo(() => {
    if (!current || current.problems.length > 0 || bare) return null;
    const described = current.parameters as unknown as LensParam[];
    const values = Object.fromEntries(described.map((p) => [p.name, properties[p.name] !== undefined ? properties[p.name] : p.default]));
    const hidden = asList(current.of.pieces).flatMap((p, i) => (ui.hidden.has(nameOf(p)) ? [i] : []));
    return { lens: current.of, values, hidden };
  }, [current, bare, properties, ui.hidden]);
  const [asked] = useDebouncedValue(shown, 80);
  const view = useLensView(root, asked?.lens, asked?.values ?? {}, frame, asked?.hidden ?? []);
  // (what the picture draws: the view of what is shown, else the last it
  // drew, while the next comes or while the lens cannot be drawn; and a
  // note of why, if it is not the lens as it is)
  const bareShown = useMemo(() => (bare ? bareView(frame) : undefined), [bare, frame]);
  const drawn = useLast(bareShown ?? (asked && !view.isError ? view.data : undefined));
  const reason: PictureNote | null =
    pieces.length === 0
      ? { text: 'No piece yet: the source as it is', bad: false }
      : problems.length > 0
        ? { text: 'The lens has a problem (see the head)', bad: true }
        : view.isError && !bare
          ? { text: problemText(view.error.message), bad: true }
          : bare
            ? { text: 'Every piece hidden: the source as it is', bad: false }
            : drawn && drawn.left_out.length > 0
              ? { text: `Not drawn: ${drawn.left_out.join(', ')} (the mass of a galaxy, which its setups lay out)`, bad: false }
              : null;
  const note = reason?.bad && drawn ? { ...reason, text: `The last picture: ${reason.text.charAt(0).toLowerCase()}${reason.text.slice(1)}` } : reason;
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
  const values = Object.fromEntries(params.map((p) => [p.name, numberOf(p, properties[p.name] as ParamValue)]));
  const setProperty = (key: string, value: unknown) => draft.edit((s) => ({ ...s, properties: withKey(s.properties, key, value) }));
  // (a value set by a handle, as by its row: settled, rounded as a drag of
  // its slider is; of a free parameter, its value in its range)
  const setValue = (key: string, value: number) => {
    const param = params.find((p) => p.name === key);
    if (!param) return;
    const v = snap(settled(value, param), naturalSpan(param, value, field));
    draft.edit((s) => {
      const now = s.properties[key];
      return { ...s, properties: withKey(s.properties, key, isObject(now) ? { ...now, value: v } : v) };
    });
  };
  const selected = ui.selected !== null && !ui.hidden.has(ui.selected) ? ui.selected : null;
  return (
    <section className="screen">
      <ScreenHead up="lenses" title={head.title}>
        <NeedsNote needs={problems.map((p) => ({ text: problemText(p.message), at: p.path?.[0] === 'pieces' ? 'ln-pieces' : 'ln-lens' }))} />
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
              <LensBlock draft={draft} options={options} viewSource={drawn?.source_redshift ?? null} />
              <PiecesBlock draft={draft} options={options} pieces={pieces} byPiece={byPiece} properties={properties} badPieces={badPieces} field={field} ui={ui} onUi={setUi} onProperty={setProperty} />
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
              handles={selected === null ? null : handlesOf(byPiece.get(selected), values)}
              dots={centresOf(pieces, byPiece, values, ui.hidden, selected)}
              onPick={(piece) => setUi((u) => ({ ...u, selected: piece, open: new Set([...(u.open ?? firstOpen(pieces)), piece]) }))}
              onClearPick={() => setUi((u) => ({ ...u, selected: null }))}
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
// as lens models quote it), of the pieces without one of their own; the
// path shows the source the picture is of (that of the lens, else of its
// first quoted piece: viewSource)
function LensBlock({ draft, options, viewSource }: { draft: LensDraft; options: Options; viewSource: number | null }) {
  const set = (key: string, value: number | undefined) => draft.edit((s) => ({ ...s, options: withKey(s.options, key, value) }));
  const z = typeof options.redshift === 'number' ? options.redshift : undefined;
  const zs = typeof options.source_redshift === 'number' ? options.source_redshift : undefined;
  const positive = (v: number) => Number.isFinite(v) && v > 0;
  const setZ = (v: number) => positive(v) && set('redshift', v);
  const setZs = (v: number) => positive(v) && set('source_redshift', v);
  const scrubZ = useScrub(z, [0, 2], setZ);
  const scrubZs = useScrub(zs, [0, 4], setZs);
  return (
    <section className="sh-block" id="ln-lens">
      <h3 className="sh-head">Lens</h3>
      <LightPath lens={z} source={zs ?? viewSource ?? undefined} />
      <div className="sh-rows">
        <Row
          title="Lens redshift"
          titleProps={scrubZ}
          box={<NumberInput value={z} label="The redshift of the lens" placeholder="needed" valid={positive} onChange={setZ} onClear={() => set('redshift', undefined)} />}
        />
        <Row
          title="Source redshift"
          titleProps={{ ...scrubZs, title: 'The redshift of the source its quoted strengths (e.g. an Einstein radius) are those of, of the pieces without their own (drag sideways to change it)' }}
          box={<NumberInput value={zs} label="The source redshift of the lens" placeholder="none" valid={positive} onChange={setZs} onClear={() => set('source_redshift', undefined)} />}
        />
      </div>
    </section>
  );
}

// Its pieces: a line each, which a click selects (its handles in the
// picture; unfolded, if it was folded) or, if it is selected, selects no
// longer; its chevron folds or unfolds it, and only that; unfolded, its
// parameters and its own source redshift; its menu renames it, shows it alone in the
// picture, duplicates it or removes it, the properties of its parameters
// with it (moved, copied or dropped, by the names the library gives them);
// what adds a piece of a type, and unfolds or folds them all
function PiecesBlock({
  draft,
  options,
  pieces,
  byPiece,
  properties,
  badPieces,
  field,
  ui,
  onUi,
  onProperty,
}: {
  draft: LensDraft;
  options: Options;
  pieces: Options[];
  byPiece: Map<string, LensParam[]>;
  properties: Options;
  badPieces: Set<string>;
  field: number;
  ui: PiecesView;
  onUi: (change: (ui: PiecesView) => PiecesView) => void;
  onProperty: (key: string, value: unknown) => void;
}) {
  const defs = draft.defs;
  const types = typesOf(defs, 'lens_piece');
  const names = pieces.map(nameOf);
  const open = ui.open ?? firstOpen(pieces);
  const freeName = (base: string, more: (k: number) => string) => {
    let name = base;
    for (let k = 2; names.includes(name); k++) name = more(k);
    return name;
  };
  // (what the screen shows of the pieces, with a name replaced, or gone)
  const renamed = (u: PiecesView, from: string, to: string | null): PiecesView => {
    const swap = (set: Set<string>) => new Set([...set].flatMap((n) => (n !== from ? [n] : to === null ? [] : [to])));
    return { ...u, open: swap(u.open ?? firstOpen(pieces)), hidden: swap(u.hidden), selected: u.selected === from ? to : u.selected };
  };
  const add = (type: string) => {
    const name = freeName(type, (k) => `${type}${k}`);
    draft.edit((s) => ({ ...s, options: { ...s.options, pieces: [...asList(s.options.pieces), name === type ? { type } : { type, name }] } }));
    onUi((u) => ({ ...u, open: new Set([...(u.open ?? firstOpen(pieces)), name]), selected: name }));
  };
  const rename = (from: string, to: string) => {
    if (!to || to === from || names.includes(to)) return;
    const moves = new Map((byPiece.get(from) ?? []).map((p) => [p.name, paramName(to, p.key)]));
    draft.edit((s) => ({
      options: { ...s.options, pieces: asList(s.options.pieces).map((p) => (nameOf(p) === from ? { ...p, name: to } : p)) },
      properties: Object.fromEntries(Object.entries(s.properties).map(([key, value]) => [moves.get(key) ?? key, value])),
    }));
    onUi((u) => renamed(u, from, to));
  };
  const duplicate = (from: string) => {
    const copy = freeName(`${from}_copy`, (k) => `${from}_copy${k}`);
    const own = byPiece.get(from) ?? [];
    draft.edit((s) => {
      const list = asList(s.options.pieces);
      const at = list.findIndex((p) => nameOf(p) === from);
      const copies = Object.fromEntries(own.filter((p) => s.properties[p.name] !== undefined).map((p) => [paramName(copy, p.key), s.properties[p.name]]));
      return { options: { ...s.options, pieces: [...list.slice(0, at + 1), { ...list[at], name: copy }, ...list.slice(at + 1)] }, properties: { ...s.properties, ...copies } };
    });
    onUi((u) => ({ ...u, open: new Set([...(u.open ?? firstOpen(pieces)), copy]), selected: copy }));
  };
  const remove = (gone: string) => {
    const dropped = new Set((byPiece.get(gone) ?? []).map((p) => p.name));
    draft.edit((s) => ({
      options: { ...s.options, pieces: asList(s.options.pieces).filter((p) => nameOf(p) !== gone) },
      properties: Object.fromEntries(Object.entries(s.properties).filter(([key]) => !dropped.has(key))),
    }));
    onUi((u) => renamed(u, gone, null));
  };
  const toggled = (set: Set<string>, name: string) => (set.has(name) ? new Set([...set].filter((n) => n !== name)) : new Set([...set, name]));
  // (whether a piece is alone in the picture: every other one hidden)
  const alone = (name: string) => names.length > 1 && !ui.hidden.has(name) && names.every((n) => n === name || ui.hidden.has(n));
  const lensSource = typeof options.source_redshift === 'number' ? options.source_redshift : null;
  return (
    <section className="sh-block" id="ln-pieces">
      <h3 className="sh-head">
        Pieces <span className="sh-count">{pieces.length}</span>
        <span className="sh-acts">
          <FoldAll
            what="piece"
            canUnfold={!names.every((n) => open.has(n))}
            canFold={!names.every((n) => !open.has(n))}
            onUnfold={() => onUi((u) => ({ ...u, open: new Set(names) }))}
            onFold={() => onUi((u) => ({ ...u, open: new Set() }))}
          />
          <AddMenu label="Add a piece" entries={types.map((t) => ({ label: t.label, description: t.description, onClick: () => add(t.value) }))} />
        </span>
      </h3>
      {pieces.length === 0 && <p className="sh-note">No piece yet: + adds an SIE, a power law, an NFW halo, a shear, a sheet, deflection maps or the mass of a galaxy.</p>}
      <div className="sh-pieces">
        {pieces.map((piece, i) => {
          const name = names[i];
          const own = byPiece.get(name) ?? [];
          const isOpen = open.has(name);
          const isHidden = ui.hidden.has(name);
          const zs = typeof piece.source_redshift === 'number' ? piece.source_redshift : null;
          return (
            <div key={i} className="sh-piece" data-open={isOpen || undefined} data-selected={name === ui.selected || undefined} data-hidden={isHidden || undefined} data-bad={badPieces.has(name) || undefined} style={{ ['--piece' as string]: colourOf(i) }}>
              <PieceHead
                kind="piece"
                name={name}
                type={titleOf(defs, 'lens_piece', String(piece.type))}
                summary={piece.type === 'mass' ? `the mass of ${String(piece.galaxy ?? '?')}` : summary(own, properties)}
                badge={zs !== null && `z_s ${zText(zs)}`}
                badgeTitle={zs === null ? undefined : `Its own source redshift (not the lens's)`}
                open={isOpen}
                selected={name === ui.selected}
                onFold={() => onUi((u) => ({ ...u, open: toggled(u.open ?? firstOpen(pieces), name) }))}
                onSelect={() =>
                  // (selected, and unfolded; or, if it was, selected no longer)
                  onUi((u) => (u.selected === name ? { ...u, selected: null } : { ...u, selected: name, open: new Set([...(u.open ?? firstOpen(pieces)), name]) }))
                }
                onRename={(to) => rename(name, to)}
                eye={
                  <button
                    type="button"
                    className="sh-icon"
                    title={isHidden ? 'Show it in the picture' : 'Hide it in the picture (not saved)'}
                    aria-label={isHidden ? 'Show the piece in the picture' : 'Hide the piece in the picture'}
                    onClick={(event) => {
                      event.stopPropagation();
                      onUi((u) => ({ ...u, hidden: toggled(u.hidden, name) }));
                    }}
                  >
                    {isHidden ? (
                      <svg viewBox="0 0 16 16"><path d="M2 2l12 12" /><path d="M6.2 4c.6-.3 1.2-.5 1.8-.5 4 0 6.5 4.5 6.5 4.5s-.7 1.3-2 2.5M4 5.6C2.4 6.8 1.5 8 1.5 8s2.5 4.5 6.5 4.5c.9 0 1.8-.2 2.6-.6" /></svg>
                    ) : (
                      <svg viewBox="0 0 16 16"><path d="M1.5 8s2.5-4.5 6.5-4.5 6.5 4.5 6.5 4.5-2.5 4.5-6.5 4.5S1.5 8 1.5 8z" /><circle cx="8" cy="8" r="1.8" /></svg>
                    )}
                  </button>
                }
                menu={[
                  { label: alone(name) ? 'Show all in the picture' : 'Show it alone in the picture', onClick: () => onUi((u) => ({ ...u, hidden: alone(name) ? new Set() : new Set(names.filter((n) => n !== name)) })) },
                  { label: 'Duplicate', onClick: () => duplicate(name) },
                  { label: 'Remove', onClick: () => remove(name) },
                ]}
                menuTitle="Rename, show alone, duplicate, remove"
              />
              {isOpen && (
                <div className="sh-piece-body">
                  {piece.type === 'mass' && <p className="sh-note">The mass of the galaxy {String(piece.galaxy ?? '?')}: its parameters are the galaxy's, which its setups lay out. Not drawn here.</p>}
                  {own.map((p) => (
                    <SliderRow key={p.name} param={p} value={properties[p.name] as ParamValue} field={field} open={ui.row === p.name} onOpen={() => onUi((u) => ({ ...u, row: u.row === p.name ? null : p.name }))} onChange={(value) => onProperty(p.name, value)} />
                  ))}
                  {canBeQuoted(defs, String(piece.type)) && (
                    <SourceOption
                      own={zs}
                      lens={lensSource}
                      onChange={(v) => draft.edit((s) => ({ ...s, options: { ...s.options, pieces: asList(s.options.pieces).map((p) => (nameOf(p) === name ? withKey(p, 'source_redshift', v) : p)) } }))}
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
  return (
    <Row
      className="lp-option"
      title="Source redshift"
      sub={own === null ? 'of the lens' : 'its own'}
      box={
        <NumberInput
          value={own ?? undefined}
          label="The redshift of the source its strength is quoted for (by default, that of the lens)"
          placeholder={lens === null ? 'needed' : zText(lens)}
          valid={(v) => Number.isFinite(v) && v > 0}
          onChange={onChange}
          onClear={() => onChange(undefined)}
        />
      }
      pill={
        own !== null && (
          <button type="button" className="sh-state" title={lens === null ? "Drop its own: use the lens's" : `Drop its own: use the lens's, z ${zText(lens)}`} aria-label="Use the source redshift of the lens" onClick={() => onChange(undefined)}>
            reset
          </button>
        )
      }
    />
  );
}

// The values of a piece in a few words, the free ones first and marked
function summary(params: LensParam[], properties: Options) {
  const free = (p: LensParam) => isObject(properties[p.name]);
  const sorted = [...params.filter(free), ...params.filter((p) => !free(p))];
  return sorted.map((p, k) => {
    const text = `${p.key} ${fmt(numberOf(p, properties[p.name] as ParamValue))}${unitText(p.unit)}`;
    return (
      <span key={p.name}>
        {k > 0 && ' · '}
        {free(p) ? <em>{text}</em> : text}
      </span>
    );
  });
}

// The handles of a piece in the picture: of a shape (its centre, and of an
// Einstein radius, its ellipse with its position angle and axis ratio) or
// of a shear (its strength and angle), by the keys of its parameters
function handlesOf(params: LensParam[] | undefined, values: Record<string, number>) {
  if (!params) return null;
  const named = (key: string) => params.find((p) => p.key === key)?.name;
  const keys = { xpos: named('xpos'), ypos: named('ypos'), posa: named('posa'), q: named('q'), theta_e: named('theta_e'), gamma: named('gamma') };
  if (keys.gamma && keys.posa) return { kind: 'shear' as const, keys, values };
  if (keys.xpos && keys.ypos) return { kind: 'shape' as const, keys, values };
  return null;
}

export type Handles = ReturnType<typeof handlesOf>;

// The centres of the pieces not selected nor hidden that have one, a dot
// each in the picture (a click selects its piece)
function centresOf(pieces: Options[], byPiece: Map<string, LensParam[]>, values: Record<string, number>, hidden: Set<string>, selected: string | null) {
  return pieces.flatMap((piece, i) => {
    const name = nameOf(piece);
    if (name === selected || hidden.has(name)) return [];
    const own = byPiece.get(name) ?? [];
    const x = own.find((p) => p.key === 'xpos')?.name;
    const y = own.find((p) => p.key === 'ypos')?.name;
    return x && y ? [{ piece: name, x: values[x], y: values[y], colour: colourOf(i) }] : [];
  });
}

export type Dots = ReturnType<typeof centresOf>;

// The parameters of each piece, by its name, of a description of the lens
// (of its options as they are, or as they were before an edit while the
// description of the next comes): matched to the pieces by name, or by
// place for a piece renamed since; each named for its piece as it is
// (see paramName)
function paramsByPiece(pieces: Options[], described: { parameters: unknown[]; of: Options } | undefined): Map<string, LensParam[]> {
  const before = asList(described?.of.pieces).map(nameOf);
  const now = pieces.map(nameOf);
  const all = (described?.parameters ?? []) as LensParam[];
  return new Map(
    pieces.map((piece, i) => {
      const name = nameOf(piece);
      const at = before.includes(name) ? before.indexOf(name) : i < before.length && !now.includes(before[i]) ? i : -1;
      return [name, all.filter((p) => p.owner[1] === at).map((p) => ({ ...p, name: paramName(name, p.key) }))];
    }),
  );
}

// The name of a parameter of a piece in its lens (as the library names
// them: <piece>_<key>, e.g. main_theta_e; see Lens.pdescs)
function paramName(piece: string, key: string): string {
  return `${piece}_${key}`;
}

// The name of a piece (by default, that of its type)
function nameOf(piece: Options): string {
  return String(piece.name ?? piece.type);
}

// The pieces unfolded as the screen opens: the first
function firstOpen(pieces: Options[]): Set<string> {
  return new Set(pieces.slice(0, 1).map(nameOf));
}

// The colour of the piece at a place
function colourOf(i: number): string {
  return `var(--${PIECE_COLOURS[i % PIECE_COLOURS.length]})`;
}

// A redshift as written (to 6 digits)
function zText(z: number): string {
  return String(Number(z.toPrecision(6)));
}

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
