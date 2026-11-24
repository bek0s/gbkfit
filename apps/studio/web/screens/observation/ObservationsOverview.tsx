// The observations of the project at a glance, the screen of the section
// before any of its blocks is opened: a tab for each kind of block of the
// section (its observations, and the PSFs, LSFs and noise they can share
// by name), each with how many there are, and a table of the blocks of the
// tab shown (see BlockRow), which the page remembers. An observation has
// its kind, its grid (its own: the data a setup pairs it with have
// theirs), and its PSF, LSF and noise in a few words, each with a link to
// its block if it is shared; a shared part has its kind and its size (the
// FWHM of a PSF or an LSF as the server draws it, the level of a noise). A
// sign before a name says the worst of its problems, as the library finds
// them (of an observation, alone), all of them on hover. A click on the
// head of a column sorts by it, as the page remembers.
import { useMemo, type ReactNode } from 'react';
import { useLsfView, useObservationProblems, usePartProblems, usePiece, useProject, usePsfView, useSchema } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { significant, when } from '../../components/format';
import { LinkIcon } from '../../components/icons';
import type { Severity } from '../../components/Problems';
import { titleOf, type Defs, type Options } from '../../components/SchemaForm';
import { sortedKeys, TableTabs, useRowFocus, type RowFocus, type Sort } from '../../components/ui/table';
import { BlockRow, ColumnHeads, useReportRow, useRowValues, type Column, type RowValues } from '../../shell/BlockTable';
import { useRemembered } from '../../shell/remember';
import { ScreenHead } from '../../shell/ScreenHead';
import { kindOf, sectionNamed } from '../../shell/sections';
import { useStudio } from '../../studio';
import { isObject, noiseOf, objectAt, pairOf, textAt } from './model';

// The folders of the section, a tab each
type Folder = 'observations' | 'psfs' | 'lsfs' | 'noise';

const NOISE_TIP = 'Added to mock data';

const TABS: { key: Folder; title: string; adds: string; tip?: string }[] = [
  { key: 'observations', title: 'Observations', adds: 'Add an observation' },
  { key: 'psfs', title: 'PSFs', adds: 'Add a PSF' },
  { key: 'lsfs', title: 'LSFs', adds: 'Add an LSF' },
  { key: 'noise', title: 'Noise', adds: 'Add a noise', tip: NOISE_TIP },
];

const OBSERVATION_COLUMNS: Column[] = [
  { key: 'name', title: 'Observation', by: 'name' },
  { key: 'kind', title: 'Kind', by: 'kind' },
  { key: 'grid', title: 'Grid', by: 'the number of values', numeric: true },
  { key: 'psf', title: 'PSF', by: 'the FWHM of the PSF' },
  { key: 'lsf', title: 'LSF', by: 'the FWHM of the LSF' },
  { key: 'noise', title: 'Noise', by: 'the noise added to mock data' },
  { key: 'modified', title: 'Modified', by: 'when its file last changed', downFirst: true },
];

// The columns of the shared parts of each kind: their size is the FWHM of
// a PSF or an LSF, the level of a noise
const PART_COLUMNS: Record<Exclude<Folder, 'observations'>, Column[]> = {
  psfs: partColumns('PSF', 'Size', 'the FWHM of the PSF'),
  lsfs: partColumns('LSF', 'Width', 'the FWHM of the LSF'),
  noise: partColumns('Noise', 'Level', 'the level of the noise'),
};

function partColumns(title: string, size: string, by: string): Column[] {
  return [
    { key: 'name', title, by: 'name' },
    { key: 'kind', title: 'Kind', by: 'kind' },
    { key: 'size', title: size, by, numeric: true },
    { key: 'modified', title: 'Modified', by: 'when its file last changed', downFirst: true },
  ];
}

export function ObservationsOverview() {
  const { project, startAdding } = useStudio();
  const info = useProject(project).data;
  const pieces = info?.pieces ?? {};
  const defs = useSchema().data?.$defs as Defs | undefined;
  const [shown, setShown] = useRemembered<Folder>('observations:tab', 'observations');
  if (project === null) return null;
  const tab = TABS.find((each) => each.key === shown) ?? TABS[0];
  const names = pieces[tab.key] ?? [];
  const modified = info?.modified[tab.key] ?? {};
  return (
    <section className="screen ov-screen">
      <ScreenHead title={sectionNamed('observations').overview?.title ?? 'Observations'} />
      <div className="ov-scroll">
        <TableTabs
          label="The blocks of the section"
          tabs={TABS.map(({ key, title, tip }) => ({ key, title, tip, count: (pieces[key] ?? []).length }))}
          shown={tab.key}
          onShow={setShown}
          addLabel={tab.adds}
          onAdd={() => startAdding(tab.key)}
        />
        {tab.key === 'observations' ? (
          <ObservationTable project={project} names={names} pieces={pieces} defs={defs} modified={modified} />
        ) : (
          // (one table of each kind: its order and its rows its own)
          <PartTable key={tab.key} project={project} folder={tab.key} names={names} defs={defs} modified={modified} />
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// The observations

function ObservationTable({
  project,
  names,
  pieces,
  defs,
  modified,
}: {
  project: string;
  names: string[];
  // The blocks of the project, by folder
  pieces: Record<string, string[]>;
  defs: Defs | undefined;
  // When the file of each last changed (ISO 8601)
  modified: Record<string, string>;
}) {
  const [sort, setSort] = useRemembered<Sort>('observations:sort', { column: 'name', down: false });
  // (the values of each row, as it reads them)
  const [values, report] = useRowValues();
  const sorted = sortedKeys(names, values, sort);
  const focusOf = useRowFocus(sorted);
  if (names.length === 0) return <p className="ov-empty">The project has no observation yet: how an instrument sees the galaxies.</p>;
  return (
    <table className="ov-table">
      <ColumnHeads columns={OBSERVATION_COLUMNS} sort={sort} onSort={setSort} />
      <tbody>
        {sorted.map((name) => (
          <ObservationRow key={name} project={project} name={name} pieces={pieces} defs={defs} modified={modified[name] ?? null} focus={focusOf(name)} onValues={report} />
        ))}
      </tbody>
    </table>
  );
}

function ObservationRow({
  project,
  name,
  pieces,
  defs,
  modified,
  focus,
  onValues,
}: {
  project: string;
  name: string;
  pieces: Record<string, string[]>;
  defs: Defs | undefined;
  // When its file last changed (ISO 8601; null: not known)
  modified: string | null;
  // How the keyboard reaches it (see useRowFocus)
  focus: RowFocus;
  onValues: (name: string, row: RowValues | null) => void;
}) {
  const options = optionsOf(usePiece(joinPath(project, `observations/${name}.yaml`)).data);
  const title = (family: string, type: unknown) => titleOf(defs, family, String(type));
  // (the path of the block of the project a part names; null: none, or
  // given in place)
  const pathOf = (folder: string, value: unknown) => (typeof value === 'string' && (pieces[folder] ?? []).includes(value) ? joinPath(project, `${folder}/${value}.yaml`) : null);

  // (its PSF, LSF and noise: of the project, or given in place; its LSF at
  // the rest of the observation, which its widths of wavelength or
  // frequency need)
  const instrument = options ? objectAt(options, 'instrument') : {};
  const psfPiece = optionsOf(usePiece(pathOf('psfs', instrument.psf)).data);
  const lsfPiece = optionsOf(usePiece(pathOf('lsfs', instrument.lsf)).data);
  const noisePiece = optionsOf(usePiece(pathOf('noise', options?.noise)).data);
  const psfOptions = isObject(instrument.psf) ? instrument.psf : psfPiece;
  const lsfOptions = isObject(instrument.lsf) ? instrument.lsf : lsfPiece;
  const noiseOptions = isObject(options?.noise) ? options.noise : noisePiece;
  const rest = textAt(objectAt(options ?? {}, 'spectral'), 'rest');
  const psf = usePartSize(project, 'psf', psfOptions, null);
  const lsf = usePartSize(project, 'lsf', lsfOptions, rest);

  const reading = options === undefined;
  const dash = <span className="ov-faint">{reading ? '…' : '—'}</span>;
  const kindTitle = options ? title('observation', options.type) : null;
  const grid = gridOf(options);
  const psfText = partText(instrument.psf, psfOptions, psf, (type) => title('psf', type), pathOf('psfs', instrument.psf) !== null);
  const lsfText = partText(instrument.lsf, lsfOptions, lsf, (type) => title('lsf', type), pathOf('lsfs', instrument.lsf) !== null);
  const noiseText = noiseOptions ? noiseOf(noiseOptions, title) : null;
  const problems = problemsOf(useObservationProblems(project, options, [psfPiece, lsfPiece, noisePiece]));

  // (what the row sorts by)
  const time = modified === null ? null : Date.parse(modified);
  const row = useMemo<RowValues>(
    () => ({ name, kind: kindTitle, grid: grid?.count ?? null, psf: psf.value, lsf: lsf.value, noise: noiseText, modified: time }),
    [name, kindTitle, grid?.count, psf.value, lsf.value, noiseText, time],
  );
  useReportRow(name, row, onValues);
  return (
    <BlockRow project={project} folder="observations" name={name} focus={focus} problems={problems}>
      <td>{kindTitle ?? dash}</td>
      <td className="ov-num">{grid?.text ?? dash}</td>
      <PartCell folder="psfs" value={instrument.psf} exists={pathOf('psfs', instrument.psf) !== null} text={psfText ?? dash} tip={psf.error} />
      <PartCell folder="lsfs" value={instrument.lsf} exists={pathOf('lsfs', instrument.lsf) !== null} text={lsfText ?? dash} tip={lsf.error} />
      <PartCell
        folder="noise"
        value={options?.noise}
        exists={pathOf('noise', options?.noise) !== null}
        text={noiseText ?? dash}
      />
      <td className="ov-time">{modified === null ? dash : when(modified)}</td>
    </BlockRow>
  );
}

// A PSF, LSF or noise of an observation: what it is in a few words, and,
// if it is shared, a link to its block
// (of one it names that the project does not have, the name alone)
function PartCell({ folder, value, exists, text, tip }: { folder: string; value: unknown; exists: boolean; text: ReactNode; tip?: string | null }) {
  return (
    <td title={tip ?? undefined}>
      <span className="ov-given">
        {(typeof value !== 'string' || exists) && text}
        {typeof value === 'string' && <BlockLink folder={folder} name={value} exists={exists} />}
      </span>
    </td>
  );
}

// A link in a row to a block of the project it names (not a click on the
// row), as a chip with a sign of a link; its name alone if the project has
// none of that name
function BlockLink({ folder, name, exists }: { folder: string; name: string; exists: boolean }) {
  const { go } = useStudio();
  const kind = kindOf(folder).toLowerCase();
  if (!exists) {
    return (
      <span className="ov-shared" data-missing title={`The project has no ${kind} ${name}`}>
        <LinkIcon />
        {name}
      </span>
    );
  }
  return (
    <button
      type="button"
      // (not a stop of Tab: the row is)
      tabIndex={-1}
      className="ov-shared"
      title={`Shared: open the ${kind} ${name}`}
      onClick={(event) => {
        event.stopPropagation();
        go({ kind: 'block', folder, name });
      }}
    >
      <LinkIcon />
      {name}
    </button>
  );
}

// A PSF or an LSF of an observation in a few words, its kind and its size
// (e.g. 'Gaussian 3.53″'), of its options (of the project, if it has the
// block it names, or in place); '…' while they are read, none if it has
// none or they cannot be
function partText(value: unknown, options: Options | undefined, size: Size, titleOf: (type: unknown) => string, exists: boolean): ReactNode | null {
  if (value == null || (typeof value === 'string' && !exists)) return null;
  if (options === undefined) return <span className="ov-faint">…</span>;
  return `${titleOf(options.type)}${size.text ? ` ${size.text}` : ''}`;
}

// The grid of an observation in a few words (e.g. '114 × 114 · 115': its
// pixels, and its channels if it has them), and its number of values,
// which its column sorts by: its size, and the number of its channels if
// given; none if it has none
function gridOf(options: Options | undefined): { text: string; count: number } | null {
  if (!options || options.size == null) return null;
  const channels = objectAt(options, 'spectral').size;
  return gridText(typeof channels === 'number' ? [...pairOf(options.size), channels] : pairOf(options.size));
}

function gridText(shape: number[]): { text: string; count: number } {
  const [x, y, channels] = shape;
  return { text: `${[x, y].filter((n) => n !== undefined).join(' × ')}${channels !== undefined ? ` · ${channels}` : ''}`, count: shape.reduce((a, b) => a * b, 1) };
}

// ---------------------------------------------------------------------------
// The shared parts: PSFs, LSFs and noise

function PartTable({
  project,
  folder,
  names,
  defs,
  modified,
}: {
  project: string;
  folder: Exclude<Folder, 'observations'>;
  names: string[];
  defs: Defs | undefined;
  modified: Record<string, string>;
}) {
  const [sort, setSort] = useRemembered<Sort>(`${folder}:sort`, { column: 'name', down: false });
  const [values, report] = useRowValues();
  const sorted = sortedKeys(names, values, sort);
  const focusOf = useRowFocus(sorted);
  if (names.length === 0) return <p className="ov-empty">None shared: each observation gives its own.</p>;
  return (
    <table className="ov-table">
      <ColumnHeads columns={PART_COLUMNS[folder]} sort={sort} onSort={setSort} />
      <tbody>
        {sorted.map((name) => (
          <PartRow key={name} project={project} folder={folder} name={name} defs={defs} modified={modified[name] ?? null} focus={focusOf(name)} onValues={report} />
        ))}
      </tbody>
    </table>
  );
}

function PartRow({
  project,
  folder,
  name,
  defs,
  modified,
  focus,
  onValues,
}: {
  project: string;
  folder: Exclude<Folder, 'observations'>;
  name: string;
  defs: Defs | undefined;
  modified: string | null;
  focus: RowFocus;
  onValues: (name: string, row: RowValues | null) => void;
}) {
  const options = optionsOf(usePiece(joinPath(project, `${folder}/${name}.yaml`)).data);
  const family = folder === 'psfs' ? 'psf' : folder === 'lsfs' ? 'lsf' : 'noise';
  // (of an LSF, as its widths give it: at no rest, which those of
  // wavelength or frequency cannot be drawn at)
  const drawn = usePartSize(project, family === 'noise' ? null : family, options, null);
  const size = family === 'noise' ? noiseLevel(options) : drawn;
  const reading = options === undefined;
  const dash = <span className="ov-faint">{reading ? '…' : '—'}</span>;
  const problems = problemsOf(usePartProblems(project, family, options));
  const kindTitle = options && typeof options.type === 'string' ? titleOf(defs, family, options.type) : null;
  const time = modified === null ? null : Date.parse(modified);
  const row = useMemo<RowValues>(() => ({ name, kind: kindTitle, size: size.value, modified: time }), [name, kindTitle, size.value, time]);
  useReportRow(name, row, onValues);
  return (
    <BlockRow project={project} folder={folder} name={name} focus={focus} problems={problems}>
      <td>{kindTitle ?? dash}</td>
      <td className="ov-num" title={size.error ?? undefined}>
        {size.text ?? dash}
      </td>
      <td className="ov-time">{modified === null ? dash : when(modified)}</td>
    </BlockRow>
  );
}

// The problems that the library finds, once it has said them (none while
// it is to say them, or if it cannot)
function problemsOf(described: { data?: { severity: Severity; message: string }[]; isSuccess: boolean; isPlaceholderData: boolean }): { severity: Severity; message: string }[] {
  return described.isSuccess && !described.isPlaceholderData ? (described.data ?? NO_PROBLEMS) : NO_PROBLEMS;
}

const NO_PROBLEMS: { severity: Severity; message: string }[] = [];

// ---------------------------------------------------------------------------
// Sizes

// The size of a PSF, an LSF or a noise in a few words, its value (what its
// column sorts by), and why it cannot be known (all null: not known yet,
// or none)
interface Size {
  text: string | null;
  value: number | null;
  error: string | null;
}

// The size of a PSF or an LSF of its options (none: not read yet; of a
// family of none, nothing), as the server draws it: of a PSF, its FWHM
// (arcsec: one number if round, else its axes; it sorts by the FWHM of
// its area); of an LSF, its FWHM in the unit of its axis (at a rest, if
// given)
function usePartSize(project: string, family: 'psf' | 'lsf' | null, options: Options | undefined, rest: string | null): Size {
  const drawable = options !== undefined && typeof options.type === 'string';
  const psf = usePsfView(project, options ?? {}, null, drawable && family === 'psf');
  const lsf = useLsfView(project, options ?? {}, { step: null, rest, unit: null, channel: null }, drawable && family === 'lsf');
  if (family === 'psf' && drawable) {
    const [major, minor] = psf.data?.axes ?? [];
    if (typeof major === 'number' && typeof minor === 'number') {
      const round = Math.abs(major - minor) <= 0.01 * major;
      return { text: round ? `${significant(major, 3)}″` : `${significant(major, 3)}″ × ${significant(minor, 3)}″`, value: Math.sqrt(major * minor), error: null };
    }
    return { text: null, value: null, error: psf.error?.message ?? null };
  }
  if (family === 'lsf' && drawable) {
    const fwhm = lsf.data?.fwhm;
    if (typeof fwhm === 'number') return { text: `${significant(fwhm, 3)} ${lsf.data?.unit}`, value: fwhm, error: null };
    return { text: null, value: null, error: lsf.error?.message ?? null };
  }
  return NO_SIZE;
}

const NO_SIZE: Size = { text: null, value: null, error: null };

// The level of a noise of its options: the dispersion it is of, or that it
// is of an error map
function noiseLevel(options: Options | undefined): Size {
  if (typeof options?.rms === 'number') return { text: significant(options.rms, 3), value: options.rms, error: null };
  if (options?.error != null) return { text: 'error map', value: null, error: null };
  return NO_SIZE;
}

// The options of a piece read (undefined: not read yet, or not options)
function optionsOf(piece: unknown): Options | undefined {
  return isObject(piece) ? piece : undefined;
}
