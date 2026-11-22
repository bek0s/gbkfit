// The datasets of the project at a glance, the screen of the section
// before any dataset is opened: one row each (a click opens it; its menu,
// at its end or of a right click, renames or deletes it), with what it is
// (its kind), its pixels and channels and their size, its beam, the unit
// of its values (of each map of moments, if theirs differ), whether prep
// made its files, and when its file last changed; a sign before its name
// says the worst of its problems (see problemsOf), all of them on hover;
// a long name or unit is cut (all of it on hover). A click on the head of
// a column sorts by it, as the page remembers: a pair of numbers (the
// pixels, their size, the beam) by the area it spans.
import { useMemo } from 'react';
import type { components } from '../../api/schema';
import { usePiece, usePreparation, useProject } from '../../api/queries';
import { significant, when } from '../../components/format';
import { Cut } from '../../components/ui/cut';
import { sortedKeys, TableHeading, useRowFocus, type RowFocus, type Sort } from '../../components/ui/table';
import { joinPath } from '../../components/FilePicker';
import { BlockRow, ColumnHeads, useReportRow, useRowValues, type Column, type RowValues } from '../../shell/BlockTable';
import { useRemembered } from '../../shell/remember';
import { ScreenHead } from '../../shell/ScreenHead';
import { sectionNamed } from '../../shell/sections';
import { useStudio } from '../../studio';
import { useDatasetProblems, useResolved } from './hooks';
import { draftOfDataset, isReading, KINDS, kindOfType, NO_DRAFT, presentSlots, slotTitle } from './model';

type ArrayInfo = components['schemas']['ArrayInfo'];

// The columns, by the key of the value of a row that sorts by each, and
// what that value is
const COLUMNS: Column[] = [
  { key: 'name', title: 'Dataset', by: 'name' },
  { key: 'kind', title: 'Kind', by: 'kind' },
  { key: 'pixels', title: 'Pixels', by: 'the number of pixels', numeric: true },
  { key: 'pixelSize', title: 'Pixel size', by: 'the area of a pixel', numeric: true },
  { key: 'channels', title: 'Channels', by: 'the number of channels', numeric: true },
  { key: 'beam', title: 'Beam', by: 'the area of the beam', numeric: true },
  { key: 'unit', title: 'Unit', by: 'unit' },
  { key: 'prepared', title: 'Prepared', by: 'whether prep made the files' },
  { key: 'modified', title: 'Modified', by: 'when its file last changed', downFirst: true },
];

// The longest unit shown whole (characters)
const UNIT_CAP = 20;

export function DatasetsOverview() {
  const { project, startAdding } = useStudio();
  const info = useProject(project).data;
  const names = info?.pieces.datasets ?? [];
  const [sort, setSort] = useRemembered<Sort>('datasets:sort', { column: 'name', down: false });
  // (the values of each row, as it reads them)
  const [values, report] = useRowValues();
  const sorted = sortedKeys(names, values, sort);
  const focusOf = useRowFocus(sorted);
  if (project === null) return null;
  return (
    <section className="screen ov-screen">
      <ScreenHead title={sectionNamed('data').overview?.title ?? 'Datasets'} />
      <div className="ov-scroll">
        <TableHeading title="Datasets" count={names.length} addLabel="Add a dataset" onAdd={() => startAdding('datasets')} />
        {names.length === 0 ? (
          <p className="ov-empty">The project has no data yet: a cube or maps from FITS files, prepared and kept beside their originals.</p>
        ) : (
          <table className="ov-table">
            <ColumnHeads columns={COLUMNS} sort={sort} onSort={setSort} />
            <tbody>
              {sorted.map((name) => (
                <DatasetRow key={name} project={project} name={name} modified={info?.modified.datasets?.[name] ?? null} focus={focusOf(name)} onValues={report} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function DatasetRow({
  project,
  name,
  modified,
  focus,
  onValues,
}: {
  project: string;
  name: string;
  // When its file last changed (ISO 8601; null: not known)
  modified: string | null;
  // How the keyboard reaches it (see useRowFocus)
  focus: RowFocus;
  onValues: (name: string, row: RowValues | null) => void;
}) {
  const options = usePiece(joinPath(project, `datasets/${name}.yaml`)).data as Record<string, unknown> | undefined;
  const kind = kindOfType(options?.type);
  const draft = useMemo(() => (kind && options ? draftOfDataset(kind, options, project) : NO_DRAFT), [kind, options, project]);
  const resolved = useResolved(kind ?? 'cube', draft);
  // (of a dataset of several maps, the first)
  const [key, item] = Object.entries(draft.items)[0] ?? [];
  const arrays = key === undefined ? undefined : resolved.arrays[key];
  const array = arrays?.found.data;
  const failed = arrays?.errors.data;
  const maps = kind === 'moments' ? Object.keys(draft.items).length : null;
  const reading = options === undefined || (array === undefined && failed === undefined);
  const dash = <span className="ov-faint">{reading ? '…' : '—'}</span>;
  const preparation = usePreparation(item?.files.data?.file ?? null);
  const problems = useDatasetProblems(project, resolved, kind !== undefined && !isReading(resolved));
  // (the unit of the values: of each map, as given or as its header says,
  // where they differ)
  const units = kind
    ? presentSlots(kind, draft).flatMap((slot) => {
        const unit = draft.items[slot.key].unit || resolved.arrays[slot.key]?.found.data?.unit;
        return unit ? [{ map: slotTitle(slot, kind), unit }] : [];
      })
    : [];
  const perMap = new Set(units.map(({ unit }) => unit)).size > 1;
  // (what the row sorts by)
  const kindTitle = kind ? KINDS[kind].title : options ? String(options.type ?? '') : null;
  const unit = perMap ? 'per map' : (units[0]?.unit ?? null);
  const [pixels, pixelSize, channelCount, beamSize] = array ? [array.shape[0] * array.shape[1], pixelSide(array), kind === 'cube' ? array.shape[2] : null, beamSide(array)] : [null, null, null, null];
  const prepared = preparation.isLoading ? null : preparation.data ? 'yes' : 'no';
  const time = modified === null ? null : Date.parse(modified);
  const row = useMemo<RowValues>(
    () => ({ name, kind: kindTitle, pixels, pixelSize, channels: channelCount, beam: beamSize, unit, prepared, modified: time }),
    [name, kindTitle, pixels, pixelSize, channelCount, beamSize, unit, prepared, time],
  );
  useReportRow(name, row, onValues);
  return (
    <BlockRow project={project} folder="datasets" name={name} focus={focus} problems={problems}>
      <td>
        {kindTitle ?? dash}
        {maps !== null && <span className="ov-faint"> · {maps}</span>}
      </td>
      <td className="ov-num">{array ? `${array.shape[0]} × ${array.shape[1]}` : dash}</td>
      <td className="ov-num">{array?.step ? pixelSizeText(array.step) : dash}</td>
      <td className="ov-num">{array ? (kind === 'cube' ? channels(array) : '—') : dash}</td>
      <td className="ov-num">{array ? beam(array) : dash}</td>
      <td className="ov-unit">
        {perMap ? (
          <span title={units.map(({ map, unit }) => `${map}: ${unit}`).join('\n')}>per map</span>
        ) : unit ? (
          <Cut max={UNIT_CAP} title={unit} />
        ) : (
          dash
        )}
      </td>
      <td>{prepared ?? dash}</td>
      <td className="ov-time">{modified === null ? dash : when(modified)}</td>
    </BlockRow>
  );
}

// The side of the square of the area of a pixel on the sky (arcsec; the
// geometric mean of its sides), what the column of their size sorts by
// (null: none known)
function pixelSide(array: ArrayInfo): number | null {
  const [x, y] = array.step ?? [];
  return x === undefined || y === undefined ? null : Math.sqrt(Math.abs(x * y));
}

// The FWHM of the round beam of the area of the beam (arcsec; the
// geometric mean of its axes), what the column of the beam sorts by
// (null: none, or one per channel)
function beamSide(array: ArrayInfo): number | null {
  const { bmaj, bmin } = array.beam ?? {};
  return typeof bmaj === 'number' && typeof bmin === 'number' ? Math.sqrt(bmaj * bmin) : null;
}

// The size of the pixels on the sky (arcsec): one number if they are
// square
function pixelSizeText(step: number[]): string {
  const [x, y] = step.map((value) => significant(Math.abs(value), 3));
  return x === y ? `${x}″` : `${x}″ × ${y}″`;
}

// The channels of a cube, and their width (km/s: the server reads the
// spectral axis as velocities; none while it cannot)
function channels(array: ArrayInfo): string {
  const width = array.step?.[2];
  return width === undefined ? `${array.shape[2]}` : `${array.shape[2]} × ${significant(Math.abs(width), 3)} km/s`;
}

// The beam of the data (FWHM, arcsec): its axes, or that it changes from
// channel to channel; none if its header gives none
function beam(array: ArrayInfo): string {
  const { bmaj, bmin } = array.beam ?? {};
  if (Array.isArray(bmaj) || Array.isArray(bmin)) return 'per channel';
  if (typeof bmaj !== 'number' || typeof bmin !== 'number') return '—';
  return `${significant(bmaj, 3)}″ × ${significant(bmin, 3)}″`;
}
