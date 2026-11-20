// The datasets of the project at a glance, the screen of the section
// before any dataset is opened: one row each (a click opens it), with what
// it is (its kind), its pixels and channels and their size, its beam, its
// errors, the unit of its values, and the observations that see it (none:
// nothing can use it yet). The dataset opened last is one click away, at
// the top.
import { useMemo } from 'react';
import type { components } from '../../api/schema';
import { usePiece, useProject } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { significant } from '../../components/format';
import { Button } from '../../components/ui/basics';
import { ScreenHead } from '../../shell/ScreenHead';
import { sectionNamed } from '../../shell/sections';
import { useStudio } from '../../studio';
import { useResolved } from './hooks';
import { draftOfDataset, KINDS, kindOfType, NO_EDITS, NO_SPECTRAL, type Draft } from './model';

type ArrayInfo = components['schemas']['ArrayInfo'];
type Link = components['schemas']['PieceLink'];

const NOTHING: Draft = { items: {}, spectral: NO_SPECTRAL, edits: NO_EDITS };

export function DatasetsOverview() {
  const { project, startAdding, visited, go } = useStudio();
  const info = useProject(project).data;
  const names = info?.pieces.datasets ?? [];
  const last = visited.find((seen) => seen.kind === 'block' && seen.folder === 'datasets' && names.includes(seen.name));
  if (project === null) return null;
  return (
    <section className="screen ov-screen">
      <ScreenHead title={sectionNamed('data').overview?.title ?? 'Datasets'} note={<span className="ov-count">{names.length === 1 ? '1 dataset' : `${names.length} datasets`}</span>}>
        <Button kind="primary" onClick={() => startAdding('datasets')}>
          + Add data
        </Button>
      </ScreenHead>
      <div className="ov-scroll">
        {last?.kind === 'block' && (
          <button className="ov-last" onClick={() => go(last)}>
            <span>Opened last</span>
            <b>{last.name}</b>
            <span aria-hidden>→</span>
          </button>
        )}
        {names.length === 0 ? (
          <p className="ov-empty">The project has no data yet: a cube or maps from FITS files, prepared and kept beside their originals.</p>
        ) : (
          <table className="ov-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Kind</th>
                <th className="ov-num">Pixels</th>
                <th className="ov-num">Pixel size</th>
                <th className="ov-num">Channels</th>
                <th className="ov-num">Beam</th>
                <th>Errors</th>
                <th>Unit</th>
                <th>Seen by</th>
              </tr>
            </thead>
            <tbody>
              {names.map((name) => (
                <DatasetRow key={name} project={project} name={name} links={info?.links ?? []} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function DatasetRow({ project, name, links }: { project: string; name: string; links: Link[] }) {
  const { go } = useStudio();
  const options = usePiece(joinPath(project, `datasets/${name}.yaml`)).data as Record<string, unknown> | undefined;
  const kind = kindOfType(options?.type);
  const draft = useMemo(() => (kind && options ? draftOfDataset(kind, options, project) : NOTHING), [kind, options, project]);
  const resolved = useResolved(kind ?? 'cube', draft);
  // (of a dataset of several maps, the first)
  const [key, item] = Object.entries(draft.items)[0] ?? [];
  const arrays = key === undefined ? undefined : resolved.arrays[key];
  const array = arrays?.found.data;
  const failed = arrays?.errors.data;
  const seenBy = links.filter((link) => link.source.folder === 'observations' && link.target.folder === 'datasets' && link.target.name === name).map((link) => link.source.name);
  const maps = kind === 'moments' ? Object.keys(draft.items).length : null;
  const reading = options === undefined || (array === undefined && failed === undefined);
  const dash = <span className="ov-faint">{reading ? '…' : '—'}</span>;
  return (
    <tr className="ov-row" onClick={() => go({ kind: 'block', folder: 'datasets', name })}>
      <td className="ov-name">{name}</td>
      <td>
        {kind ? KINDS[kind].title : options ? String(options.type ?? '—') : dash}
        {maps !== null && <span className="ov-faint"> · {maps}</span>}
      </td>
      {failed !== undefined ? (
        <td colSpan={4} className="ov-problem">
          {failed}
        </td>
      ) : (
        <>
          <td className="ov-num">{array ? `${array.shape[0]} × ${array.shape[1]}` : dash}</td>
          <td className="ov-num">{array?.step ? pixelSize(array.step) : dash}</td>
          <td className="ov-num">{array ? (kind === 'cube' ? channels(array) : '—') : dash}</td>
          <td className="ov-num">{array ? beam(array) : dash}</td>
        </>
      )}
      <td>{item ? item.files.error ? 'from a file' : item.errorValue !== null ? item.errorValue : <span className="ov-warn">none</span> : dash}</td>
      <td className="ov-unit">{array?.unit ?? dash}</td>
      <td onClick={(event) => event.stopPropagation()}>
        {seenBy.length === 0 ? (
          <span className="ov-warn">no observation</span>
        ) : (
          <span className="ov-chips">
            {seenBy.map((observation) => (
              <button key={observation} className="ov-chip" title={`Open the observation ${observation}`} onClick={() => go({ kind: 'block', folder: 'observations', name: observation })}>
                {observation}
              </button>
            ))}
          </span>
        )}
      </td>
    </tr>
  );
}

// The size of the pixels on the sky (arcsec): one number if they are
// square
function pixelSize(step: number[]): string {
  const [x, y] = step.map((value) => significant(Math.abs(value), 3));
  return x === y ? `${x}″` : `${x} × ${y}″`;
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
