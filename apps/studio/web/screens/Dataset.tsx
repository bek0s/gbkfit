// A dataset: its cube, to look at (see CubeViewer), and at the side its
// files (its data, their errors and their mask: an array of a file each;
// the errors can be one value for all the data instead) and what GBKFIT
// reads from their headers, with the fixes it takes (axes of length 1 left
// out; the convention of velocities that a header leaves ambiguous). A
// new one is named, given its files and their edits (see Edits), and
// added to the project once nothing stops it: of its files as they are,
// or of the copies that prep edits as it is added. One of the project is
// shown as it is.
import { useEffect, useRef, useState } from 'react';
import { useCreateBlock, useJob, usePiece, useProject, useSubmitJob } from '../api/queries';
import { CubeViewer } from '../components/CubeViewer';
import { joinPath } from '../components/FilePicker';
import type { Region } from '../components/MapView';
import { Button } from '../components/ui/basics';
import { SidePanes } from '../components/ui/panes';
import { useBlockHead } from '../shell/BlockMenu';
import { dataKind, dataKindOfType, type DataKind } from '../shell/blocks';
import { useStudio } from '../studio';
import {
  cropsOf,
  EditsCard,
  keptOf,
  needsPrep,
  NO_EDITS,
  prepOptions,
  withChannels,
  withRegion,
  type Edits,
  type PrepFiles,
  type Prepared,
} from './Edits';
import {
  errorKindOf,
  errorSibling,
  Facts,
  FilesCard,
  itemBlocker,
  itemOptions,
  itemSources,
  kept,
  layersOf,
  NO_SOURCES,
  prepFile,
  useArrays,
  type Arrays,
  type Sources,
} from './items';
import { ImageDataset } from './Image';
import { Moments } from './Moments';

export function NewDataset({ data }: { data: string }) {
  const { project, go } = useStudio();
  const info = useProject(project);
  const create = useCreateBlock(project ?? '.');
  const submit = useSubmitJob();
  const kind = dataKind(data);
  const [name, setName] = useState('');
  const [sources, setSources] = useState<Sources>(NO_SOURCES);
  const [edits, setEdits] = useState<Edits>(NO_EDITS);
  // The job of prep that edits the files as the dataset is added
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId).data;
  const arrays = useArrays(sources);
  const added = useRef<string | null>(null);
  // (the errors of an array of the file of the data, named as errors, are
  // given once for that file; removed, they stay removed)
  const guessed = useRef<string | null>(null);
  useEffect(() => {
    const data = sources.files.data;
    if (data === undefined || arrays.found.data === undefined || guessed.current === data.file) return;
    guessed.current = data.file;
    const errors = errorSibling(arrays);
    if (errors !== undefined && sources.files.error === undefined && sources.errorValue === null) {
      setSources({ ...sources, files: { ...sources.files, error: { file: data.file, array: errors } } });
    }
  }, [sources, arrays]);
  const array = arrays.found.data;
  const prep = project !== null && array !== undefined && needsPrep(edits, array);
  const request = prep ? prepOptions(edits, prepFiles(sources, arrays), errorKindOf(sources, arrays), array, sources.velocity, project) : null;
  const finish = (content: Record<string, unknown>) =>
    create.mutate({ folder: 'datasets', name: name.trim(), content }, { onSuccess: () => go({ kind: 'block', folder: 'datasets', name: name.trim() }) });
  // (the files edited, the dataset of the copies is added)
  useEffect(() => {
    if (kind && project !== null && job?.state === 'complete' && job.files && added.current !== job.id) {
      added.current = job.id;
      finish(preparedBlock(kind, job.files, sources, project));
    }
  });
  if (kind === undefined || project === null) return null;
  const taken = info.data?.pieces.datasets.includes(name.trim()) ?? false;
  const reason =
    blocker(kind, arrays, sources) ??
    (typeof request === 'string' ? request : null) ??
    (name.trim() === '' ? 'Name the dataset' : taken ? 'The project has a dataset of that name' : null);
  const busy = submit.isPending || job?.state === 'queued' || job?.state === 'running' || create.isPending;
  const add = () => {
    if (request === null) finish(toBlock(kind, arrays, sources, project));
    else if (typeof request !== 'string') {
      submit.mutate({ task: 'prep_pixel_spectra', options: request }, { onSuccess: (submitted) => setJobId(submitted.id) });
    }
  };
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">New dataset · {kind.title}</div>
          <input
            className="name-input"
            autoFocus
            value={name}
            placeholder="Name of the dataset"
            aria-label="Name of the dataset"
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        <div className="head-actions">
          {busy ? (
            <span className="note">{create.isPending ? 'Adding…' : 'Writing the edited copy…'}</span>
          ) : (
            reason && <span className="why">{reason}</span>
          )}
          <Button onClick={() => go({ kind: 'home' })}>Cancel</Button>
          <Button kind="primary" disabled={reason !== null || busy} onClick={add}>
            Add to the project
          </Button>
        </div>
      </div>
      {(create.error ?? submit.error) && <p className="error screen-error">{(create.error ?? submit.error)?.message}</p>}
      {job?.state === 'failed' && <pre className="error screen-error">{lastLines(job.output, 4)}</pre>}
      <DatasetBody
        project={project}
        sources={sources}
        arrays={arrays}
        edits={edits}
        onChange={(changed) => {
          setSources(changed);
          // (other files, other edits)
          if (changed.files.data?.file !== sources.files.data?.file) setEdits(NO_EDITS);
        }}
        onEdits={setEdits}
      />
    </section>
  );
}

function lastLines(text: string, count: number): string {
  return text.trimEnd().split('\n').slice(-count).join('\n');
}

// A dataset of the project, as its piece gives it: of a cube, of moment
// maps (see Moments), or of a flux map (see Image)
export function Dataset({ name }: { name: string }) {
  const { project } = useStudio();
  const piece = usePiece(project === null ? null : joinPath(project, `datasets/${name}.yaml`));
  const kind = dataKindOfType(piece.data?.type);
  const head = useBlockHead(project ?? '', 'datasets', name);
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Dataset{kind ? ` · ${kind.title}` : ''}</div>
          {project === null ? <h1>{name}</h1> : head.title}
        </div>
        {project !== null && (
          <div className="head-actions">
            {head.menu}
          </div>
        )}
      </div>
      {piece.error && <p className="error screen-error">{piece.error.message}</p>}
      {project !== null &&
        piece.data !== undefined &&
        (piece.data.type === 'pixel_moments' ? (
          <Moments options={piece.data} project={project} />
        ) : piece.data.type === 'pixel_brightness' ? (
          <ImageDataset options={piece.data} project={project} />
        ) : (
          <CubeDataset options={piece.data} project={project} />
        ))}
    </section>
  );
}

// A dataset of a cube of the project, of its options
function CubeDataset({ options, project }: { options: Record<string, unknown>; project: string }) {
  const sources = fromBlock(options, project);
  const arrays = useArrays(sources);
  return <DatasetBody project={project} sources={sources} arrays={arrays} />;
}

// Why a dataset of these arrays cannot be added yet (null: it can)
function blocker(kind: DataKind, arrays: Arrays, sources: Sources): string | null {
  return itemBlocker(kind.title.toLowerCase(), arrays, sources, 3);
}

// The options of a dataset of a cube: its type, its item, and the
// convention of its velocities
function toBlock(kind: DataKind, arrays: Arrays, sources: Sources, project: string): Record<string, unknown> {
  const options: Record<string, unknown> = { type: kind.type, ...itemOptions(arrays, sources, project) };
  if (sources.velocity !== null) options.spectral = { convention: sources.velocity };
  return options;
}

// The sources of the options of a dataset of a cube: its item, and the
// convention of its velocities
function fromBlock(options: Record<string, unknown>, project: string): Sources {
  const convention = (options.spectral as { convention?: unknown } | undefined)?.convention;
  const velocity = convention === 'radio' || convention === 'optical' ? convention : null;
  return { ...itemSources(options, project), velocity };
}

// The files of a dataset as prep takes them
function prepFiles(sources: Sources, arrays: Arrays): PrepFiles {
  return {
    file_d: prepFile(sources, arrays, 'data')!,
    file_e: sources.errorValue === null ? prepFile(sources, arrays, 'error') : null,
    file_m: prepFile(sources, arrays, 'mask'),
  };
}

// The options of a dataset of the files that prep edited (paths of the
// workspace): its errors are those prepared (of a file, or of the noise;
// standard deviations), or else stay one value; the prepared header states
// the convention of the velocities
function preparedBlock(kind: DataKind, prepared: Prepared, sources: Sources, project: string): Record<string, unknown> {
  const prefix = project === '.' ? '' : `${project}/`;
  const relative = (path: string) => (path.startsWith(prefix) ? path.slice(prefix.length) : path);
  const options: Record<string, unknown> = { type: kind.type, data: relative(prepared.file_d!) };
  if (prepared.file_e !== undefined) options.error = relative(prepared.file_e);
  else if (sources.errorValue !== null) options.error = Number(sources.errorValue);
  if (prepared.file_m !== undefined) options.mask = relative(prepared.file_m);
  return options;
}

// The cube of a dataset, and at the side its files and what GBKFIT reads
// from them, and the edits of a new one; its files and edits can change
// when it says how
function DatasetBody({
  project,
  sources,
  arrays,
  edits,
  onChange,
  onEdits,
}: {
  project: string;
  sources: Sources;
  arrays: Arrays;
  edits?: Edits;
  onChange?: (sources: Sources) => void;
  onEdits?: (edits: Edits) => void;
}) {
  const data = arrays.found.data;
  const cube = data && kept(data);
  const isCube = sources.files.data !== undefined && data !== undefined && cube?.length === 3;
  const crops = edits && cube ? cropsOf(edits, cube) : null;
  // The selection on the cube (to look at; the crop is set from it)
  const [selection, setSelection] = useState<Region | null>(null);
  const [band, setBand] = useState<[number, number] | null>(null);
  // (the column at the side is as wide as dragged, and folds away)
  const viewer =
    isCube ? (
      <CubeViewer
        key={`${sources.files.data!.file}:${data.index}`}
        layers={layersOf(sources, arrays)}
        shape={cube}
        selection={selection}
        band={band}
        onSelection={setSelection}
        onBand={setBand}
        crop={crops?.region}
        cropChannels={crops?.channels}
        kept={edits ? keptOf(edits) : null}
      />
    ) : (
      <div className="card viewer-empty">
        <p className="note">Import or upload the data (at the side) to see them here.</p>
      </div>
    );
  const column = (
    <div className="col">
      <FilesCard project={project} sources={sources} arrays={arrays} onChange={onChange} />
      {isCube && edits && onEdits && (
        <EditsCard
          data={data}
          edits={edits}
          hasErrorFile={sources.files.error !== undefined}
          onChange={onEdits}
          onCropToSelection={
            selection === null && band === null
              ? undefined
              : () => {
                  let next = edits;
                  if (selection !== null) next = withRegion(next, selection);
                  if (band !== null) next = withChannels(next, band);
                  onEdits(next);
                }
          }
        />
      )}
      <Facts array={data} velocity={sources.velocity} onVelocity={onChange && ((velocity) => onChange({ ...sources, velocity }))} />
    </div>
  );
  return (
    <div className="screen-body dataset">
      <SidePanes main={viewer} side={column} />
    </div>
  );
}

