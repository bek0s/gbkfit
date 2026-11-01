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
import { upload, type ArrayInfo } from '../api/client';
import {
  useCreateBlock,
  useDescription,
  useImportData,
  useJob,
  usePiece,
  useProject,
  useSubmitJob,
  type Velocity,
} from '../api/queries';
import { CubeViewer, type Layer } from '../components/CubeViewer';
import type { Region } from '../components/MapView';
import { FilePicker, joinPath } from '../components/FilePicker';
import { Panel } from '../components/ui/panel';
import { PanelsIcon } from '../components/icons';
import { Button, Callout, Close, IconButton } from '../components/ui/basics';
import { Choice, Segments } from '../components/ui/inputs';
import { SidePanes, type PanesControl } from '../components/ui/panes';
import { significant } from '../components/format';
import { dataKind, dataKindOfType, type DataKind } from '../shell/blocks';
import { useStudio } from '../studio';
import {
  cropsOf,
  EditsCard,
  ERROR_KINDS,
  keptOf,
  needsPrep,
  NO_EDITS,
  prepOptions,
  withChannels,
  withRegion,
  type Edits,
  type ErrorKind,
  type PrepFile,
  type PrepFiles,
  type Prepared,
} from './Edits';

type Role = 'data' | 'error' | 'mask';

const ROLES: { role: Role; title: string }[] = [
  { role: 'data', title: 'Data' },
  { role: 'error', title: 'Errors' },
  { role: 'mask', title: 'Mask' },
];

// A file of the workspace, and its array: by name, or by index if it has
// none of its own; none for a file just chosen, whose array is then the
// likeliest one (see likeliest)
interface FileArray {
  file: string;
  array?: string | number;
}

type Files = Partial<Record<Role, FileArray>>;

// What a dataset is made of: the arrays of files of its roles; for its
// errors, one value for all the data instead (as typed; null: none), and
// what their file holds (null: not chosen, guessed from the name of their
// array; see errorKindOf); and the convention of its velocities, if its
// header leaves it ambiguous
interface Sources {
  files: Files;
  errorValue: string | null;
  errorKind: ErrorKind | null;
  velocity: Velocity | null;
}

const NO_SOURCES: Sources = { files: {}, errorValue: null, errorKind: null, velocity: null };

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

export function Dataset({ name }: { name: string }) {
  const { project } = useStudio();
  const piece = usePiece(project === null ? null : joinPath(project, `datasets/${name}.yaml`));
  const sources = piece.data === undefined || project === null ? NO_SOURCES : fromBlock(piece.data, project);
  const arrays = useArrays(sources);
  const kind = dataKindOfType(piece.data?.type);
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Dataset{kind ? ` · ${kind.title}` : ''}</div>
          <h1>{name}</h1>
        </div>
      </div>
      {piece.error && <p className="error screen-error">{piece.error.message}</p>}
      {project !== null && <DatasetBody project={project} sources={sources} arrays={arrays} />}
    </section>
  );
}

// The arrays of the files of a dataset, by role, from their descriptions
// (undefined until described, or if the file cannot be)
interface Arrays {
  found: Partial<Record<Role, ArrayInfo>>;
  // The arrays of the file of each role, and why one cannot be described
  inFile: Partial<Record<Role, ArrayInfo[]>>;
  errors: Partial<Record<Role, string>>;
}

function useArrays({ files, velocity }: Sources): Arrays {
  const described = {
    data: useDescription(files.data?.file ?? null, velocity),
    error: useDescription(files.error?.file ?? null, velocity),
    mask: useDescription(files.mask?.file ?? null, velocity),
  };
  const arrays: Arrays = { found: {}, inFile: {}, errors: {} };
  for (const { role } of ROLES) {
    const chosen = files[role];
    const { data, error } = described[role];
    if (chosen === undefined) continue;
    if (error) arrays.errors[role] = error.message;
    if (data === undefined) continue;
    arrays.inFile[role] = data;
    arrays.found[role] =
      chosen.array === undefined ? likeliest(role, data, arrays.found.data) : findArray(data, chosen.array);
  }
  return arrays;
}

// The shape of an array without its axes of length 1, as GBKFIT reads it
function kept(array: ArrayInfo): number[] {
  return array.shape.filter((size) => size > 1);
}

// The types of the axes of an array without those of length 1
function keptAxes(array: ArrayInfo): string[] {
  return array.axes.filter((_, i) => array.shape[i] > 1);
}

// The likeliest array of a file for a role: for the data, the first with
// three axes (a cube); for their errors and mask, the first of the shape
// of the data
function likeliest(role: Role, arrays: ArrayInfo[], data: ArrayInfo | undefined): ArrayInfo | undefined {
  const fits =
    role === 'data'
      ? (a: ArrayInfo) => kept(a).length === 3
      : (a: ArrayInfo) => data !== undefined && kept(a).join() === kept(data).join();
  return arrays.find(fits) ?? arrays[0];
}

// The array of the file of the data that holds their errors, by its name
// (ERR, ERROR, SIGMA, NOISE, STAT, VAR..., IVAR...), of their shape, if one
// does (a mask is never guessed: its conventions differ, e.g. a DQ of
// JWST is 0 where good)
function errorSibling(arrays: Arrays): string | number | undefined {
  const data = arrays.found.data;
  const inFile = arrays.inFile.data ?? [];
  const found = inFile.find(
    (a) =>
      data !== undefined &&
      a.index !== data.index &&
      kept(a).join() === kept(data).join() &&
      /^(ERR|ERROR|ERRORS|SIGMA|NOISE|STAT|VAR|IVAR|INVVAR|INV_VAR)/i.test(a.name ?? ''),
  );
  return found && arrayRef(inFile, found);
}

// What the file of the errors of a dataset holds: as chosen, or else as
// the name of its array says (variances: STAT of MUSE, VAR_* of JWST;
// inverse variances: IVAR of MaNGA), or else standard deviations
function errorKindOf(sources: Sources, arrays: Arrays): ErrorKind {
  if (sources.errorKind !== null) return sources.errorKind;
  const name = arrays.found.error?.name ?? '';
  if (/^(STAT|VAR)/i.test(name)) return 'variance';
  if (/^(IVAR|INVVAR|INV_VAR)/i.test(name)) return 'inverse_variance';
  return 'sigma';
}

function findArray(arrays: ArrayInfo[], array: string | number): ArrayInfo | undefined {
  return typeof array === 'number' ? arrays.find((a) => a.index === array) : arrays.find((a) => a.name === array);
}

// How an array is named in a dataset: by its name, if it is its own in
// its file, or else by its index
function arrayRef(arrays: ArrayInfo[], array: ArrayInfo): string | number {
  return array.name && arrays.filter((a) => a.name === array.name).length === 1 ? array.name : array.index;
}

// Why a dataset of these arrays cannot be added yet (null: it can)
function blocker(kind: DataKind, arrays: Arrays, sources: Sources): string | null {
  const data = arrays.found.data;
  if (arrays.errors.data) return 'The data cannot be read';
  if (data === undefined) return `Import or upload the ${kind.title.toLowerCase()}`;
  if (kept(data).length !== 3) return `The data are not a cube: they have ${kept(data).length} axes`;
  if (data.ambiguous_velocity && sources.velocity === null) return 'Say whether the velocities are radio or optical';
  if (data.problems.length > 0) return 'The data need a fix (see What GBKFIT reads)';
  for (const role of ['error', 'mask'] as const) {
    const array = arrays.found[role];
    if (arrays.errors[role]) return `The ${role === 'error' ? 'errors' : 'mask'} cannot be read`;
    if (array !== undefined && kept(array).join() !== kept(data).join()) {
      return `The ${role === 'error' ? 'errors are' : 'mask is'} not of the shape of the data`;
    }
  }
  if (sources.errorValue !== null && !(Number(sources.errorValue) > 0)) return 'Give the error a value above 0';
  return null;
}

// The options of a dataset of the arrays of files (each {file, hdu}, its
// path relative to the project), its error value or what the file of its
// errors holds (unless standard deviations, the default), and the
// convention of its velocities
function toBlock(kind: DataKind, arrays: Arrays, sources: Sources, project: string): Record<string, unknown> {
  const prefix = project === '.' ? '' : `${project}/`;
  const options: Record<string, unknown> = { type: kind.type };
  for (const { role } of ROLES) {
    if (role === 'error' && sources.errorValue !== null) {
      options.error = Number(sources.errorValue);
      continue;
    }
    const chosen = sources.files[role];
    const array = arrays.found[role];
    if (chosen !== undefined && array !== undefined) {
      options[role] = { file: chosen.file.slice(prefix.length), hdu: arrayRef(arrays.inFile[role] ?? [], array) };
    }
  }
  const errorKind = errorKindOf(sources, arrays);
  if (sources.errorValue === null && options.error !== undefined && errorKind !== 'sigma') {
    options.error_kind = errorKind;
  }
  if (sources.velocity !== null) options.spectral = { convention: sources.velocity };
  return options;
}

// The sources of the options of a dataset: a file, or a file and its
// array ({file, hdu}), or for the errors a number, and what their file
// holds (standard deviations unless it says; never guessed); and the
// convention of its velocities
function fromBlock(options: Record<string, unknown>, project: string): Sources {
  const files: Files = {};
  for (const { role } of ROLES) {
    const value = options[role];
    const { file, hdu = 0 } = (typeof value === 'string' ? { file: value } : (value ?? {})) as {
      file?: unknown;
      hdu?: unknown;
    };
    if (typeof file === 'string' && (typeof hdu === 'string' || typeof hdu === 'number')) {
      files[role] = { file: joinPath(project, file), array: hdu };
    }
  }
  const errorValue = typeof options.error === 'number' ? String(options.error) : null;
  const errorKind = ERROR_KINDS.find(({ kind }) => kind === options.error_kind)?.kind ?? 'sigma';
  const convention = (options.spectral as { convention?: unknown } | undefined)?.convention;
  const velocity = convention === 'radio' || convention === 'optical' ? convention : null;
  return { files, errorValue, errorKind, velocity };
}

// A file of a role as prep takes it: its path, or its path and its array
// if the file has others
function prepFile(sources: Sources, arrays: Arrays, role: Role): PrepFile | null {
  const chosen = sources.files[role];
  const array = arrays.found[role];
  const inFile = arrays.inFile[role] ?? [];
  if (chosen === undefined || array === undefined) return null;
  return inFile.length === 1 ? chosen.file : { file: chosen.file, hdu: arrayRef(inFile, array) };
}

// The cubes of a dataset to look at: its data, and the errors and the mask
// of files (as the data, once described)
function layersOf(sources: Sources, arrays: Arrays): Layer[] {
  const layers: Layer[] = [];
  for (const { role, title } of ROLES) {
    const chosen = sources.files[role];
    const array = arrays.found[role];
    if (chosen !== undefined && array !== undefined) {
      layers.push({ key: role, title, path: chosen.file, hdu: array.index, unit: array.unit });
    }
  }
  return layers;
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
  const viewer = ({ folded, toggle }: PanesControl) =>
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
        actions={
          <IconButton label={folded ? 'Show the panels at the side' : 'Hide the panels at the side'} pressed={!folded} onClick={toggle}>
            <PanelsIcon />
          </IconButton>
        }
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

// The files of a dataset: a line for each of its data, their errors and
// their mask
function FilesCard({
  project,
  sources,
  arrays,
  onChange,
}: {
  project: string;
  sources: Sources;
  arrays: Arrays;
  onChange?: (sources: Sources) => void;
}) {
  const { files, errorValue } = sources;
  const set = (role: Role, value: FileArray | undefined) => onChange?.({ ...sources, files: { ...files, [role]: value } });
  const setErrorValue = (value: string | null) => onChange?.({ ...sources, errorValue: value });
  const data = arrays.found.data;
  // The other arrays of the file of the data of its shape (e.g. its
  // errors), offered for the errors and the mask
  const siblings = (arrays.inFile.data ?? []).filter(
    (a) => data !== undefined && a.index !== data.index && kept(a).join() === kept(data).join(),
  );
  return (
    <Panel id="dataset-files" title="Files" bodyClass="files">
      {ROLES.map(({ role, title }) => {
        const chosen = files[role];
        const inFile = arrays.inFile[role] ?? [];
        let slot;
        if (role === 'error' && errorValue !== null) {
          slot = (
            <div className="slot">
              {onChange ? (
                <input
                  className="slot-value"
                  type="number"
                  step="any"
                  min="0"
                  autoFocus
                  value={errorValue}
                  aria-label="The error of every value"
                  onChange={(event) => setErrorValue(event.target.value)}
                />
              ) : (
                <span className="slot-file">{errorValue}</span>
              )}
              <span className="note">{data?.unit ?? 'in the unit of the data'} · every value</span>
              {onChange && (
                <Close label="Remove the error" onClick={() => setErrorValue(null)} />
              )}
            </div>
          );
        } else if (chosen === undefined) {
          slot = (
            <div className="slot empty">
              {onChange ? (
                <>
                  <FileSources project={project} onFile={(file) => set(role, { file })} />
                  {role === 'error' && (
                    <Button
                      kind="small"
                      title="One error for every value of the data (e.g. the rms of a radio cube)"
                      onClick={() => setErrorValue(data?.rms != null ? String(data.rms) : '')}
                    >
                      One value
                    </Button>
                  )}
                  {role !== 'data' &&
                    files.data !== undefined &&
                    siblings.map((a) => (
                      <Button
                        key={a.index}
                        kind="small"
                        title="An array of the file of the data"
                        onClick={() => set(role, { file: files.data!.file, array: arrayRef(arrays.inFile.data ?? [], a) })}
                      >
                        use {a.name || `array ${a.index}`}
                      </Button>
                    ))}
                </>
              ) : (
                <span className="note">none</span>
              )}
            </div>
          );
        } else {
          slot = (
            <div className="slot">
              <span className="slot-file" title={chosen.file}>
                {chosen.file.split('/').pop()}
              </span>
              {inFile.length > 1 && onChange ? (
                <Choice
                  items={inFile.map((a) => ({ value: String(a.index), label: a.name || `array ${a.index}` }))}
                  value={String(arrays.found[role]?.index ?? '')}
                  label={`The array of the ${title.toLowerCase()}`}
                  onChange={(index) => {
                    const array = inFile.find((a) => a.index === Number(index));
                    if (array) set(role, { file: chosen.file, array: arrayRef(inFile, array) });
                  }}
                />
              ) : (
                <span className="note">{arrays.found[role]?.name || String(chosen.array ?? '')}</span>
              )}
              {role === 'error' &&
                (onChange ? (
                  <Choice
                    items={ERROR_KINDS.map(({ kind, title }) => ({ value: kind, label: title }))}
                    value={errorKindOf(sources, arrays)}
                    label="What the errors are"
                    title="What the errors are: standard deviations (σ), variances (σ²) or inverse variances (1/σ²)"
                    onChange={(kind) => onChange({ ...sources, errorKind: kind as ErrorKind })}
                  />
                ) : (
                  <span className="note">{ERROR_KINDS.find(({ kind }) => kind === errorKindOf(sources, arrays))?.title}</span>
                ))}
              {onChange && (
                <Close label={`Remove the ${title.toLowerCase()}`} onClick={() => set(role, undefined)} />
              )}
            </div>
          );
        }
        return (
          <div className="file-row" key={role}>
            <span className="file-label">{title}</span>
            {slot}
            {arrays.errors[role] && <p className="error file-error">{arrays.errors[role]}</p>}
          </div>
        );
      })}
    </Panel>
  );
}

// The two ways to give a slot its file, each copying it into the project:
// a file of the workspace of the server (Import), or one of this computer
// (Upload)
function FileSources({ project, onFile }: { project: string; onFile: (path: string) => void }) {
  const importing = useImportData(project);
  const [picking, setPicking] = useState(false);
  // The fraction of the file sent while uploading, and why it failed
  const [sent, setSent] = useState<number | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const send = (file: File) => {
    setFailure(null);
    setSent(0);
    upload(project, file, setSent)
      .then(onFile)
      .catch((error: Error) => setFailure(error.message))
      .finally(() => setSent(null));
  };
  const error = failure ?? importing.error?.message;
  return (
    <>
      <Button kind="small" title="A file of the server's workspace; the project keeps a copy" onClick={() => setPicking(true)}>
        Import
      </Button>
      <Button kind="small" title="A file of this computer; the project keeps a copy" onClick={() => input.current?.click()}>
        Upload
      </Button>
      <input
        ref={input}
        type="file"
        accept=".fits,.fit,.fts,.gz"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) send(file);
        }}
      />
      {sent !== null && <span className="note">uploading {Math.round(100 * sent)}%</span>}
      {importing.isPending && <span className="note">copying…</span>}
      {error && <span className="error">{error}</span>}
      {picking && (
        <FilePicker
          title="Import a file"
          root="."
          label="workspace"
          onClose={() => setPicking(false)}
          onChoose={(path) => {
            setPicking(false);
            importing.mutate(path, { onSuccess: onFile });
          }}
        />
      )}
    </>
  );
}

// What GBKFIT reads from the header of the data, the fixes it takes, and
// what still stops it
function Facts({
  array,
  velocity,
  onVelocity,
}: {
  array: ArrayInfo | undefined;
  velocity: Velocity | null;
  // How the convention of ambiguous velocities is chosen (none: shown)
  onVelocity?: (velocity: Velocity) => void;
}) {
  if (array === undefined) {
    return (
      <Panel id="dataset-facts" title="What GBKFIT reads">
        <p className="note">The facts of the data come from their header, once they are given.</p>
      </Panel>
    );
  }
  const flat = array.shape.map((size, i) => [size, array.axes[i]] as const).filter(([size]) => size === 1);
  const [x, y, z] = kept(array);
  return (
    <Panel id="dataset-facts" title="What GBKFIT reads">
      <dl className="facts">
        <dt>shape</dt>
        <dd>{z === undefined ? kept(array).join(' × ') : `${x} × ${y} pixels × ${z} channels`}</dd>
        <dt>pixel</dt>
        <dd>{array.step ? `${significant(Math.abs(array.step[0]), 3)}″ × ${significant(Math.abs(array.step[1]), 3)}″` : 'unknown'}</dd>
        {z !== undefined && (
          <>
            <dt>spectral</dt>
            <dd>
              {keptAxes(array)[2] || 'no type'}
              {array.step && array.step.length === 3 && ` · ${significant(Math.abs(array.step[2]), 3)} km/s a channel`}
            </dd>
          </>
        )}
        <dt>unit</dt>
        <dd>{array.unit ?? 'none in the header'}</dd>
        <dt>beam</dt>
        <dd>{beamText(array.beam)}</dd>
        {array.rms !== null && (
          <>
            <dt>noise</dt>
            <dd>{significant(array.rms, 3)} (the header)</dd>
          </>
        )}
      </dl>
      {(flat.length > 0 || array.ambiguous_velocity) && (
        <div className="fixes">
          {flat.length > 0 && (
            <div className="fixed">
              ✓ Left out: the {flat.length > 1 ? 'axes' : 'axis'} of length 1 ({flat.map(([, axis]) => axis || 'no type').join(', ')})
            </div>
          )}
          {array.ambiguous_velocity && (
            <Callout>
              <div>
                The header does not say whether the velocities are radio or optical (
                {keptAxes(array)[2]} without VELREF; MIRIAD writes radio ones).
              </div>
              <Segments
                items={(['radio', 'optical'] as const).map((convention) => ({
                  value: convention,
                  label: convention === 'radio' ? 'Radio' : 'Optical',
                  disabled: onVelocity === undefined,
                }))}
                value={velocity}
                label="The convention of the velocities"
                onChange={(convention) => onVelocity?.(convention)}
              />
            </Callout>
          )}
        </div>
      )}
      {array.problems.length > 0 && (
        <ul className="problems">
          {array.problems.map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function beamText(beam: ArrayInfo['beam']): string {
  if (!beam) return 'none in the header';
  const { bmaj, bmin, bpa } = beam;
  if (Array.isArray(bmaj)) {
    return `one for each channel: ${significant(Math.min(...bmaj), 3)}″ to ${significant(Math.max(...bmaj), 3)}″`;
  }
  return `${significant(bmaj, 3)}″ × ${significant(Number(bmin), 3)}″ at ${significant(Number(bpa), 3)}°`;
}
