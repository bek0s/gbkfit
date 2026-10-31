// A dataset: its cube, to look at (see CubeViewer), and at the side its
// files (its data, their errors and their mask: an array of a file each;
// the errors can be one value for all the data instead) and what GBKFIT
// reads from their headers, with the fixes it takes (axes of length 1 left
// out; the convention of velocities that a header leaves ambiguous). A
// new one is named, given its files (which prep can prepare, see Prepare)
// and added to the project, once nothing stops it; one of the project is
// shown as it is.
import { useRef, useState } from 'react';
import { upload, type ArrayInfo } from '../api/client';
import {
  useCreateBlock,
  useDescription,
  useImportData,
  usePiece,
  useProject,
  type Velocity,
} from '../api/queries';
import { CubeViewer } from '../components/CubeViewer';
import { FilePicker, joinPath } from '../components/FilePicker';
import { significant } from '../components/format';
import { dataKind, dataKindOfType, type DataKind } from '../shell/blocks';
import { useStudio } from '../studio';
import { Prepare, type PrepFile, type Prepared } from './Prepare';

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
// errors, one value for all the data instead (as typed; null: none); and
// the convention of its velocities, if its header leaves it ambiguous
interface Sources {
  files: Files;
  errorValue: string | null;
  velocity: Velocity | null;
}

const NO_SOURCES: Sources = { files: {}, errorValue: null, velocity: null };

export function NewDataset({ data }: { data: string }) {
  const { project, go } = useStudio();
  const info = useProject(project);
  const create = useCreateBlock(project ?? '.');
  const kind = dataKind(data);
  const [name, setName] = useState('');
  const [sources, setSources] = useState<Sources>(NO_SOURCES);
  const arrays = useArrays(sources);
  if (kind === undefined || project === null) return null;
  const taken = info.data?.pieces.datasets.includes(name.trim()) ?? false;
  const reason =
    blocker(kind, arrays, sources) ??
    (name.trim() === '' ? 'Name the dataset' : taken ? 'The project has a dataset of that name' : null);
  const add = () =>
    create.mutate(
      { folder: 'datasets', name: name.trim(), content: toBlock(kind, arrays, sources, project) },
      { onSuccess: () => go({ kind: 'block', folder: 'datasets', name: name.trim() }) },
    );
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
          {reason && <span className="why">{reason}</span>}
          <button className="ghost" onClick={() => go({ kind: 'home' })}>
            Cancel
          </button>
          <button className="primary" disabled={reason !== null || create.isPending} onClick={add}>
            Add to the project
          </button>
        </div>
      </div>
      {create.error && <p className="error screen-error">{create.error.message}</p>}
      <DatasetBody project={project} sources={sources} arrays={arrays} onChange={setSources} />
    </section>
  );
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
// path relative to the project), its error value and the convention of
// its velocities
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
  if (sources.velocity !== null) options.spectral = { convention: sources.velocity };
  return options;
}

// The sources of the options of a dataset: a file, or a file and its
// array ({file, hdu}), or for the errors a number; and the convention of
// its velocities
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
  const convention = (options.spectral as { convention?: unknown } | undefined)?.convention;
  const velocity = convention === 'radio' || convention === 'optical' ? convention : null;
  return { files, errorValue, velocity };
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

// The sources of a dataset whose files were prepared: its errors are those
// prepared (of a file, or of the noise), or else stay one value; the
// prepared header states the convention of the velocities
function withPrepared(sources: Sources, prepared: Prepared): Sources {
  const file = (path: string | undefined) => (path === undefined ? undefined : { file: path });
  return {
    files: { data: file(prepared.file_d), error: file(prepared.file_e), mask: file(prepared.file_m) },
    errorValue: prepared.file_e === undefined ? sources.errorValue : null,
    velocity: null,
  };
}

// The cube of a dataset, and at the side its files (editable when it
// changes them, and then preparable), and what GBKFIT reads from them
function DatasetBody({
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
  const data = arrays.found.data;
  const cube = data && kept(data);
  return (
    <div className="screen-body dataset">
      {sources.files.data !== undefined && data !== undefined && cube?.length === 3 ? (
        <CubeViewer
          key={`${sources.files.data.file}:${data.index}`}
          path={sources.files.data.file}
          hdu={data.index}
          shape={cube}
          unit={data.unit}
        />
      ) : (
        <div className="card viewer-empty">
          <p className="note">Import or upload the data (at the side) to see them here.</p>
        </div>
      )}
      <div className="col">
        <FilesCard project={project} sources={sources} arrays={arrays} onChange={onChange} />
        <Facts array={data} velocity={sources.velocity} onVelocity={onChange && ((velocity) => onChange({ ...sources, velocity }))} />
        {onChange && sources.files.data !== undefined && data !== undefined && cube?.length === 3 && (
          <Prepare
            key={sources.files.data.file}
            project={project}
            data={data}
            files={{
              file_d: prepFile(sources, arrays, 'data')!,
              file_e: sources.errorValue === null ? prepFile(sources, arrays, 'error') : null,
              file_m: prepFile(sources, arrays, 'mask'),
            }}
            velocity={sources.velocity}
            onPrepared={(prepared) => onChange(withPrepared(sources, prepared))}
          />
        )}
      </div>
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
  // errors), offered for the errors
  const siblings = (arrays.inFile.data ?? []).filter(
    (a) => data !== undefined && a.index !== data.index && kept(a).join() === kept(data).join(),
  );
  return (
    <div className="card">
      <h2 className="card-head">Files</h2>
      <div className="card-body files">
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
                  <button className="slot-clear" aria-label="Remove the error" onClick={() => setErrorValue(null)}>
                    ×
                  </button>
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
                      <button
                        className="link"
                        title="One error for every value of the data (e.g. the rms of a radio cube)"
                        onClick={() => setErrorValue(data?.rms != null ? String(data.rms) : '')}
                      >
                        One value
                      </button>
                    )}
                    {role === 'error' &&
                      files.data !== undefined &&
                      siblings.map((a) => (
                        <button
                          key={a.index}
                          className="link"
                          onClick={() => set(role, { file: files.data!.file, array: arrayRef(arrays.inFile.data ?? [], a) })}
                        >
                          use {a.name || `array ${a.index}`}
                        </button>
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
                  <select
                    aria-label={`The array of the ${title.toLowerCase()}`}
                    value={arrays.found[role]?.index ?? ''}
                    onChange={(event) => {
                      const array = inFile.find((a) => a.index === Number(event.target.value));
                      if (array) set(role, { file: chosen.file, array: arrayRef(inFile, array) });
                    }}
                  >
                    {inFile.map((a) => (
                      <option key={a.index} value={a.index}>
                        {a.name || `array ${a.index}`}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="note">{arrays.found[role]?.name || String(chosen.array ?? '')}</span>
                )}
                {onChange && (
                  <button className="slot-clear" aria-label={`Remove the ${title.toLowerCase()}`} onClick={() => set(role, undefined)}>
                    ×
                  </button>
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
      </div>
    </div>
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
      <button className="link" title="A file of the server's workspace; the project keeps a copy" onClick={() => setPicking(true)}>
        Import
      </button>
      <button className="link" title="A file of this computer; the project keeps a copy" onClick={() => input.current?.click()}>
        Upload
      </button>
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
      <div className="card">
        <h2 className="card-head">What GBKFIT reads</h2>
        <p className="card-body note">The facts of the data come from their header, once they are given.</p>
      </div>
    );
  }
  const flat = array.shape.map((size, i) => [size, array.axes[i]] as const).filter(([size]) => size === 1);
  const [x, y, z] = kept(array);
  return (
    <div className="card">
      <h2 className="card-head">What GBKFIT reads</h2>
      <div className="card-body">
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
              <>
                <div>
                  The header does not say whether the velocities are radio or optical (
                  {keptAxes(array)[2]} without VELREF; MIRIAD writes radio ones).
                </div>
                <div className="seg" role="group" aria-label="The convention of the velocities">
                  {(['radio', 'optical'] as const).map((convention) => (
                    <button
                      key={convention}
                      aria-pressed={velocity === convention}
                      disabled={onVelocity === undefined}
                      onClick={() => onVelocity?.(convention)}
                    >
                      {convention === 'radio' ? 'Radio' : 'Optical'}
                    </button>
                  ))}
                </div>
              </>
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
      </div>
    </div>
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
