// A dataset: its files (its data, their errors and their mask: an array
// of a file each) and what GBKFIT reads from their headers. A new one is
// named, given its files and added to the project, once nothing stops
// it; one of the project is shown as it is.
import { useRef, useState } from 'react';
import { upload, type ArrayInfo } from '../api/client';
import { useCreateBlock, useDescription, useImportData, usePiece, useProject } from '../api/queries';
import { FilePicker, joinPath } from '../components/FilePicker';
import { dataKind, dataKindOfType, type DataKind } from '../shell/blocks';
import { useStudio } from '../studio';

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

export function NewDataset({ data }: { data: string }) {
  const { project, go } = useStudio();
  const info = useProject(project);
  const create = useCreateBlock(project ?? '.');
  const kind = dataKind(data);
  const [name, setName] = useState('');
  const [files, setFiles] = useState<Files>({});
  const arrays = useArrays(files);
  if (kind === undefined || project === null) return null;
  const taken = info.data?.pieces.datasets.includes(name.trim()) ?? false;
  const reason =
    blocker(kind, arrays) ??
    (name.trim() === '' ? 'Name the dataset' : taken ? 'The project has a dataset of that name' : null);
  const add = () =>
    create.mutate(
      { folder: 'datasets', name: name.trim(), content: toBlock(kind, arrays, files, project) },
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
      <DatasetBody project={project} files={files} arrays={arrays} onChange={setFiles} />
    </section>
  );
}

export function Dataset({ name }: { name: string }) {
  const { project } = useStudio();
  const piece = usePiece(project === null ? null : joinPath(project, `datasets/${name}.yaml`));
  const files = piece.data === undefined || project === null ? {} : fromBlock(piece.data, project);
  const arrays = useArrays(files);
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
      {project !== null && <DatasetBody project={project} files={files} arrays={arrays} />}
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

function useArrays(files: Files): Arrays {
  const described = {
    data: useDescription(files.data?.file ?? null),
    error: useDescription(files.error?.file ?? null),
    mask: useDescription(files.mask?.file ?? null),
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

// The likeliest array of a file for a role: for the data, the first with
// three axes (a cube); for their errors and mask, the first of the shape
// of the data
function likeliest(role: Role, arrays: ArrayInfo[], data: ArrayInfo | undefined): ArrayInfo | undefined {
  const fits =
    role === 'data'
      ? (a: ArrayInfo) => a.shape.length === 3
      : (a: ArrayInfo) => data !== undefined && a.shape.join() === data.shape.join();
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
function blocker(kind: DataKind, arrays: Arrays): string | null {
  const data = arrays.found.data;
  if (arrays.errors.data) return 'The data cannot be read';
  if (data === undefined) return `Choose the ${kind.title.toLowerCase()}`;
  if (data.shape.length !== 3) return `The data are not a cube: they have ${data.shape.length} axes`;
  if (data.problems.length > 0) return 'The data need a fix (see What GBKFIT reads)';
  for (const role of ['error', 'mask'] as const) {
    const array = arrays.found[role];
    if (arrays.errors[role]) return `The ${role === 'error' ? 'errors' : 'mask'} cannot be read`;
    if (array !== undefined && array.shape.join() !== data.shape.join()) {
      return `The ${role === 'error' ? 'errors are' : 'mask is'} not of the shape of the data`;
    }
  }
  return null;
}

// The options of a dataset of the arrays of files (their paths relative
// to the project)
function toBlock(kind: DataKind, arrays: Arrays, files: Files, project: string): Record<string, unknown> {
  const prefix = project === '.' ? '' : `${project}/`;
  const options: Record<string, unknown> = { type: kind.type };
  for (const { role } of ROLES) {
    const chosen = files[role];
    const array = arrays.found[role];
    if (chosen !== undefined && array !== undefined) {
      options[role] = [chosen.file.slice(prefix.length), arrayRef(arrays.inFile[role] ?? [], array)];
    }
  }
  return options;
}

// The files of the options of a dataset: a file, or a file and its array
function fromBlock(options: Record<string, unknown>, project: string): Files {
  const files: Files = {};
  for (const { role } of ROLES) {
    const value = options[role];
    const [file, array] = typeof value === 'string' ? [value, 0] : Array.isArray(value) ? value : [];
    if (typeof file === 'string' && (typeof array === 'string' || typeof array === 'number')) {
      files[role] = { file: joinPath(project, file), array };
    }
  }
  return files;
}

// The files of a dataset (editable when it changes them), and what GBKFIT
// reads from them
function DatasetBody({
  project,
  files,
  arrays,
  onChange,
}: {
  project: string;
  files: Files;
  arrays: Arrays;
  onChange?: (files: Files) => void;
}) {
  const set = (role: Role, value: FileArray | undefined) => onChange?.({ ...files, [role]: value });
  const data = arrays.found.data;
  // The other arrays of the file of the data of its shape (e.g. its
  // errors), offered for the errors
  const siblings = (arrays.inFile.data ?? []).filter(
    (a) => data !== undefined && a.index !== data.index && a.shape.join() === data.shape.join(),
  );
  return (
    <div className="screen-body dataset">
      <div className="card">
        <h2 className="card-head">Files</h2>
        <div className="card-body slots">
          {ROLES.map(({ role, title }) => {
            const chosen = files[role];
            const inFile = arrays.inFile[role] ?? [];
            return (
              <div className="slot-row" key={role}>
                <span className="slot-label">{title}</span>
                {chosen === undefined ? (
                  <div className="slot empty">
                    {onChange ? (
                      <>
                        <FileSources project={project} onFile={(file) => set(role, { file })} />
                        {role === 'error' &&
                          files.data !== undefined &&
                          siblings.map((a) => (
                            <button
                              key={a.index}
                              className="link"
                              onClick={() => set(role, { file: files.data!.file, array: arrayRef(arrays.inFile.data ?? [], a) })}
                            >
                              use {a.name || `array ${a.index}`} of the same file
                            </button>
                          ))}
                      </>
                    ) : (
                      <span className="note">none</span>
                    )}
                  </div>
                ) : (
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
                )}
                {arrays.errors[role] && <p className="error slot-error">{arrays.errors[role]}</p>}
              </div>
            );
          })}
        </div>
      </div>
      <Facts array={data} />
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

// What GBKFIT reads from the header of the data, and what stops it
function Facts({ array }: { array: ArrayInfo | undefined }) {
  return (
    <div className="card">
      <h2 className="card-head">What GBKFIT reads</h2>
      <div className="card-body">
        {array === undefined ? (
          <p className="note">Choose the data to see what GBKFIT reads from their header.</p>
        ) : (
          <dl className="facts">
            <dt>shape</dt>
            <dd>{shapeText(array.shape)}</dd>
            <dt>pixel</dt>
            <dd>{array.step ? `${round(Math.abs(array.step[0]))}″ × ${round(Math.abs(array.step[1]))}″` : 'unknown'}</dd>
            {array.shape.length === 3 && (
              <>
                <dt>spectral</dt>
                <dd>
                  {array.axes[2] || 'no type'}
                  {array.step && ` · ${round(Math.abs(array.step[2]))} km/s a channel`}
                </dd>
              </>
            )}
            <dt>unit</dt>
            <dd>{array.unit ?? 'none in the header'}</dd>
            <dt>beam</dt>
            <dd>{beamText(array.beam)}</dd>
            {array.problems.length > 0 && (
              <>
                <dt>needs</dt>
                <dd>
                  <ul className="problems">
                    {array.problems.map((problem) => (
                      <li key={problem}>{problem}</li>
                    ))}
                  </ul>
                </dd>
              </>
            )}
          </dl>
        )}
      </div>
    </div>
  );
}

function shapeText(shape: number[]): string {
  const [x, y, z] = shape;
  return shape.length === 3 ? `${x} × ${y} pixels × ${z} channels` : shape.join(' × ');
}

function beamText(beam: ArrayInfo['beam']): string {
  if (!beam) return 'none in the header';
  const { bmaj, bmin, bpa } = beam;
  if (Array.isArray(bmaj)) return `one for each channel: ${round(Math.min(...bmaj))}″ to ${round(Math.max(...bmaj))}″`;
  return `${round(bmaj)}″ × ${round(Number(bmin))}″ at ${round(Number(bpa))}°`;
}

// A value to 3 significant digits
function round(value: number): string {
  return String(Number(value.toPrecision(3)));
}
