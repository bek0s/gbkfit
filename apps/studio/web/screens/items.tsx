// The items of datasets (a cube; each map of moment maps), of the screens
// of datasets: their files (their data, their errors and their mask: an
// array of a file each; the errors can be one value for all the data
// instead), the arrays of those files (described by the server), their
// options in a dataset and back, and their cards: their files, and what
// GBKFIT reads from the header of their data.
import { useRef, useState, type ReactNode } from 'react';
import { upload, type ArrayInfo } from '../api/client';
import { useDescription, useImportData, type Velocity } from '../api/queries';
import type { Layer } from '../components/CubeViewer';
import { FilePicker, joinPath } from '../components/FilePicker';
import { significant } from '../components/format';
import { Button, Callout, Close } from '../components/ui/basics';
import { Choice, Segments } from '../components/ui/inputs';
import { DropMenu } from '../components/ui/overlays';
import { Panel } from '../components/ui/panel';
import { ERROR_KINDS, type ErrorKind, type PrepFile } from './Edits';

export type Role = 'data' | 'error' | 'mask';

export const ROLES: { role: Role; title: string }[] = [
  { role: 'data', title: 'Data' },
  { role: 'error', title: 'Errors' },
  { role: 'mask', title: 'Mask' },
];

// A file of the workspace, and its array: by name, or by index if it has
// none of its own; none for a file just chosen, whose array is then the
// likeliest one (see likeliest)
export interface FileArray {
  file: string;
  array?: string | number;
}

export type Files = Partial<Record<Role, FileArray>>;

// What a dataset is made of: the arrays of files of its roles; for its
// errors, one value for all the data instead (as typed; null: none), and
// what their file holds (null: not chosen, guessed from the name of their
// array; see errorKindOf); and the convention of its velocities, if its
// header leaves it ambiguous
export interface Sources {
  files: Files;
  errorValue: string | null;
  errorKind: ErrorKind | null;
  velocity: Velocity | null;
}

export const NO_SOURCES: Sources = { files: {}, errorValue: null, errorKind: null, velocity: null };

// The arrays of the files of a dataset, by role, from their descriptions
// (undefined until described, or if the file cannot be)
export interface Arrays {
  found: Partial<Record<Role, ArrayInfo>>;
  // The arrays of the file of each role, and why one cannot be described
  inFile: Partial<Record<Role, ArrayInfo[]>>;
  errors: Partial<Record<Role, string>>;
}

export function useArrays({ files, velocity }: Sources, ndim = 3): Arrays {
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
      chosen.array === undefined ? likeliest(role, data, arrays.found.data, ndim) : findArray(data, chosen.array);
  }
  return arrays;
}

// The shape of an array without its axes of length 1, as GBKFIT reads it
export function kept(array: ArrayInfo): number[] {
  return array.shape.filter((size) => size > 1);
}

// The types of the axes of an array without those of length 1
export function keptAxes(array: ArrayInfo): string[] {
  return array.axes.filter((_, i) => array.shape[i] > 1);
}

// The likeliest array of a file for a role: for the data, the first of the
// number of axes of the data (three: a cube; two: a map); for their errors
// and mask, the first of the shape of the data
function likeliest(role: Role, arrays: ArrayInfo[], data: ArrayInfo | undefined, ndim: number): ArrayInfo | undefined {
  const fits =
    role === 'data'
      ? (a: ArrayInfo) => kept(a).length === ndim
      : (a: ArrayInfo) => data !== undefined && kept(a).join() === kept(data).join();
  return arrays.find(fits) ?? arrays[0];
}

// The array of the file of the data that holds their errors, by its name
// (ERR, ERROR, SIGMA, NOISE, STAT, VAR..., IVAR..., or those after the
// name of the data, e.g. MOM1_ERR), of their shape, if one does (a mask is never guessed: its conventions differ, e.g. a DQ of
// JWST is 0 where good)
export function errorSibling(arrays: Arrays): string | number | undefined {
  const data = arrays.found.data;
  const inFile = arrays.inFile.data ?? [];
  const siblings = inFile.filter((a) => data !== undefined && a.index !== data.index && kept(a).join() === kept(data).join());
  // (those of the data by name first, e.g. MOM1_ERR of MOM1, of a file of
  // several maps and their errors)
  const own = data?.name ? new RegExp(`^${data.name.replace(/[^\w]/g, '\\$&')}[_-]?${ERROR_NAMES}$`, 'i') : null;
  const found =
    (own && siblings.find((a) => own.test(a.name ?? ''))) ?? siblings.find((a) => new RegExp(`^${ERROR_NAMES}`, 'i').test(a.name ?? ''));
  return found && arrayRef(inFile, found);
}

// The names of arrays of errors (see errorSibling)
const ERROR_NAMES = '(ERR|ERROR|ERRORS|SIGMA|NOISE|STAT|VAR|IVAR|INVVAR|INV_VAR)';

// What the file of the errors of a dataset holds: as chosen, or else as
// the name of its array says (variances: STAT of MUSE, VAR_* of JWST;
// inverse variances: IVAR of MaNGA), or else standard deviations
export function errorKindOf(sources: Sources, arrays: Arrays): ErrorKind {
  if (sources.errorKind !== null) return sources.errorKind;
  const name = arrays.found.error?.name ?? '';
  if (/(^|_)(STAT|VAR)/i.test(name)) return 'variance';
  if (/(^|_)(IVAR|INVVAR|INV_VAR)/i.test(name)) return 'inverse_variance';
  return 'sigma';
}

function findArray(arrays: ArrayInfo[], array: string | number): ArrayInfo | undefined {
  return typeof array === 'number' ? arrays.find((a) => a.index === array) : arrays.find((a) => a.name === array);
}

// How an array is named in a dataset: by its name, if it is its own in
// its file, or else by its index
export function arrayRef(arrays: ArrayInfo[], array: ArrayInfo): string | number {
  return array.name && arrays.filter((a) => a.name === array.name).length === 1 ? array.name : array.index;
}

// The options of an item of a dataset (its files: data, error and mask, each
// {file, hdu}, its path relative to the project; its error value or what
// the file of its errors holds, unless standard deviations, the default)
export function itemOptions(arrays: Arrays, sources: Sources, project: string): Record<string, unknown> {
  const prefix = project === '.' ? '' : `${project}/`;
  const options: Record<string, unknown> = {};
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
  return options;
}

// The sources of the options of an item of a dataset: a file, or a file
// and its array ({file, hdu}), or for the errors a number, and what their
// file holds (standard deviations unless it says; never guessed)
export function itemSources(options: Record<string, unknown>, project: string): Sources {
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
  return { files, errorValue, errorKind, velocity: null };
}

// Why the files of an item (of data of ndim axes) cannot be a dataset yet
// (null: they can); what the data are, as the page names them
export function itemBlocker(what: string, arrays: Arrays, sources: Sources, ndim: number): string | null {
  const data = arrays.found.data;
  if (arrays.errors.data) return `The ${what} cannot be read`;
  if (data === undefined) return `Import or upload the ${what}`;
  if (kept(data).length !== ndim) return `The ${what} have ${kept(data).length} axes, not ${ndim}`;
  // (the convention of velocities matters to a cube alone)
  if (ndim === 3 && data.ambiguous_velocity && sources.velocity === null) return 'Say whether the velocities are radio or optical';
  if (data.problems.length > 0) return `The ${what} need a fix (see What GBKFIT reads)`;
  for (const role of ['error', 'mask'] as const) {
    const array = arrays.found[role];
    if (arrays.errors[role]) return `The ${role === 'error' ? 'errors' : 'mask'} of the ${what} cannot be read`;
    if (array !== undefined && kept(array).join() !== kept(data).join()) {
      return `The ${role === 'error' ? 'errors are' : 'mask is'} not of the shape of the ${what}`;
    }
  }
  if (sources.errorValue !== null && !(Number(sources.errorValue) > 0)) return 'Give the error a value above 0';
  return null;
}

// A file of a role as prep takes it: its path, or its path and its array
// if the file has others
export function prepFile(sources: Sources, arrays: Arrays, role: Role): PrepFile | null {
  const chosen = sources.files[role];
  const array = arrays.found[role];
  const inFile = arrays.inFile[role] ?? [];
  if (chosen === undefined || array === undefined) return null;
  return inFile.length === 1 ? chosen.file : { file: chosen.file, hdu: arrayRef(inFile, array) };
}

// The cubes of a dataset to look at: its data, and the errors and the mask
// of files (as the data, once described)
export function layersOf(sources: Sources, arrays: Arrays): Layer[] {
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

// The files of an item of a dataset (its data, their errors and their
// mask): a line for each, in a panel of a title (and actions, e.g. to
// remove it), under what is wrong with the item as a whole, if anything
export function FilesCard({
  project,
  sources,
  arrays,
  onChange,
  id = 'dataset-files',
  title = 'Files',
  actions,
  problem,
  ndim = 3,
}: {
  project: string;
  sources: Sources;
  arrays: Arrays;
  onChange?: (sources: Sources) => void;
  id?: string;
  title?: string;
  actions?: ReactNode;
  problem?: string | null;
  // The number of axes of the data (3: a cube; 2: a map)
  ndim?: number;
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
    <Panel id={id} title={title} actions={actions} bodyClass="files">
      {problem && <p className="error item-problem">{problem}</p>}
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
                    (siblings.length <= 2 ? (
                      siblings.map((a) => (
                        <Button
                          key={a.index}
                          kind="small"
                          title="An array of the file of the data"
                          onClick={() => set(role, { file: files.data!.file, array: arrayRef(arrays.inFile.data ?? [], a) })}
                        >
                          use {a.name || `array ${a.index}`}
                        </Button>
                      ))
                    ) : (
                      <DropMenu
                        label="An array of the file of the data"
                        entries={siblings.map((a) => ({
                          label: a.name || `array ${a.index}`,
                          onClick: () => set(role, { file: files.data!.file, array: arrayRef(arrays.inFile.data ?? [], a) }),
                        }))}
                        target={
                          <Button kind="small" title="An array of the file of the data, of its shape" right="▾">
                            use an array of the data
                          </Button>
                        }
                      />
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
                  items={inFile.map((a) => {
                    const why = unusable(role, a, data, ndim);
                    return {
                      value: String(a.index),
                      label: `${a.name || `array ${a.index}`} · ${kept(a).join(' × ') || 'one value'}${why ? ` (${why})` : ''}`,
                      // (the one chosen stays, to show what is wrong)
                      disabled: why !== null && a.index !== arrays.found[role]?.index,
                    };
                  })}
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

// Why an array of a file cannot be of a role (null: it can): the data
// must have ndim axes (of length above 1), their errors and mask their
// shape
function unusable(role: Role, array: ArrayInfo, data: ArrayInfo | undefined, ndim: number): string | null {
  const axes = kept(array).length;
  if (role === 'data') return axes === ndim ? null : `${axes} ${axes === 1 ? 'axis' : 'axes'}`;
  if (data === undefined || kept(array).join() === kept(data).join()) return null;
  return 'another shape';
}

// The two ways to give a slot its file, each copying it into the project:
// a file of the workspace of the server (Import), or one of this computer
// (Upload)
export function FileSources({ project, onFile }: { project: string; onFile: (path: string) => void }) {
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
export function Facts({
  array,
  velocity,
  onVelocity,
  cube = true,
}: {
  array: ArrayInfo | undefined;
  velocity: Velocity | null;
  // Whether the data are a cube (of velocities, whose convention matters)
  cube?: boolean;
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
      {(flat.length > 0 || (cube && array.ambiguous_velocity)) && (
        <div className="fixes">
          {flat.length > 0 && (
            <div className="fixed">
              ✓ Left out: the {flat.length > 1 ? 'axes' : 'axis'} of length 1 ({flat.map(([, axis]) => axis || 'no type').join(', ')})
            </div>
          )}
          {cube && array.ambiguous_velocity && (
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
