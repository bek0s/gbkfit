// The preparation of the files of a cube with the tools of prep (see
// gbkfit.tasks.prep.prep_pixel_spectra), as a job of the server: crop them,
// leave out values (below one, outliers, out of the largest regions), make
// their errors from the noise, crop them to their valid pixels, and make
// their channels velocities. The project keeps the prepared copies, and
// the files of the dataset become them.
import { useEffect, useRef, useState } from 'react';
import type { ArrayInfo } from '../api/client';
import { useJob, useSubmitJob, type Velocity } from '../api/queries';

// A file as prep takes it: its path, or its path and its array
export type PrepFile = string | { file: string; hdu: string | number };

// The files of the data, their errors and their mask, by the options of
// prep (and the prepared ones, by the same)
export interface PrepFiles {
  file_d: PrepFile;
  file_e: PrepFile | null;
  file_m: PrepFile | null;
}

export type Prepared = Partial<Record<keyof PrepFiles, string>>;

// The options as typed (blank: not used)
interface Options {
  x: [string, string];
  y: [string, string];
  channels: [string, string];
  below: string;
  sigma: string;
  regions: string;
  pixels: string;
  ratio: string;
  noise: boolean;
  minify: boolean;
  pad: string;
  line: string;
  lineUnit: string;
  redshift: string;
}

const INITIAL: Options = {
  x: ['', ''],
  y: ['', ''],
  channels: ['', ''],
  below: '',
  sigma: '',
  regions: '',
  pixels: '',
  ratio: '',
  noise: false,
  minify: false,
  pad: '',
  line: '',
  lineUnit: '',
  redshift: '',
};

// The units of the lines of spectral axes of wavelengths and frequencies
const LINE_UNITS = { wave: ['Angstrom', 'nm', 'micron'], freq: ['GHz', 'MHz'] };

// The iterations of sigma clipping (as the command line)
const SIGMA_ITERATIONS = 5;

export function Prepare({
  project,
  data,
  files,
  velocity,
  onPrepared,
}: {
  project: string;
  // The array of the data
  data: ArrayInfo;
  files: PrepFiles;
  // The convention of its velocities, if its header leaves it ambiguous
  velocity: Velocity | null;
  onPrepared: (prepared: Prepared) => void;
}) {
  const [options, setOptions] = useState<Options>(INITIAL);
  const submit = useSubmitJob();
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId).data;
  const done = useRef<string | null>(null);
  useEffect(() => {
    if (job?.state === 'complete' && job.files && done.current !== job.id) {
      done.current = job.id;
      onPrepared(job.files);
    }
  }, [job, onPrepared]);
  const set = (changed: Partial<Options>) => setOptions({ ...options, ...changed });
  const shape = data.shape.filter((size) => size > 1);
  const [nx, ny, nchannels] = shape;
  const spectral = spectralKind(data);
  const lineUnit = options.lineUnit || (spectral ? LINE_UNITS[spectral][0] : '');
  const request = taskOptions(options, files, data, shape, spectral, lineUnit, velocity, project);
  const running = submit.isPending || job?.state === 'queued' || job?.state === 'running';
  const reason = typeof request === 'string' ? request : null;
  const prepare = () => {
    if (typeof request === 'string') return;
    submit.mutate({ task: 'prep_pixel_spectra', options: request }, { onSuccess: (submitted) => setJobId(submitted.id) });
  };
  const range = (key: 'x' | 'y' | 'channels', first: number, last: number, label: string) => (
    <span className="range">
      {label}
      <input
        type="number"
        value={options[key][0]}
        placeholder={String(first)}
        aria-label={`The first of ${label}`}
        onChange={(event) => set({ [key]: [event.target.value, options[key][1]] })}
      />
      –
      <input
        type="number"
        value={options[key][1]}
        placeholder={String(last)}
        aria-label={`The last of ${label}`}
        onChange={(event) => set({ [key]: [options[key][0], event.target.value] })}
      />
    </span>
  );
  const number = (key: 'below' | 'sigma' | 'regions' | 'pixels' | 'ratio' | 'pad', label: string, title?: string) => (
    <input
      type="number"
      step="any"
      value={options[key]}
      aria-label={label}
      title={title}
      onChange={(event) => set({ [key]: event.target.value })}
    />
  );
  return (
    <div className="card">
      <h2 className="card-head">Prepare</h2>
      <div className="card-body">
        <p className="note prep-note">Writes a prepared copy into the project; the files above become it.</p>
        <div className="prep">
          <span className="prep-label">crop</span>
          <div className="prep-controls">
            {range('x', 0, nx - 1, 'x')}
            {range('y', 0, ny - 1, 'y')}
            {range('channels', 1, nchannels, 'channels')}
          </div>
          <span className="prep-label">leave out</span>
          <div className="prep-controls">
            <span>
              values below {number('below', 'Leave out the values below')} {data.unit ?? ''}
            </span>
            <span title={`Sigma clipping, ${SIGMA_ITERATIONS} times`}>
              outliers beyond {number('sigma', 'Leave out the outliers beyond this many σ')} σ
            </span>
          </div>
          <span className="prep-label" title="Of the pixels left, those connected (connected-component labelling)">
            regions
          </span>
          <div className="prep-controls">
            <span>keep the largest {number('regions', 'The number of regions kept')}</span>
            <span>of ≥ {number('pixels', 'The pixels of a region kept, at least')} pixels</span>
            <span>≥ {number('ratio', 'The size of a region kept, at least, as a fraction of the largest')} × the largest</span>
          </div>
          {files.file_e === null && (
            <>
              <span className="prep-label">errors</span>
              <label className="prep-controls" title="From the negative values of each channel, before leaving any out">
                <input type="checkbox" checked={options.noise} onChange={(event) => set({ noise: event.target.checked })} />
                from the noise of each channel
              </label>
            </>
          )}
          <span className="prep-label">then</span>
          <div className="prep-controls">
            <label>
              <input type="checkbox" checked={options.minify} onChange={(event) => set({ minify: event.target.checked })} />
              crop to the pixels left
            </label>
            <span>pad by {number('pad', 'Pad by this many empty pixels')} pixels</span>
          </div>
          {spectral && (
            <>
              <span className="prep-label">velocities</span>
              <div className="prep-controls">
                <span title="The velocities of the channels are measured from the line at the redshift">
                  line{' '}
                  <input
                    type="number"
                    step="any"
                    value={options.line}
                    aria-label="The rest of the line"
                    onChange={(event) => set({ line: event.target.value })}
                  />{' '}
                  <select value={lineUnit} aria-label="The unit of the line" onChange={(event) => set({ lineUnit: event.target.value })}>
                    {LINE_UNITS[spectral].map((unit) => (
                      <option key={unit}>{unit}</option>
                    ))}
                  </select>
                </span>
                <span>
                  at redshift{' '}
                  <input
                    type="number"
                    step="any"
                    value={options.redshift}
                    placeholder="0"
                    aria-label="The redshift"
                    onChange={(event) => set({ redshift: event.target.value })}
                  />
                </span>
              </div>
            </>
          )}
        </div>
        <div className="prep-run">
          <button className="primary" disabled={reason !== null || running} onClick={prepare}>
            Prepare
          </button>
          {reason && <span className="why">{reason}</span>}
          {running && <span className="note">Preparing…</span>}
        </div>
        {submit.error && <p className="error">{submit.error.message}</p>}
        {job?.state === 'failed' && <pre className="error prep-output">{lastLines(job.output, 4)}</pre>}
      </div>
    </div>
  );
}

// Whether the channels of an array are of wavelengths or frequencies (or
// neither: velocities, or unknown)
function spectralKind(data: ArrayInfo): 'wave' | 'freq' | null {
  const axis = data.axes.filter((_, i) => data.shape[i] > 1)[2]?.slice(0, 4);
  return axis === 'WAVE' || axis === 'AWAV' ? 'wave' : axis === 'FREQ' ? 'freq' : null;
}

// The options of prep_pixel_spectra from those typed, or why they cannot
// be: the crops as prep takes them (pixels from 0 to the last + 1; the
// first channel typed is 1, as shown), and the rest of the velocities, the
// line at the redshift
function taskOptions(
  options: Options,
  files: PrepFiles,
  data: ArrayInfo,
  shape: number[],
  spectral: 'wave' | 'freq' | null,
  lineUnit: string,
  velocity: Velocity | null,
  project: string,
): Record<string, unknown> | string {
  const value = (text: string) => (text.trim() === '' ? null : Number(text));
  const typed = [
    ...[options.below, options.sigma, options.regions, options.pixels, options.ratio, options.pad],
    ...[options.line, options.redshift, ...options.x, ...options.y, ...options.channels],
  ];
  if (typed.some((text) => Number.isNaN(value(text)))) return 'A value is not a number';
  if (data.ambiguous_velocity && velocity === null) return 'Say whether the velocities are radio or optical';
  const [nx, ny, nchannels] = shape;
  // A crop typed (from the first to the last, numbered from base; blank:
  // the first or the last of the axis), as prep takes it
  const crop = ([first, last]: [string, string], size: number, base: number) => {
    if (first.trim() === '' && last.trim() === '') return null;
    return [(value(first) ?? base) - base, (value(last) ?? size - 1 + base) - base + 1];
  };
  const x = crop(options.x, nx, 0);
  const y = crop(options.y, ny, 0);
  const channels = crop(options.channels, nchannels, 1);
  for (const [crop, size, name] of [[x, nx, 'x'], [y, ny, 'y'], [channels, nchannels, 'channels']] as const) {
    if (crop && !(0 <= crop[0] && crop[0] < crop[1] && crop[1] <= size)) return `The crop of ${name} is out of the cube`;
  }
  let rest: string | null = null;
  if (spectral) {
    const line = value(options.line);
    if (line === null || line <= 0) return 'Give the line, for the velocities of the channels';
    const z = value(options.redshift) ?? 0;
    rest = `${spectral === 'wave' ? line * (1 + z) : line / (1 + z)} ${lineUnit}`;
  }
  const integer = (text: string) => {
    const number = value(text);
    return number === null ? null : Math.round(number);
  };
  return {
    file_d: files.file_d,
    file_e: options.noise ? null : files.file_e,
    file_m: files.file_m,
    roi_spat: x || y ? [...(x ?? [0, nx]), ...(y ?? [0, ny])] : null,
    roi_spec: channels,
    clip_min: value(options.below),
    clip_max: null,
    ccl_lcount: integer(options.regions),
    ccl_pcount: integer(options.pixels),
    ccl_lratio: value(options.ratio),
    sclip_sigma: value(options.sigma),
    sclip_iters: SIGMA_ITERATIONS,
    minify: options.minify,
    nanpad: integer(options.pad),
    dtype: 'float32',
    velocity_rest: rest,
    velocity,
    noise_error: options.noise,
    output_dir: project === '.' ? 'data' : `${project}/data`,
  };
}

function lastLines(text: string, count: number): string {
  return text.trimEnd().split('\n').slice(-count).join('\n');
}
