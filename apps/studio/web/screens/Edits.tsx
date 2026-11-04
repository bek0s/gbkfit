// The edits of the files of a new cube, which prep makes when the dataset
// is added (see gbkfit.tasks.prep.prep_pixel_spectra): crop them (typed,
// or to the selection on the map and the spectrum), make their errors from the
// noise, leave out values (out of a range, outliers, out of the largest
// regions), crop them to the pixels left, and make their channels
// velocities. The project keeps the original files and the prepared
// copies; the dataset is of the copies.
import { useState } from 'react';
import type { ArrayInfo } from '../api/client';
import type { Velocity } from '../api/queries';
import { Panel } from '../components/ui/panel';
import { Button } from '../components/ui/basics';
import { significant } from '../components/format';
import { Check, Choice, NumberField } from '../components/ui/inputs';
import { ValueRange, ValueSlider } from '../components/ui/sliders';

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

// What a file of errors holds (see ErrorKind of gbkfit):
// standard deviations, variances or inverse variances; prep writes
// standard deviations
export type ErrorKind = 'sigma' | 'variance' | 'inverse_variance';

export const ERROR_KINDS: { kind: ErrorKind; title: string }[] = [
  { kind: 'sigma', title: 'σ' },
  { kind: 'variance', title: 'σ²' },
  { kind: 'inverse_variance', title: '1/σ²' },
];

// The edits as typed (blank: none); the crops are inclusive, of pixels
// from 0 and of channels from 1, as shown
export interface Edits {
  x: [string, string];
  y: [string, string];
  channels: [string, string];
  below: string;
  above: string;
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

export const NO_EDITS: Edits = {
  x: ['', ''],
  y: ['', ''],
  channels: ['', ''],
  below: '',
  above: '',
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
export const SIGMA_ITERATIONS = 5;

export const value = (text: string) => (text.trim() === '' ? null : Number(text));

// Whether edits are typed (or checked)
function hasEdits(edits: Edits): boolean {
  const typed = [
    ...[edits.below, edits.above, edits.sigma, edits.regions, edits.pixels, edits.ratio, edits.pad],
    ...[...edits.x, ...edits.y, ...edits.channels],
  ];
  return typed.some((text) => text.trim() !== '') || edits.noise || edits.minify;
}

// Whether the files need prep to become a dataset: edits are typed, or
// their channels are of wavelengths or frequencies (made velocities)
export function needsPrep(edits: Edits, data: ArrayInfo): boolean {
  return hasEdits(edits) || spectralKind(data) !== null;
}

// A crop typed, as the first and the last (inclusive) from 0, or null if
// none; base is the number of the first as typed
export function cropOf([first, last]: [string, string], size: number, base: number): [number, number] | null {
  if (first.trim() === '' && last.trim() === '') return null;
  return [(value(first) ?? base) - base, (value(last) ?? size - 1 + base) - base];
}

// The crops of the edits to draw (pixels and channels from 0, inclusive;
// null: none), for the map and the spectrum
export function cropsOf(edits: Edits, shape: number[]) {
  const [nx, ny, nchannels] = shape;
  const x = cropOf(edits.x, nx, 0);
  const y = cropOf(edits.y, ny, 0);
  return {
    region: x || y ? ([...(x ?? [0, nx - 1]), ...(y ?? [0, ny - 1])] as [number, number, number, number]) : null,
    channels: cropOf(edits.channels, nchannels, 1),
  };
}

// A crop typed, made whole numbers in the axis (from base to its size - 1
// + base), the first before the last; a blank stays blank
export function clampedCrop([first, last]: [string, string], size: number, base: number): [string, string] {
  const clamp = (text: string) => {
    const number = value(text);
    return number === null || Number.isNaN(number) ? null : Math.min(size - 1 + base, Math.max(base, Math.round(number)));
  };
  const [a, b] = [clamp(first), clamp(last)];
  const [low, high] = a !== null && b !== null && a > b ? [b, a] : [a, b];
  return [low === null ? first : String(low), high === null ? last : String(high)];
}

// The edits of a crop dragged on the map (x0, x1, y0, y1, inclusive), or
// cleared (null)
export function withRegion(edits: Edits, region: [number, number, number, number] | null): Edits {
  if (region === null) return { ...edits, x: ['', ''], y: ['', ''] };
  const [x0, x1, y0, y1] = region;
  return { ...edits, x: [String(x0), String(x1)], y: [String(y0), String(y1)] };
}

// The edits of channels dragged on the spectrum (from 0, inclusive), or
// cleared (null)
export function withChannels(edits: Edits, channels: [number, number] | null): Edits {
  return { ...edits, channels: channels === null ? ['', ''] : [String(channels[0] + 1), String(channels[1] + 1)] };
}

// The range of the values kept (null: open at that end), as typed
export function keptOf(edits: Edits): [number | null, number | null] {
  const typed = (text: string) => {
    const number = value(text);
    return number === null || Number.isNaN(number) ? null : number;
  };
  return [typed(edits.below), typed(edits.above)];
}

// Whether the channels of an array are of wavelengths or frequencies (or
// neither: velocities, or unknown)
function spectralKind(data: ArrayInfo): 'wave' | 'freq' | null {
  const axis = data.axes.filter((_, i) => data.shape[i] > 1)[2]?.slice(0, 4);
  return axis === 'WAVE' || axis === 'AWAV' ? 'wave' : axis === 'FREQ' ? 'freq' : null;
}

// The options of prep_pixel_spectra of the edits, or why they cannot be:
// the crops as prep takes them (pixels from the first to the last + 1),
// and the rest of the velocities, the line at the redshift
export function prepOptions(
  edits: Edits,
  files: PrepFiles,
  errorKind: ErrorKind,
  data: ArrayInfo,
  velocity: Velocity | null,
  project: string,
): Record<string, unknown> | string {
  const typed = [
    ...[edits.below, edits.above, edits.sigma, edits.regions, edits.pixels, edits.ratio, edits.pad],
    ...[edits.line, edits.redshift, ...edits.x, ...edits.y, ...edits.channels],
  ];
  if (typed.some((text) => Number.isNaN(value(text)))) return 'An edit is not a number';
  const shape = data.shape.filter((size) => size > 1);
  const [nx, ny, nchannels] = shape;
  const { region, channels } = cropsOf(edits, shape);
  const x = region && [region[0], region[1] + 1];
  const y = region && [region[2], region[3] + 1];
  const z = channels && [channels[0], channels[1] + 1];
  for (const [crop, size, name] of [[x, nx, 'x'], [y, ny, 'y'], [z, nchannels, 'channels']] as const) {
    if (crop && !(0 <= crop[0] && crop[0] < crop[1] && crop[1] <= size)) return `The crop of ${name} is out of the cube`;
  }
  const spectral = spectralKind(data);
  let rest: string | null = null;
  if (spectral) {
    const line = value(edits.line);
    if (line === null || line <= 0) return 'Give the line, for the velocities of the channels';
    const z = value(edits.redshift) ?? 0;
    const unit = edits.lineUnit || LINE_UNITS[spectral][0];
    rest = `${spectral === 'wave' ? line * (1 + z) : line / (1 + z)} ${unit}`;
  }
  const integer = (text: string) => {
    const number = value(text);
    return number === null ? null : Math.round(number);
  };
  return {
    file_d: files.file_d,
    file_e: edits.noise ? null : files.file_e,
    error_kind: edits.noise || files.file_e === null ? 'sigma' : errorKind,
    file_m: files.file_m,
    roi_spat: x && y ? [...x, ...y] : null,
    roi_spec: z,
    clip_min: value(edits.below),
    clip_max: value(edits.above),
    ccl_lcount: integer(edits.regions),
    ccl_pcount: integer(edits.pixels),
    ccl_lratio: value(edits.ratio),
    sclip_sigma: value(edits.sigma),
    sclip_iters: SIGMA_ITERATIONS,
    minify: edits.minify,
    nanpad: integer(edits.pad),
    dtype: 'float32',
    velocity_rest: rest,
    velocity,
    noise_error: edits.noise,
    output_dir: project === '.' ? 'data' : `${project}/data`,
  };
}

// The edits, as a form of aligned rows: a label, its input (or the first
// and the last of a crop), and its unit
export function EditsCard({
  data,
  edits,
  hasErrorFile,
  onChange,
  onCropToSelection,
}: {
  // The array of the data
  data: ArrayInfo;
  edits: Edits;
  // Whether the errors are of a file (else they can be made from the noise)
  hasErrorFile: boolean;
  onChange: (edits: Edits) => void;
  // How the crop becomes the selection on the cube (none: nothing selected)
  onCropToSelection?: () => void;
}) {
  const set = (changed: Partial<Edits>) => onChange({ ...edits, ...changed });
  const [nx, ny, nchannels] = data.shape.filter((size) => size > 1);
  // The noise of the header, the unit of the sliders of values
  const noise = data.rms !== null && data.rms > 0 ? data.rms : null;
  const spectral = spectralKind(data);
  const lineUnit = edits.lineUnit || (spectral ? LINE_UNITS[spectral][0] : '');
  // (a crop is clamped to the cube as its fields are left, the first before
  // the last)
  const crop = (key: 'x' | 'y' | 'channels', label: string, first: number, last: number, unit: string) => (
    <>
      <span className="edit-label">{label}</span>
      <span className="edit-range" onBlur={() => set({ [key]: clampedCrop(edits[key], last - first + 1, first) })}>
        <NumberField
          value={edits[key][0]}
          min={first}
          max={last}
          placeholder={String(first)}
          label={`The first of ${label}`}
          onChange={(text) => set({ [key]: [text, edits[key][1]] })}
        />
        <span>–</span>
        <NumberField
          value={edits[key][1]}
          min={first}
          max={last}
          placeholder={String(last)}
          label={`The last of ${label}`}
          onChange={(text) => set({ [key]: [edits[key][0], text] })}
        />
      </span>
      <span className="edit-unit">{unit}</span>
    </>
  );
  // The value of a slider in units of the noise, in the unit of the data
  // (blank: open)
  const fromNoise = (at: number | null) => (at === null || noise === null ? '' : significant(at * noise, 3));
  const inNoise = (text: string) => {
    const number = value(text);
    return number === null || Number.isNaN(number) || noise === null ? null : number / noise;
  };
  const number = (key: 'sigma' | 'regions' | 'pixels' | 'ratio' | 'pad', label: string, unit: string, title?: string) => (
    <>
      <span className="edit-label" title={title}>
        {label}
      </span>
      <NumberField value={edits[key]} label={label} title={title} onChange={(text) => set({ [key]: text })} />
      <span className="edit-unit">{unit}</span>
    </>
  );
  // (the values left out are open at first if any are typed, and then as
  // opened or closed: never closed by an edit, e.g. a handle to the end)
  const [leaveOut, setLeaveOut] = useState(() =>
    [edits.below, edits.above, edits.sigma, edits.regions, edits.pixels, edits.ratio].some((text) => text.trim() !== ''),
  );
  return (
    <Panel
      id="dataset-edits"
      title="Edits"
      actions={
        hasEdits(edits) && (
          <Button kind="small" title="Clear the edits" onClick={() => onChange({ ...NO_EDITS, line: edits.line, lineUnit: edits.lineUnit, redshift: edits.redshift })}>
            clear
          </Button>
        )
      }
    >
      <div className="edits">
        <div className="edit-group edit-group-action">
          crop
          <Button
            kind="small"
            disabled={onCropToSelection === undefined}
            title={
              onCropToSelection === undefined
                ? 'Select a region of the map, or channels of the spectrum, to crop to'
                : 'Crop to the region and the channels selected on the cube'
            }
            onClick={() => onCropToSelection?.()}
          >
            Crop to selection
          </Button>
        </div>
        {crop('x', 'x', 0, nx - 1, 'pixels')}
        {crop('y', 'y', 0, ny - 1, 'pixels')}
        {crop('channels', 'channels', 1, nchannels, '')}
        {spectral && (
          <>
            <div className="edit-group">velocities</div>
            <span className="edit-label" title="The velocities of the channels are measured from the line at the redshift">
              Line
            </span>
            <NumberField value={edits.line} label="The rest of the line" onChange={(text) => set({ line: text })} />
            <Choice
              items={LINE_UNITS[spectral].map((unit) => ({ value: unit, label: unit }))}
              value={lineUnit}
              label="The unit of the line"
              onChange={(unit) => set({ lineUnit: unit })}
            />
            <span className="edit-label">At redshift</span>
            <NumberField value={edits.redshift} placeholder="0" label="The redshift" onChange={(text) => set({ redshift: text })} />
            <span />
          </>
        )}
        {!hasErrorFile && (
          <>
            <div className="edit-group">errors</div>
            <div className="edit-check">
              <Check
                checked={edits.noise}
                label="From the noise of each channel"
                title="The robust noise of the negative values of each channel, before any are left out"
                onChange={(noise) => set({ noise })}
              />
            </div>
          </>
        )}
        <details className="edit-more" open={leaveOut} onToggle={(event) => setLeaveOut(event.currentTarget.open)}>
          <summary className="edit-group">leave out values</summary>
          <div className="edits">
            <span className="edit-label" title="Leave out the values out of this range (blank: no limit), shown blank on the channel map">
              Keep values
            </span>
            <span className="edit-range">
              <NumberField value={edits.below} placeholder="−∞" label="Keep the values from" onChange={(below) => set({ below })} />
              <span>–</span>
              <NumberField value={edits.above} placeholder="∞" label="Keep the values up to" onChange={(above) => set({ above })} />
            </span>
            <span className="edit-unit">{data.unit ?? ''}</span>
            {noise !== null && (
              <div className="edit-wide">
                <ValueRange
                  min={-3}
                  max={15}
                  step={0.1}
                  value={[inNoise(edits.below), inNoise(edits.above)]}
                  mark={0}
                  format={(at) => `${significant(at, 2)}σ`}
                  label="The values kept, in units of the noise of the header"
                  onChange={([low, high]) => set({ below: fromNoise(low), above: fromNoise(high) })}
                />
              </div>
            )}
            {number('sigma', 'outliers beyond', 'σ of them all', `Sigma clipping of all the values, ${SIGMA_ITERATIONS} times, on both sides`)}
            <div className="edit-wide">
              <ValueSlider
                min={0}
                max={10}
                step={0.5}
                value={value(edits.sigma)}
                format={(at) => `${at}σ`}
                label="Sigma clipping: the values beyond this many standard deviations of them all"
                onChange={(at) => set({ sigma: at === null ? '' : String(at) })}
              />
            </div>
            {edits.sigma.trim() !== '' && (
              <p className="edit-warn">
                Sigma clipping leaves out the values far from the mean of all of them: in a cube of a line, its emission too. It is for
                images with bad pixels.
              </p>
            )}
            <span className="edit-sub" title="Of the pixels left, those connected (connected-component labelling)">
              of the pixels left, keep the regions
            </span>
            {number('regions', 'largest', 'regions', 'The number of regions kept, the largest first')}
            {number('pixels', 'of at least', 'pixels', 'The pixels of a region kept, at least')}
            {number('ratio', 'and at least', '× the largest', 'The size of a region kept, at least, as a fraction of the largest')}
            <div className="edit-check">
              <Check checked={edits.minify} label="Crop to the pixels left" title="Crop the files to the pixels left" onChange={(minify) => set({ minify })} />
            </div>
            {number('pad', 'pad by', 'pixels', 'Pad the files by this many empty pixels on each side')}
          </div>
        </details>
      </div>
      {needsPrep(edits, data) && (
        <p className="note edits-note">Adding writes an edited copy of the files into the project; the originals stay.</p>
      )}
    </Panel>
  );
}
