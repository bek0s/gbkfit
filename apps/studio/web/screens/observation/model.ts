// The model of an observation of the Studio, without React: its options
// (those of the library: see gbkfit.observation), what each stage of it
// is at a glance, and the checks of it against its data.
import type { ArrayInfo } from '../../api/client';
import { membersOf, propertiesOf, typeOf, type Defs, type Options, type Schema } from '../../components/SchemaForm';

// The stages of an observation on the sky, in the order of the path of the
// light, from the galaxies to the data, as the dock and the strip have
// them (by the id of their section)
export const STAGES = [
  { key: 'sees', title: 'Sees', section: 'ob-sees' },
  { key: 'instrument', title: 'Instrument', section: 'ob-instrument' },
  { key: 'records', title: 'Records', section: 'ob-records' },
  { key: 'noise', title: 'Noise', section: 'ob-noise' },
  { key: 'data', title: 'Data', section: 'ob-data' },
] as const;

// The options of a sky observation, by the section that sets them: what it
// sees; how its data meet the model (with its data); how accurately and
// fast its model is made (folded); the others are those its sections set
// by hand
export const LIGHT_KEYS = ['galaxies', 'components', 'lines', 'continuum', 'band'];
export const MEETING_KEYS = ['offsets', 'flux_scale', 'pixel_weights', 'smooth_weights', 'median_size'];
export const ACCURACY_KEYS = ['oversampling', 'depth', 'source_grid', 'dtype', 'fft_dtype', 'name'];
export const HANDLED_KEYS = ['type', 'dataset', 'instrument', 'noise', 'unit', ...LIGHT_KEYS, ...MEETING_KEYS, ...ACCURACY_KEYS];

// The kinds of observations whose data are what their instrument records
// plus the noise, in one item (whose errors are then the noise of their mock
// data, by default; see the library's noise_of): not those measured from it
// (e.g. moments of a cube)
const RECORDED = ['pixel_spectra', 'pixel_brightness', 'slit_spectra', 'slitless_spectra', 'visibilities'];

export function isRecorded(type: unknown): boolean {
  return RECORDED.includes(String(type));
}

// Whether a dataset (its piece) has errors: of an item, or of its one item
export function datasetHasErrors(dataset: Options | undefined): boolean {
  if (!dataset) return false;
  if (dataset.error != null) return true;
  return Object.values(dataset).some((item) => isObject(item) && item.error != null);
}

// The noise of mock data of an observation in a few words, with the name
// of the noise of the project it is (its options, of the project or in
// place; undefined: not read yet)
export function noiseText(noise: unknown, options: Options | undefined, titleOf: (family: string, type: unknown) => string): string {
  if (noise == null) return 'None';
  if (!options) return typeof noise === 'string' ? noise : '…';
  return `${typeof noise === 'string' ? `${noise}: ` : ''}${noiseOf(options, titleOf)}`;
}

// A noise of its options in a few words: its kind, its level and how it
// is correlated (e.g. 'Gaussian 0.01 · as the PSF')
export function noiseOf(options: Options, titleOf: (family: string, type: unknown) => string): string {
  const kind = titleOf('noise', options.type);
  const level = typeof options.rms === 'number' ? ` ${options.rms}` : options.error != null ? ' of an error map' : '';
  const spatial = options.spatial_correlation;
  const correlated = spatial === 'instrument' ? ' · as the PSF' : spatial != null ? ' · correlated on the sky' : '';
  return `${kind}${level}${correlated}`;
}

// The two numbers of an option of a pair given as one or two (e.g. the
// size of an observation: 64 or [64, 48]; 1 for one not given)
export function pairOf(value: unknown): [number, number] {
  const [a, b] = Array.isArray(value) ? value : [value, value];
  return [Number(a) || 1, Number(b ?? a) || 1];
}

// The kinds of observations (and datasets) whose data have a spectral axis
const SPECTRAL = ['pixel_spectra', 'pixel_moments', 'slit_spectra', 'slitless_spectra', 'region_spectra', 'region_moments', 'visibilities'];

export function isSpectral(type: unknown): boolean {
  return SPECTRAL.includes(String(type));
}

export function isObject(value: unknown): value is Options {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// An option of an observation as an object (none: an empty one)
export function objectAt(options: Options, key: string): Options {
  const value = options[key];
  return isObject(value) ? value : {};
}

// The text of an option (none if not given, or not text)
export function textAt(options: Options, key: string): string | null {
  const value = options[key];
  return typeof value === 'string' ? value : null;
}

// A value set or removed (undefined) in options
export function withKey(options: Options, key: string, value: unknown): Options {
  const { [key]: _old, ...rest } = options;
  return value === undefined ? rest : { ...rest, [key]: value };
}

// What a stage of an observation is, at a glance (the noise as given; see
// noiseText)
export function glanceOf(key: (typeof STAGES)[number]['key'], options: Options, titleOf: (family: string, type: unknown) => string, noise: string): string {
  if (key === 'sees') {
    const galaxies = options.galaxies;
    const lines = Array.isArray(options.lines) && options.lines.length ? ` · ${options.lines.join(', ')}` : '';
    return `${Array.isArray(galaxies) && galaxies.length ? galaxies.join(', ') : 'All of the setup'}${lines}`;
  }
  if (key === 'instrument') {
    const instrument = objectAt(options, 'instrument');
    const parts = (['psf', 'lsf'] as const).flatMap((part) => {
      const value = instrument[part];
      if (value == null) return [];
      return [`${part.toUpperCase()} ${typeof value === 'string' ? value : titleOf(part, (value as Options).type)}`];
    });
    return parts.join(' · ') || 'A perfect one';
  }
  if (key === 'records') return titleOf('observation', options.type);
  if (key === 'noise') return noise;
  return typeof options.dataset === 'string' ? options.dataset : options.dataset ? 'Given in place' : 'None: a model';
}

// The ellipse of a PSF or a beam at half its maximum: the FWHM of its
// major and minor axes (arcsec), and the position angle of its major axis
// (degrees, east of north)
export interface Ellipse {
  major: number;
  minor: number;
  angle: number;
}

// The ellipse of a PSF as the server measures it (see PSFViewInfo.axes)
export function psfEllipse(axes: [number, number, number | null] | null | undefined): Ellipse | null {
  return axes ? { major: axes[0], minor: axes[1], angle: axes[2] ?? 0 } : null;
}

// The beam of the header of the data, as an ellipse (none: none, or one
// for each channel)
export function headerBeam(array: ArrayInfo | undefined): Ellipse | null {
  const beam = array?.beam;
  if (!beam || Array.isArray(beam.bmaj)) return null;
  return { major: Number(beam.bmaj), minor: Number(beam.bmin), angle: Number(beam.bpa) };
}

// A check of an observation against its data: passed, failed, or a note
export interface Check {
  ok: boolean | null;
  text: string;
}

export function checksOf({
  options,
  datasets,
  datasetType,
  data,
  psf,
  lsfWidth,
  errors,
  noise,
  titleOf,
}: {
  options: Options;
  datasets: string[];
  // The type of the dataset named (undefined: not read, or none)
  datasetType: string | undefined;
  data: ArrayInfo | undefined;
  psf: Ellipse | null;
  lsfWidth: number | null;
  // Whether the data have errors, and the noise of mock data in a few
  // words (see noiseText)
  errors: boolean;
  noise: string;
  titleOf: (family: string, type: unknown) => string;
}): Check[] {
  const checks: Check[] = [];
  const instrument = objectAt(options, 'instrument');
  const named = typeof options.dataset === 'string' ? options.dataset : null;
  if (options.dataset == null) checks.push({ ok: null, text: 'No data: the observation makes a model (or mock data).' });
  else if (named !== null && !datasets.includes(named)) checks.push({ ok: false, text: `The project has no dataset ${named}.` });
  else if (datasetType !== undefined) {
    checks.push(
      datasetType === options.type
        ? { ok: true, text: `The observation and the data are both ${titleOf('observation', options.type)}.` }
        : { ok: false, text: `The observation is ${titleOf('observation', options.type)}, but the data are ${titleOf('observation', datasetType)}.` },
    );
  }
  const beam = headerBeam(data);
  if (instrument.psf == null) {
    checks.push(
      beam
        ? { ok: false, text: `The data have a beam of ${fixed(beam.major)}″ × ${fixed(beam.minor)}″, but the observation has no PSF.` }
        : { ok: null, text: 'No PSF: a perfect instrument on the sky.' },
    );
  } else if (beam && psf) {
    // (within 5%: the beam; beyond 20%: the model is smoothed by another,
    // and its light per beam is per beam of the data, as the library warns)
    const ratio = psf.major / beam.major;
    checks.push(
      Math.abs(ratio - 1) < 0.05
        ? { ok: true, text: `The PSF (${fixed(psf.major)}″) matches the beam of the data (${fixed(beam.major)}″).` }
        : Math.abs(ratio - 1) > 0.2
          ? { ok: false, text: `The PSF (${fixed(psf.major)}″) differs from the beam of the data (${fixed(beam.major)}″): the light of the model per beam is per beam of the data.` }
          : { ok: null, text: `The PSF is ${fixed(psf.major)}″ across; the beam of the data, ${fixed(beam.major)}″.` },
    );
  }
  const width = data?.step && data.step.length === 3 ? Math.abs(data.step[2]) : null;
  if (isSpectral(options.type)) {
    if (instrument.lsf == null) checks.push({ ok: null, text: 'No LSF: a perfect spectrograph.' });
    else if (lsfWidth !== null && width !== null) {
      checks.push({ ok: null, text: `The LSF is ${fixed(lsfWidth)} km/s across (FWHM): ${fixed(lsfWidth / width)} channels of the data.` });
    }
  }
  // (the sampling of the PSF and the LSF: at least two pixels or channels
  // across their FWHM)
  const pixel = data?.step ? Math.abs(data.step[0]) : typeof options.step === 'number' ? options.step : Array.isArray(options.step) ? Number(options.step[0]) : null;
  if (psf && pixel) {
    const across = psf.major / pixel;
    checks.push(across >= 2 ? { ok: true, text: `The PSF spans ${fixed(across)} pixels: well sampled.` } : { ok: false, text: `The PSF spans ${fixed(across)} pixels: under-sampled (at least 2 are needed).` });
  }
  if (data?.step) {
    checks.push({ ok: null, text: `The data have pixels of ${fixed(Math.abs(data.step[0]))}″${width !== null ? ` and channels of ${fixed(width)} km/s` : ''}.` });
  }
  // (of the data: fits need their errors; of mock data: their noise)
  if (named !== null && datasetType !== undefined && !errors) checks.push({ ok: false, text: 'The data have no errors: their fits need them (give the dataset its errors).' });
  if (options.noise != null) checks.push({ ok: null, text: `Mock noise: ${noise}.` });
  else if (options.dataset != null && isRecorded(options.type) && errors) checks.push({ ok: null, text: 'Mock noise: the errors of the data, as Gaussian noise.' });
  else checks.push({ ok: null, text: 'No noise for mock data: a setup cannot draw them (give it one in Noise).' });
  return checks;
}

function fixed(value: number): string {
  return Number(value.toPrecision(3)).toString();
}

// The member of the observation of a type (see the schema)
export function observationMember(defs: Defs, type: unknown): Schema | undefined {
  return membersOf(defs.observation ?? {}, defs).find((m) => typeOf(m) === type);
}

// The options of the grid of a kind on the sky
const GRID = ['size', 'step', 'rpix', 'rval', 'rota', 'slit_pa'];

// The options of every observation on the sky: those that all the kinds of
// data share, but their grids (the others are of each kind: its grid, its
// channels...)
export function skyKeys(defs: Defs): string[] {
  const kinds = membersOf(defs.observation ?? {}, defs).filter((m) => typeOf(m) !== 'profile');
  const keys = kinds.map((m) => Object.keys(propertiesOf(m)));
  return keys.length ? keys[0].filter((key) => !GRID.includes(key) && keys.every((own) => own.includes(key))) : [];
}

// The options of a kind required without data, which the data give (see
// the schema: those required unless the dataset is given)
export function requiredWithoutData(member: Schema | undefined): string[] {
  for (const condition of (member?.allOf as Schema[] | undefined) ?? []) {
    const [ifData, without] = (condition.anyOf as Schema[] | undefined) ?? [];
    if ((ifData?.required as string[] | undefined)?.[0] === 'dataset') return (without?.required as string[] | undefined) ?? [];
  }
  return [];
}

// The options of a kind that its data give: those required without them,
// and with its size, the rest of its grid (see GRID)
export function dataOptions(member: Schema | undefined): string[] {
  const needed = requiredWithoutData(member);
  return needed.includes('size') ? [...new Set([...GRID, ...needed])] : needed;
}

// The options of an observation that its data give, but given too (they
// serve only without data)
export function gridOptions(defs: Defs, options: Options): string[] {
  return dataOptions(observationMember(defs, options.type)).filter((key) => options[key] != null);
}

// Whether the data of a type have channels (those of the model with them)
export function dataHaveChannels(type: string | undefined): boolean {
  return type !== undefined && isSpectral(type) && !['pixel_moments', 'region_moments'].includes(type);
}

const NAMES: Record<string, string> = { size: 'size', spectral: 'channels', regions: 'regions', coverage: 'coverage', fov: 'field of view', trace: 'trace' };

// What an observation needs before the library takes it (by the section
// of the dock where it is given), the most basic first
export function needsOf({
  options,
  defs,
  datasets,
  datasetType,
  titleOf,
}: {
  options: Options;
  defs: Defs;
  datasets: string[];
  datasetType: string | undefined;
  titleOf: (family: string, type: unknown) => string;
}): { text: string; section: string }[] {
  const needs: { text: string; section: string }[] = [];
  const withData = options.dataset != null;
  const member = observationMember(defs, options.type);
  if (typeof options.dataset === 'string' && !datasets.includes(options.dataset)) needs.push({ section: 'ob-data', text: `The project has no dataset ${options.dataset}` });
  if (withData && datasetType !== undefined && datasetType !== options.type) {
    needs.push({ section: 'ob-records', text: `The observation is not of the type of the data (${titleOf('observation', datasetType)})` });
  }
  const required = [...((member?.required as string[] | undefined) ?? []), ...(withData ? [] : requiredWithoutData(member))];
  const missing = required.filter((key) => key !== 'type' && options[key] == null);
  if (missing.length) needs.push({ section: 'ob-records', text: `Give the observation its ${missing.map((key) => NAMES[key] ?? key).join(', ')}` });
  else if (member && 'spectral' in propertiesOf(member) && isSpectral(options.type) && !(withData && dataHaveChannels(datasetType))) {
    const spectral = objectAt(options, 'spectral');
    if (spectral.size == null || spectral.step == null) needs.push({ section: 'ob-records', text: 'Give the channels of the model (their number and width)' });
  }
  return needs;
}
