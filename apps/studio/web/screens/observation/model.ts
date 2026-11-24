// The model of an observation of the Studio, without React: its options
// (those of the library: see gbkfit.observation), what each stage of it
// is at a glance, its checks, and the sections of the dock where the
// problems the library finds in it are fixed. An observation stands
// alone, on a sampling of its own: the data it is fitted to are paired
// with it by the setups (their section datasets).
import { membersOf, propertiesOf, typeOf, type Defs, type Options, type Schema } from '../../components/SchemaForm';

// The stages of an observation on the sky, in the order of the path of the
// light, from the galaxies to its mock data, as the dock and the strip
// have them (by the id of their section)
export const STAGES = [
  { key: 'sees', title: 'Sees', section: 'ob-sees' },
  { key: 'instrument', title: 'Instrument', section: 'ob-instrument' },
  { key: 'records', title: 'Records', section: 'ob-records' },
  { key: 'noise', title: 'Noise', section: 'ob-noise' },
] as const;

// The options of a sky observation, by the section that sets them: what it
// sees; how accurately and fast its model is made (folded); the others are
// those its sections set by hand, and the rest are its other options
export const LIGHT_KEYS = ['galaxies', 'components', 'lines', 'continuum', 'band'];
export const ACCURACY_KEYS = ['oversampling', 'depth', 'source_grid', 'dtype', 'fft_dtype', 'name'];
export const HANDLED_KEYS = ['type', 'instrument', 'noise', 'unit', ...LIGHT_KEYS, ...ACCURACY_KEYS];

// The kinds of observations whose data are what their instrument records
// plus the noise, in one item (whose errors are then the noise of their mock
// data, by default, paired with data that have errors; see the library's
// noise_of): not those measured from it (e.g. moments of a cube)
const RECORDED = ['pixel_spectra', 'pixel_brightness', 'slit_spectra', 'slitless_spectra', 'visibilities'];

export function isRecorded(type: unknown): boolean {
  return RECORDED.includes(String(type));
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
  return noise;
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

// A check of an observation: passed, failed, or a note
export interface Check {
  ok: boolean | null;
  text: string;
}

export function checksOf({
  options,
  psf,
  lsfWidth,
  noise,
}: {
  options: Options;
  psf: Ellipse | null;
  lsfWidth: number | null;
  // The noise of mock data in a few words (see noiseText)
  noise: string;
}): Check[] {
  const checks: Check[] = [];
  const instrument = objectAt(options, 'instrument');
  if (instrument.psf == null) checks.push({ ok: null, text: 'No PSF: a perfect instrument on the sky.' });
  const spectral = objectAt(options, 'spectral');
  const width = typeof spectral.step === 'number' ? Math.abs(spectral.step) : null;
  if (isSpectral(options.type)) {
    if (instrument.lsf == null) checks.push({ ok: null, text: 'No LSF: a perfect spectrograph.' });
    else if (lsfWidth !== null && width !== null) {
      checks.push({ ok: null, text: `The LSF is ${fixed(lsfWidth)} km/s across (FWHM): ${fixed(lsfWidth / width)} channels.` });
    }
  }
  // (the sampling of the PSF: at least two pixels across its FWHM)
  const pixel = typeof options.step === 'number' ? options.step : Array.isArray(options.step) ? Number(options.step[0]) : null;
  if (psf && pixel) {
    const across = psf.major / pixel;
    checks.push(across >= 2 ? { ok: true, text: `The PSF spans ${fixed(across)} pixels: well sampled.` } : { ok: false, text: `The PSF spans ${fixed(across)} pixels: under-sampled (at least 2 are needed).` });
  }
  if (options.noise != null) checks.push({ ok: null, text: `Mock noise: ${noise}.` });
  else if (isRecorded(options.type)) checks.push({ ok: null, text: 'No noise for mock data: they take the errors of the data a setup pairs it with, if any (or give it one in Noise).' });
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

// The section of the dock where a problem the library finds is fixed, by
// the option of its path: what it sees, its instrument, its noise, its
// accuracy, its other options (others: their keys), or else its kind and
// sampling (as the problems of the whole observation)
export function sectionOf(path: (string | number)[], others: string[]): string {
  const key = String(path[0] ?? '');
  if (LIGHT_KEYS.includes(key)) return 'ob-sees';
  if (key === 'instrument') return 'ob-instrument';
  if (key === 'noise') return 'ob-noise';
  if (ACCURACY_KEYS.includes(key)) return 'ob-accuracy';
  if (others.includes(key)) return 'ob-more';
  return 'ob-records';
}
