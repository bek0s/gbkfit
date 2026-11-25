// The model of an observation of the Studio, without React: its options
// (those of the library: see gbkfit.observation), in the blocks of its
// sheet (its kind, its sampling, its instrument, what it sees, the noise
// of its mock data, and more), and what each is in a few words. An
// observation stands alone, on a sampling of its own: the data it is
// fitted to are paired with it by the fits.
import { membersOf, propertiesOf, typeOf, type Defs, type Options, type Schema } from '../../components/SchemaForm';
import { firstSentence } from '../galaxy/schema';

// The options of an observation by the block of the sheet that sets them
// (which block shows an option; every rule of them is the library's):
// what it sees; its instrument and the noise of its mock data; how its
// model is matched to the data a fit pairs it with; and how accurately
// and fast it is made. The others of its kind (e.g. its pixels, its
// channels, its regions, its UV coverage) are its sampling.
export const SEES_KEYS = ['lines', 'continuum', 'band'];
export const MATCHING_KEYS = ['unit', 'offsets', 'flux_scale', 'pixel_weights', 'smooth_weights', 'median_size', 'name'];
export const ACCURACY_KEYS = ['oversampling', 'depth', 'source_grid', 'dtype', 'fft_dtype'];
const GROUPED = new Set(['type', 'instrument', 'noise', ...SEES_KEYS, ...MATCHING_KEYS, ...ACCURACY_KEYS]);

// The options of the sampling that place it on the sky (its reference
// pixel and position there, its rotation): of a fold of their own, under
// the others
export const PLACE_KEYS = ['rpix', 'rval', 'rota', 'sky_rval', 'slit_pa'];

// The pairs of the sky whose x and y the sheet links (one value typed sets
// both): the number and the size of pixels, a field of view; not those of
// two coordinates (a reference pixel, RA and Dec)
export const LINKED_PAIRS = ['size', 'step', 'fov'];

// The options of a spectral axis shown first (its channels: how many, how
// wide, where, and the convention of their velocities); the others in a
// fold
export const CHANNEL_KEYS = ['size', 'step', 'rval', 'convention'];

// The member of the observation of a type (its schema)
export function observationMember(defs: Defs | undefined, type: unknown): Schema | undefined {
  if (!defs) return undefined;
  return membersOf(defs.observation ?? {}, defs).find((m) => typeOf(m) === type);
}

// The kinds of observations, in the order of the library: each its type,
// title and what it is in a line (the first clause of its description)
export function kindsOf(defs: Defs | undefined): { type: string; title: string; line: string }[] {
  if (!defs) return [];
  const seen = new Set<string>();
  return membersOf(defs.observation ?? {}, defs).flatMap((m) => {
    const type = typeOf(m);
    if (type === undefined || seen.has(type)) return [];
    seen.add(type);
    return [{ type, title: typeof m.title === 'string' ? m.title : type, line: kindLine(m.description) }];
  });
}

// What a kind is in a line: the first clause of its description (to its
// first colon), else its first sentence
export function kindLine(description: unknown): string {
  if (typeof description !== 'string') return '';
  const sentence = firstSentence(description);
  const clause = sentence.split(':')[0].replace(/\.$/, '');
  return clause.trim();
}

// The form of a member of a family (e.g. a Gaussian PSF by its dispersion,
// or by the axes of a beam) that options are of: the first of their type
// whose required options they have, else the first of their type; and
// the forms of their type
export function formsOf(defs: Defs, family: string, options: Options): { form: Schema | undefined; forms: Schema[] } {
  const forms = membersOf(defs[family] ?? {}, defs).filter((m) => typeOf(m) === options.type);
  const form = forms.find((m) => ((m.required as string[] | undefined) ?? []).every((key) => key === 'type' || key in options)) ?? forms[0];
  return { form, forms };
}

// The options of the sampling of a kind (see the module), in the order of
// its schema
export function samplingKeys(member: Schema | undefined): string[] {
  return member ? Object.keys(propertiesOf(member)).filter((key) => !GROUPED.has(key)) : [];
}

// Whether a kind has an option (by its schema)
export function hasOption(member: Schema | undefined, key: string): boolean {
  return member !== undefined && key in propertiesOf(member);
}

// An option given another kind: the options of the observation that the
// new kind has too are kept (what it sees, its instrument, its noise,
// and e.g. its pixels), the others dropped
export function retypedObservation(options: Options, member: Schema | undefined, type: string): Options {
  const keys = member ? Object.keys(propertiesOf(member)) : [];
  return { ...Object.fromEntries(Object.entries(options).filter(([key]) => key !== 'type' && keys.includes(key))), type };
}

// The number of the options of keys that are given
export function givenCount(options: Options, keys: string[]): number {
  return keys.filter((key) => options[key] != null).length;
}

// The noise of mock data of its options in a few words: its kind, its
// level and how it is correlated (e.g. 'Gaussian 0.01 · as the PSF')
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

// An option set or removed at a path of options (an object made on the
// way, one left empty removed)
export function withPath(options: Options, path: string[], value: unknown): Options {
  if (path.length === 1) return withKey(options, path[0], value);
  const inner = withPath(objectAt(options, path[0]), path.slice(1), value);
  return withKey(options, path[0], Object.keys(inner).length ? inner : undefined);
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

// A number in a few digits (e.g. of a FWHM)
export function short(value: number, digits = 3): string {
  return String(Number(value.toPrecision(digits)));
}
