// The model of a dataset of the Studio, without React: its kinds, the
// draft that the dataset screen edits (its items, the spectral axis and
// the edits of prep), what a draft still needs before it can be a
// dataset, and the conversions between a draft and what the server takes
// and gives (the options of a dataset, of a job of prep, and the record of
// a preparation).
//
// A dataset holds items: a cube, a flux map, or moment maps (one item per
// map). An item is a file of data, with its errors (a file, or one value)
// and its mask, each an array of a FITS file.
import type { ArrayInfo, CatalogueLine } from '../../api/client';
import type { Velocity } from '../../api/queries';

export type Kind = 'cube' | 'moments' | 'image';
export type Role = 'data' | 'error' | 'mask';
export type ErrorKind = 'sigma' | 'variance' | 'inverse_variance';
export type Colours = 'sequential' | 'median' | 'zero';

export const ROLES: { role: Role; title: string }[] = [
  { role: 'data', title: 'Data' },
  { role: 'error', title: 'Errors' },
  { role: 'mask', title: 'Mask' },
];

export const ERROR_KINDS: { kind: ErrorKind; title: string; text: string }[] = [
  { kind: 'sigma', title: 'σ', text: 'Standard deviations' },
  { kind: 'variance', title: 'σ²', text: 'Variances' },
  { kind: 'inverse_variance', title: '1/σ²', text: 'Inverse variances' },
];

// A place for an item in a dataset of a kind: its key in the options of
// the dataset (moment maps) or in the draft, how the page names it, and
// how the colours of its map run
export interface Slot {
  key: string;
  title: string;
  short: string;
  colours: Colours;
}

// The tasks of prep of the files of each kind (see gbkfit.tasks.prep)
export type PrepTask = 'prep_pixel_spectra' | 'prep_pixel_moments' | 'prep_pixel_brightness';

export interface KindInfo {
  kind: Kind;
  // The type of the dataset in GBKFIT, and the task of prep of its files
  type: string;
  task: PrepTask;
  title: string;
  // What the data of an item are, in a sentence ("the cube"), and their
  // number of axes
  what: string;
  ndim: 2 | 3;
  slots: Slot[];
  // The items a new dataset starts with
  first: string[];
}

const MOMENT_SLOTS: Slot[] = [
  { key: 'moment0', title: 'Intensity', short: '0', colours: 'sequential' },
  { key: 'moment1', title: 'Velocity', short: '1', colours: 'median' },
  { key: 'moment2', title: 'Dispersion', short: '2', colours: 'sequential' },
  ...[3, 4, 5, 6, 7].map((n): Slot => ({ key: `moment${n}`, title: `Moment ${n}`, short: String(n), colours: 'zero' })),
  { key: 'h3', title: 'Gauss-Hermite h3', short: 'h3', colours: 'zero' },
  { key: 'h4', title: 'Gauss-Hermite h4', short: 'h4', colours: 'zero' },
];

export const KINDS: Record<Kind, KindInfo> = {
  cube: {
    kind: 'cube',
    type: 'pixel_spectra',
    task: 'prep_pixel_spectra',
    title: 'Cube',
    what: 'the cube',
    ndim: 3,
    slots: [{ key: 'cube', title: 'Cube', short: 'C', colours: 'sequential' }],
    first: ['cube'],
  },
  image: {
    kind: 'image',
    type: 'pixel_brightness',
    task: 'prep_pixel_brightness',
    title: 'Flux map',
    what: 'the flux map',
    ndim: 2,
    slots: [{ key: 'image', title: 'Flux', short: 'F', colours: 'sequential' }],
    first: ['image'],
  },
  moments: {
    kind: 'moments',
    type: 'pixel_moments',
    task: 'prep_pixel_moments',
    title: 'Moment maps',
    what: 'the maps',
    ndim: 2,
    slots: MOMENT_SLOTS,
    first: ['moment0', 'moment1', 'moment2'],
  },
};

export function kindOfType(type: unknown): Kind | undefined {
  return (Object.keys(KINDS) as Kind[]).find((kind) => KINDS[kind].type === type);
}

export function isKind(key: string): key is Kind {
  return key in KINDS;
}

// The title of a slot in a sentence or a heading ("Moment 1 · Velocity")
export function slotTitle(slot: Slot, kind: Kind): string {
  if (kind !== 'moments' || slot.short.startsWith('h') || slot.title.startsWith('Moment')) return slot.title;
  return `Moment ${slot.short} · ${slot.title}`;
}

// ---------------------------------------------------------------------------
// The draft

// An array of a file of the workspace: by its name, or by its index if it
// has none of its own; none for a file just chosen (the likeliest array)
export interface FileArray {
  file: string;
  array?: string | number;
}

export interface Item {
  files: Partial<Record<Role, FileArray>>;
  // One error for all the values (as typed), instead of a file of errors
  errorValue: string | null;
  // What the file of errors holds (null: as the name of its array says)
  errorKind: ErrorKind | null;
}

export const EMPTY_ITEM: Item = { files: {}, errorValue: null, errorKind: null };

// The spectral axis of a cube, as the user gives it: what the axis is
// where its header gives it a type FITS does not define (e.g. CALIFA's
// WAVELENGTH: never taken from its name, see SPECTRAL_TYPES), the
// convention of velocities a header leaves ambiguous, and the line (and
// the redshift it is seen at) that makes channels of wavelengths or
// frequencies velocities
export interface Spectral {
  type: SpectralType | null;
  convention: Velocity | null;
  line: string;
  unit: string;
  redshift: string;
}

export const NO_SPECTRAL: Spectral = { type: null, convention: null, line: '', unit: '', redshift: '' };

// What a spectral axis whose header gives it a type FITS does not define
// can be said to be (the types of FITS that GBKFIT reads it as; see
// gbkfit.grid.SPECTRAL_TYPES)
export type SpectralType = 'WAVE' | 'AWAV' | 'WAVE-LOG' | 'AWAV-LOG' | 'FREQ' | 'VRAD' | 'VOPT';
export const SPECTRAL_TYPES: { type: SpectralType; text: string }[] = [
  { type: 'WAVE', text: 'Wavelengths in vacuum' },
  { type: 'AWAV', text: 'Wavelengths in air' },
  { type: 'WAVE-LOG', text: 'Logarithmic wavelengths in vacuum' },
  { type: 'AWAV-LOG', text: 'Logarithmic wavelengths in air' },
  { type: 'FREQ', text: 'Frequencies' },
  { type: 'VRAD', text: 'Radio velocities' },
  { type: 'VOPT', text: 'Optical velocities' },
];

function spectralTypeOption(value: unknown): SpectralType | null {
  return SPECTRAL_TYPES.find(({ type }) => type === value)?.type ?? null;
}

// A range typed (the first and the last, inclusive; blank: open)
export type Range = [string, string];

// The edits of the values of an item (blank: none)
export interface ValueEdits {
  below: string;
  above: string;
  sigma: string;
}

export const NO_VALUE_EDITS: ValueEdits = { below: '', above: '', sigma: '' };

// The edits that prep makes (blank: none). Pixels and channels count from
// 0, as the map and the spectrum show them.
export interface Edits {
  x: Range;
  y: Range;
  channels: Range;
  values: Record<string, ValueEdits>;
  regions: string;
  pixels: string;
  ratio: string;
  minify: boolean;
  pad: string;
  // Errors from the noise of each channel (of a cube without a file of
  // errors)
  noise: boolean;
}

export const NO_EDITS: Edits = {
  x: ['', ''],
  y: ['', ''],
  channels: ['', ''],
  values: {},
  regions: '',
  pixels: '',
  ratio: '',
  minify: false,
  pad: '',
  noise: false,
};

export interface Draft {
  items: Record<string, Item>;
  spectral: Spectral;
  edits: Edits;
}

export function newDraft(kind: Kind): Draft {
  return { items: Object.fromEntries(KINDS[kind].first.map((key) => [key, EMPTY_ITEM])), spectral: NO_SPECTRAL, edits: NO_EDITS };
}

// The slots of the items a draft has, in their order
export function presentSlots(kind: Kind, draft: Draft): Slot[] {
  return KINDS[kind].slots.filter((slot) => draft.items[slot.key] !== undefined);
}

// ---------------------------------------------------------------------------
// The arrays of the files of an item, as the server describes them

export interface Arrays {
  // The array of each role (undefined until described)
  found: Partial<Record<Role, ArrayInfo>>;
  // All the arrays of the file of each role
  inFile: Partial<Record<Role, ArrayInfo[]>>;
  // Why the file of a role cannot be described
  errors: Partial<Record<Role, string>>;
  // Whether every description is of the inputs given (the spectral axis
  // as said): one being read again for others is not
  current: boolean;
}

export const NO_ARRAYS: Arrays = { found: {}, inFile: {}, errors: {}, current: true };

// What the server said of a file (see useDescriptions): its arrays, why it
// cannot be described, and whether that is of the inputs given now
export interface Described {
  arrays?: ArrayInfo[];
  error?: string;
  current: boolean;
}

export function arraysOf(item: Item, described: (path: string) => Described, ndim: number): Arrays {
  const arrays: Arrays = { found: {}, inFile: {}, errors: {}, current: true };
  for (const { role } of ROLES) {
    const chosen = item.files[role];
    if (chosen === undefined) continue;
    const { arrays: inFile, error, current } = described(chosen.file);
    if (!current) arrays.current = false;
    if (error) arrays.errors[role] = error;
    if (inFile === undefined) continue;
    arrays.inFile[role] = inFile;
    arrays.found[role] = chosen.array === undefined ? likeliest(role, inFile, arrays.found.data, ndim) : findArray(inFile, chosen.array);
  }
  return arrays;
}

// The shape of an array without its axes of length 1, as GBKFIT reads it
export function kept(array: ArrayInfo): number[] {
  return array.shape.filter((size) => size > 1);
}

export function keptAxes(array: ArrayInfo): string[] {
  return array.axes.filter((_, i) => array.shape[i] > 1);
}

export function sameShape(a: ArrayInfo, b: ArrayInfo): boolean {
  return kept(a).join() === kept(b).join();
}

// The likeliest array of a file for a role: for the data, the first with
// their number of axes; for their errors and mask, the first of their
// shape
function likeliest(role: Role, arrays: ArrayInfo[], data: ArrayInfo | undefined, ndim: number): ArrayInfo | undefined {
  const fits = role === 'data' ? (a: ArrayInfo) => kept(a).length === ndim : (a: ArrayInfo) => data !== undefined && sameShape(a, data);
  return arrays.find(fits) ?? arrays[0];
}

function findArray(arrays: ArrayInfo[], array: string | number): ArrayInfo | undefined {
  return typeof array === 'number' ? arrays.find((a) => a.index === array) : arrays.find((a) => a.name === array);
}

// How a dataset names an array: by its name if no other array of its file
// has it, or else by its index
export function arrayRef(arrays: ArrayInfo[], array: ArrayInfo): string | number {
  return array.name && arrays.filter((a) => a.name === array.name).length === 1 ? array.name : array.index;
}

export function arrayLabel(array: ArrayInfo): string {
  return array.name || `array ${array.index}`;
}

// The names of arrays of errors, and of quality flags (e.g. the DQ of HST,
// JWST and MUSE: 0 where the data are good)
const ERROR_NAMES = '(ERR|ERROR|ERRORS|SIGMA|NOISE|STAT|VAR|IVAR|INVVAR|INV_VAR)';
const FLAG_NAMES = /^(DQ|QUAL|QUALITY|FLAG|FLAGS)$/i;

export function isFlags(array: ArrayInfo): boolean {
  return FLAG_NAMES.test(array.name);
}

function looksLikeErrors(array: ArrayInfo): boolean {
  return new RegExp(`(^|_)${ERROR_NAMES}`, 'i').test(array.name);
}

// The other arrays of the file of the data, of their shape: offered for
// the errors (those named as errors first) and for the mask (never errors
// or quality flags: GBKFIT keeps the pixels where a mask is not 0)
export function siblings(arrays: Arrays, role: 'error' | 'mask'): ArrayInfo[] {
  const data = arrays.found.data;
  if (data === undefined) return [];
  const others = (arrays.inFile.data ?? []).filter((a) => a.index !== data.index && sameShape(a, data));
  if (role === 'mask') return others.filter((a) => !looksLikeErrors(a) && !isFlags(a));
  return [...others.filter(looksLikeErrors), ...others.filter((a) => !looksLikeErrors(a) && !isFlags(a))];
}

// What the file of errors holds: as chosen, or else as its header states
// it (see ArrayInfo.holds: e.g. STAT of MUSE, variances; IVAR of MaNGA,
// inverse variances); null if neither says: the user says, it is never
// guessed (e.g. from the name of the array)
export function errorKindOf(item: Item, arrays: Arrays): ErrorKind | null {
  return item.errorKind ?? arrays.found.error?.holds ?? null;
}

// ---------------------------------------------------------------------------
// The spectral axis

export type Spectrum = 'wave' | 'freq';

export const LINE_UNITS: Record<Spectrum, string[]> = { wave: ['Angstrom', 'nm', 'micron'], freq: ['GHz', 'MHz'] };

// Whether the channels of an array are of wavelengths or frequencies,
// which a rest makes velocities (null: velocities, no spectral axis, or one
// whose type FITS does not define and that is not said: then nothing is
// known of it); by the type of its header, or else the type given
export function spectrumOf(array: ArrayInfo | undefined, type: SpectralType | null): Spectrum | null {
  const axis = array && (array.nonstandard_type ? type : keptAxes(array)[2])?.slice(0, 4);
  return axis === 'WAVE' || axis === 'AWAV' ? 'wave' : axis === 'FREQ' ? 'freq' : null;
}

// The rest of the spectral axis as GBKFIT takes it (e.g. '6628.43
// Angstrom'): the line at its redshift, if a line is typed
export function restOf(spectral: Spectral): string | null {
  const line = numberOf(spectral.line);
  const z = numberOf(spectral.redshift) ?? 0;
  if (line === null || !(line > 0) || !Number.isFinite(z) || !spectral.unit) return null;
  const isFrequency = LINE_UNITS.freq.includes(spectral.unit);
  const rest = isFrequency ? line / (1 + z) : line * (1 + z);
  return `${Number(rest.toPrecision(10))} ${spectral.unit}`;
}

// The medium of the wavelengths of a spectral axis, by its type (FITS:
// AWAV in air, WAVE in vacuum; as GBKFIT reads them); null for others
export function mediumOf(axis: string): 'air' | 'vacuum' | null {
  return axis.startsWith('AWAV') ? 'air' : axis.startsWith('WAVE') ? 'vacuum' : null;
}

// The size of each unit of a line (see LINE_UNITS), in m or Hz
const UNIT_SIZES: Record<string, number> = { Angstrom: 1e-10, nm: 1e-9, micron: 1e-6, GHz: 1e9, MHz: 1e6 };

// The lines of the catalogue (see the server's /lines) that a spectral
// axis of wavelengths in a medium, or of frequencies, takes, each with
// its rest in a unit of the axis (as the rest of a line is typed: 8
// digits, within 0.015 km/s)
export function catalogueRests(catalogue: CatalogueLine[], spectrum: Spectrum, medium: 'air' | 'vacuum' | null, unit: string): { name: string; rest: string }[] {
  const size = UNIT_SIZES[unit];
  if (size === undefined || !LINE_UNITS[spectrum].includes(unit)) return [];
  return catalogue.flatMap((line) => {
    const value = spectrum === 'freq' ? line.frequency : medium === 'air' ? line.air : line.wavelength;
    return value === null ? [] : [{ name: line.name, rest: String(Number((value / size).toPrecision(8))) }];
  });
}

// The line of the catalogue whose rest a typed rest is, if any
export function catalogueLineOf(rests: { name: string; rest: string }[], line: string): string | null {
  const value = numberOf(line);
  return rests.find(({ rest }) => Number(rest) === value)?.name ?? null;
}

// The spectral axis of a rest as GBKFIT writes it ('6628.43 Angstrom')
function spectralOfRest(rest: unknown, convention: unknown, type: unknown): Spectral {
  const [line = '', unit = ''] = typeof rest === 'string' ? rest.trim().split(/\s+/, 2) : [];
  return { type: spectralTypeOption(type), convention: convention === 'radio' || convention === 'optical' ? convention : null, line, unit, redshift: '' };
}

// ---------------------------------------------------------------------------
// Numbers typed

// A number typed: null if blank, NaN if not a number
function numberOf(text: string): number | null {
  return text.trim() === '' ? null : Number(text);
}

function integerOf(text: string): number | null {
  const number = numberOf(text);
  return number === null ? null : Math.round(number);
}

// A range typed, from 0 and inclusive, within an axis of a size (null: the
// whole axis)
export function rangeOf([first, last]: Range, size: number): [number, number] | null {
  if (first.trim() === '' && last.trim() === '') return null;
  return [numberOf(first) ?? 0, numberOf(last) ?? size - 1];
}

// A range typed made whole numbers within the axis, the first before the
// last (a blank end stays blank)
export function clampRange([first, last]: Range, size: number): Range {
  const clamp = (text: string) => {
    const number = numberOf(text);
    return number === null || Number.isNaN(number) ? null : Math.min(size - 1, Math.max(0, Math.round(number)));
  };
  const [a, b] = [clamp(first), clamp(last)];
  const [low, high] = a !== null && b !== null && a > b ? [b, a] : [a, b];
  return [low === null ? first : String(low), high === null ? last : String(high)];
}

// The region of the crop (x0, x1, y0, y1, inclusive; null: none)
export function cropRegion(edits: Edits, [nx, ny]: number[]): [number, number, number, number] | null {
  const x = rangeOf(edits.x, nx);
  const y = rangeOf(edits.y, ny);
  return x || y ? [...(x ?? [0, nx - 1]), ...(y ?? [0, ny - 1])] : null;
}

// The values kept of an item (null: open at that end)
export function keptValues(edits: Edits, key: string): [number | null, number | null] {
  const own = edits.values[key] ?? NO_VALUE_EDITS;
  const valid = (text: string) => {
    const number = numberOf(text);
    return number === null || Number.isNaN(number) ? null : number;
  };
  return [valid(own.below), valid(own.above)];
}

// Whether any edit is typed or checked (of the items given)
export function hasEdits(edits: Edits, keys: string[]): boolean {
  const typed = [
    ...edits.x,
    ...edits.y,
    ...edits.channels,
    edits.regions,
    edits.pixels,
    edits.ratio,
    edits.pad,
    ...keys.flatMap((key) => Object.values(edits.values[key] ?? NO_VALUE_EDITS)),
  ];
  return typed.some((text) => text.trim() !== '') || edits.minify || edits.noise;
}

// ---------------------------------------------------------------------------
// What a draft needs

// Where the user gives what is needed
export type Place = 'name' | 'files' | 'spectral' | 'prepare';

export interface Need {
  place: Place;
  text: string;
}

// A draft as the server describes its files: its kind, the draft, and
// the arrays of each item (by key)
export interface Resolved {
  kind: Kind;
  draft: Draft;
  arrays: Record<string, Arrays>;
}

// What the draft still needs before it can be a dataset, the most basic
// first (none: it can be one): what the user is to give, then the problems
// of the headers (see headerProblems), then of the edits. The name is the
// caller's.
export function needsOf(resolved: Resolved): Need[] {
  const needs = inputNeeds(resolved);
  for (const problems of Object.values(headerProblems(resolved, needs))) if (problems[0]) needs.push({ place: 'files', text: problems[0] });
  const first = firstData(resolved);
  if (first) needs.push(...editNeeds(resolved.kind, resolved.draft, resolved.arrays, first));
  return needs;
}

// The problems of the header of the data of each item that a dataset of
// them has (by its key): once nothing the user is to give is missing (until
// then, they are those of what is missing, e.g. the line of an axis of
// wavelengths, which is asked for) and every description is of what is
// given (one read again for a line just typed is of the line before)
export function headerProblems(resolved: Resolved, needs: Need[] = inputNeeds(resolved)): Record<string, string[]> {
  const { kind, draft, arrays } = resolved;
  const slots = presentSlots(kind, draft);
  if (needs.length > 0 || slots.some((slot) => !(arrays[slot.key] ?? NO_ARRAYS).current)) return {};
  return Object.fromEntries(slots.map((slot) => [slot.key, arrays[slot.key]?.found.data?.problems ?? []]));
}

// (the data of the first item with data of the number of axes of the kind)
function firstData({ kind, draft, arrays }: Resolved): ArrayInfo | undefined {
  return presentSlots(kind, draft)
    .map((slot) => arrays[slot.key]?.found.data)
    .find((data) => data !== undefined && kept(data).length === KINDS[kind].ndim);
}

// What the user is to give before the draft can be a dataset
function inputNeeds({ kind, draft, arrays }: Resolved): Need[] {
  const info = KINDS[kind];
  const slots = presentSlots(kind, draft);
  const needs: Need[] = [];
  if (slots.length === 0) needs.push({ place: 'files', text: 'Add a map' });
  for (const slot of slots) {
    const of = arrays[slot.key] ?? NO_ARRAYS;
    const item = draft.items[slot.key];
    const what = kind === 'moments' ? `the map of ${slot.title.toLowerCase()}` : info.what;
    const data = of.found.data;
    const file = item.files.data?.file.split('/').pop();
    if (item.files.data === undefined) needs.push({ place: 'files', text: `Choose ${what}` });
    else if (of.errors.data) needs.push({ place: 'files', text: `${file} cannot be read` });
    else if (data === undefined) needs.push({ place: 'files', text: `Reading the header of ${file}…` });
    else if (kept(data).length !== info.ndim) needs.push({ place: 'files', text: `${file} has ${axes(kept(data).length)}, not ${info.ndim}` });
    else {
      for (const role of ['error', 'mask'] as const) {
        const array = of.found[role];
        const title = role === 'error' ? 'errors' : 'mask';
        if (of.errors[role]) needs.push({ place: 'files', text: `The ${title} of ${what} cannot be read` });
        else if (array !== undefined && !sameShape(array, data)) needs.push({ place: 'files', text: `The ${title} of ${what} are not of its shape` });
        else if (role === 'error' && array !== undefined && item.errorValue === null && errorKindOf(item, of) === null)
          needs.push({ place: 'files', text: `Say what the errors of ${what} are (σ, σ² or 1/σ²): ${array.name || 'their header'} does not say` });
      }
      if (item.errorValue !== null && !(Number(item.errorValue) > 0)) needs.push({ place: 'files', text: 'Give the error a value above 0' });
    }
  }
  const first = firstData({ kind, draft, arrays });
  if (kind === 'moments' && first) {
    const misfit = slots.find((slot) => {
      const data = arrays[slot.key]?.found.data;
      return data !== undefined && kept(data).length === 2 && !sameShape(data, first);
    });
    if (misfit) needs.push({ place: 'files', text: `The map of ${misfit.title.toLowerCase()} is not of the shape of the others` });
  }
  if (kind === 'cube' && first) {
    if (first.nonstandard_type && draft.spectral.type === null) {
      needs.push({ place: 'spectral', text: `Say what the spectral axis is (${first.nonstandard_type} is not a FITS type)` });
    }
    if (first.ambiguous_velocity && draft.spectral.convention === null) {
      needs.push({ place: 'spectral', text: 'Say whether the velocities are radio or optical' });
    }
    if (spectrumOf(first, draft.spectral.type) !== null && restOf(draft.spectral) === null) {
      needs.push({ place: 'spectral', text: 'Give the line, for the velocities of the channels' });
    }
  }
  return needs;
}

function axes(count: number): string {
  return `${count} ${count === 1 ? 'axis' : 'axes'}`;
}

function editNeeds(kind: Kind, draft: Draft, arrays: Record<string, Arrays>, data: ArrayInfo): Need[] {
  const { edits } = draft;
  const keys = presentSlots(kind, draft).map((slot) => slot.key);
  const typed = [
    ...edits.x,
    ...edits.y,
    ...edits.channels,
    edits.regions,
    edits.pixels,
    edits.ratio,
    edits.pad,
    ...keys.flatMap((key) => Object.values(edits.values[key] ?? NO_VALUE_EDITS)),
  ];
  if (typed.some((text) => Number.isNaN(numberOf(text)))) return [{ place: 'prepare', text: 'An edit is not a number' }];
  const shape = kept(data);
  const ranges = [
    [edits.x, shape[0], 'x'],
    [edits.y, shape[1], 'y'],
    ...(kind === 'cube' ? [[edits.channels, shape[2], 'channels'] as const] : []),
  ] as const;
  for (const [range, size, name] of ranges) {
    const crop = rangeOf(range, size);
    if (crop && !(0 <= crop[0] && crop[0] <= crop[1] && crop[1] < size)) return [{ place: 'prepare', text: `The crop of ${name} is out of the data` }];
  }
  if (kind === 'moments' && hasEdits(edits, keys)) {
    const errors = keys.filter((key) => draft.items[key].files.error !== undefined && draft.items[key].errorValue === null);
    const kinds = new Set(errors.map((key) => errorKindOf(draft.items[key], arrays[key] ?? NO_ARRAYS)).filter((k) => k !== null));
    if (kinds.size > 1) return [{ place: 'prepare', text: 'The errors of the maps are of different kinds (σ, σ², 1/σ²); prep needs one' }];
  }
  return [];
}

// ---------------------------------------------------------------------------
// The options of a dataset, and of prep

// A file of an item as the options of a dataset name it ({file, hdu},
// relative to the project), or as prep takes it (relative to the
// workspace: its path, or its path and its array if its file has others)
function fileOption(item: Item, arrays: Arrays, role: Role, project: string): Record<string, unknown> | undefined {
  const chosen = item.files[role];
  const array = arrays.found[role];
  if (chosen === undefined || array === undefined) return undefined;
  return { file: inProject(chosen.file, project), hdu: arrayRef(arrays.inFile[role] ?? [], array) };
}

type PrepFile = string | { file: string; hdu: string | number };

function prepFile(item: Item, arrays: Arrays, role: Role): PrepFile | null {
  const chosen = item.files[role];
  const array = arrays.found[role];
  if (chosen === undefined || array === undefined) return null;
  if (role === 'error' && item.errorValue !== null) return null;
  const inFile = arrays.inFile[role] ?? [];
  return inFile.length === 1 ? chosen.file : { file: chosen.file, hdu: arrayRef(inFile, array) };
}

// A path of the workspace relative to a project, and back
export function inProject(path: string, project: string): string {
  const prefix = project === '.' ? '' : `${project}/`;
  return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}

function inWorkspace(path: string, project: string): string {
  return project === '.' ? path : `${project}/${path}`;
}

// The options of an item in a dataset (its files, its error value, and
// what its file of errors holds unless standard deviations)
function itemOptions(item: Item, arrays: Arrays, project: string): Record<string, unknown> {
  const options: Record<string, unknown> = {};
  for (const { role } of ROLES) {
    if (role === 'error' && item.errorValue !== null) options.error = Number(item.errorValue);
    else {
      const file = fileOption(item, arrays, role, project);
      if (file !== undefined) options[role] = file;
    }
  }
  const errorKind = errorKindOf(item, arrays);
  if (item.errorValue === null && options.error !== undefined && errorKind !== null && errorKind !== 'sigma') options.error_kind = errorKind;
  return options;
}

// The options of the dataset of a draft as its files are (no prep)
export function datasetOptions({ kind, draft, arrays }: Resolved, project: string): Record<string, unknown> {
  const options: Record<string, unknown> = { type: KINDS[kind].type };
  if (kind === 'moments') {
    for (const slot of presentSlots(kind, draft)) options[slot.key] = itemOptions(draft.items[slot.key], arrays[slot.key] ?? NO_ARRAYS, project);
    return options;
  }
  const key = KINDS[kind].first[0];
  Object.assign(options, itemOptions(draft.items[key], arrays[key] ?? NO_ARRAYS, project));
  if (kind === 'cube') {
    const spectral: Record<string, unknown> = {};
    const data = arrays[key]?.found.data;
    if (data?.nonstandard_type && draft.spectral.type !== null) spectral.type = draft.spectral.type;
    if (draft.spectral.convention !== null) spectral.convention = draft.spectral.convention;
    const rest = spectrumOf(data, draft.spectral.type) !== null ? restOf(draft.spectral) : null;
    if (rest !== null) spectral.rest = rest;
    if (Object.keys(spectral).length > 0) options.spectral = spectral;
  }
  return options;
}

// Whether the draft needs prep to be a dataset: only its edits do (a cube
// of wavelengths becomes velocities by the rest of its dataset)
export function needsPrep({ kind, draft }: Resolved): boolean {
  return hasEdits(draft.edits, presentSlots(kind, draft).map((slot) => slot.key));
}

export interface PrepRequest {
  task: PrepTask;
  options: Record<string, unknown>;
}

// The job of prep of a draft (its edits), writing into the folder data of
// the project. The draft has no needs (see needsOf).
export function prepRequest({ kind, draft, arrays }: Resolved, project: string): PrepRequest {
  const { edits } = draft;
  const slots = presentSlots(kind, draft);
  const first = arrays[slots[0].key]?.found.data;
  const shape = first ? kept(first) : [0, 0, 0];
  const region = cropRegion(edits, shape);
  const roi = region && [region[0], region[1] + 1, region[2], region[3] + 1];
  const common = {
    roi_spat: roi,
    ccl_lcount: integerOf(edits.regions),
    ccl_pcount: integerOf(edits.pixels),
    ccl_lratio: numberOf(edits.ratio),
    sclip_iters: SIGMA_ITERATIONS,
    minify: edits.minify,
    nanpad: integerOf(edits.pad),
    dtype: 'float32',
    output_dir: inWorkspace('data', project),
  };
  const files = (key: string) => {
    const item = draft.items[key];
    const of = arrays[key] ?? NO_ARRAYS;
    return { file_d: prepFile(item, of, 'data'), file_e: prepFile(item, of, 'error'), file_m: prepFile(item, of, 'mask'), kind: errorKindOf(item, of) ?? undefined };
  };
  const values = (key: string) => {
    const [below, above] = keptValues(edits, key);
    return { clip_min: below, clip_max: above, sclip_sigma: numberOf(edits.values[key]?.sigma ?? '') };
  };
  if (kind === 'moments') {
    const maps = slots.map((slot) => ({ ...files(slot.key), ...values(slot.key) }));
    const list = <T,>(of: (map: (typeof maps)[number]) => T | null) => {
      const items = maps.map(of);
      return items.some((item) => item !== null) ? items : null;
    };
    return {
      task: KINDS.moments.task,
      options: {
        ...common,
        file_d: maps.map((map) => map.file_d),
        file_e: list((map) => map.file_e),
        file_m: list((map) => map.file_m),
        error_kind: maps.find((map) => map.file_e !== null)?.kind ?? 'sigma',
        clip_min: list((map) => map.clip_min),
        clip_max: list((map) => map.clip_max),
        sclip_sigma: list((map) => map.sclip_sigma),
      },
    };
  }
  const key = slots[0].key;
  const own = files(key);
  const base = { ...common, ...values(key), file_d: own.file_d, file_m: own.file_m };
  if (kind === 'image') {
    return { task: KINDS.image.task, options: { ...base, file_e: own.file_e, error_kind: own.file_e === null ? 'sigma' : own.kind } };
  }
  const channels = rangeOf(edits.channels, shape[2]);
  const noise = edits.noise && own.file_e === null;
  return {
    task: KINDS.cube.task,
    options: {
      ...base,
      file_e: own.file_e,
      error_kind: own.file_e === null ? 'sigma' : own.kind,
      roi_spec: channels && [channels[0], channels[1] + 1],
      velocity_rest: spectrumOf(first, draft.spectral.type) !== null ? restOf(draft.spectral) : null,
      velocity: draft.spectral.convention,
      spectral_type: first?.nonstandard_type ? draft.spectral.type : null,
      noise_error: noise,
    },
  };
}

// The sigma clipping of prep iterates this many times (as the command
// line does)
export const SIGMA_ITERATIONS = 5;

// The files that a job of prep wrote, by the options of their files (of
// moment maps, a list of each, one for each map)
export type Prepared = Record<string, string | (string | null)[]>;

// The options of a dataset of the files that prep wrote: their errors are
// standard deviations (or the one value of the item stays); a prepared
// cube has velocities in its header
export function preparedOptions({ kind, draft }: Resolved, prepared: Prepared, project: string): Record<string, unknown> {
  const options: Record<string, unknown> = { type: KINDS[kind].type };
  const slots = presentSlots(kind, draft);
  slots.forEach((slot, i) => {
    const of = (key: string) => {
      const value = prepared[key];
      return (Array.isArray(value) ? value[i] : value) ?? null;
    };
    const item: Record<string, unknown> = { data: inProject(of('file_d')!, project) };
    const error = of('file_e');
    if (error !== null) item.error = inProject(error, project);
    else if (draft.items[slot.key].errorValue !== null) item.error = Number(draft.items[slot.key].errorValue);
    const mask = of('file_m');
    if (mask !== null) item.mask = inProject(mask, project);
    if (kind === 'moments') options[slot.key] = item;
    else Object.assign(options, item);
  });
  return options;
}

// ---------------------------------------------------------------------------
// From a dataset of the project, and from the record of its preparation

// A file option of a dataset or of prep: a path, or {file, hdu}
function fileArrayOf(value: unknown, base: (path: string) => string): FileArray | undefined {
  if (typeof value === 'string') return { file: base(value), array: 0 };
  if (value !== null && typeof value === 'object') {
    const { file, hdu = 0 } = value as { file?: unknown; hdu?: unknown };
    if (typeof file === 'string' && (typeof hdu === 'string' || typeof hdu === 'number')) return { file: base(file), array: hdu };
  }
  return undefined;
}

function errorKindOption(value: unknown): ErrorKind | null {
  return ERROR_KINDS.find(({ kind }) => kind === value)?.kind ?? null;
}

// The item of the options of an item of a dataset (paths relative to the
// project)
function itemOfOptions(options: Record<string, unknown>, project: string): Item {
  const base = (path: string) => inWorkspace(path, project);
  const files: Item['files'] = {};
  for (const { role } of ROLES) {
    const file = fileArrayOf(options[role], base);
    if (file !== undefined) files[role] = file;
  }
  return {
    files,
    errorValue: typeof options.error === 'number' ? String(options.error) : null,
    errorKind: errorKindOption(options.error_kind) ?? 'sigma',
  };
}

// The draft of a dataset of the project as it is (its files, no edits)
export function draftOfDataset(kind: Kind, options: Record<string, unknown>, project: string): Draft {
  const items: Record<string, Item> = {};
  if (kind === 'moments') {
    for (const slot of KINDS.moments.slots) {
      const item = options[slot.key];
      if (item !== null && typeof item === 'object') items[slot.key] = itemOfOptions(item as Record<string, unknown>, project);
    }
  } else items[KINDS[kind].first[0]] = itemOfOptions(options, project);
  const spectral = (options.spectral ?? {}) as { rest?: unknown; convention?: unknown; type?: unknown };
  return { items, spectral: spectralOfRest(spectral.rest, spectral.convention, spectral.type), edits: NO_EDITS };
}

// How the files of a dataset were prepared (see the server's
// PreparationInfo): paths relative to the workspace
export interface Preparation {
  task: string;
  options: Record<string, unknown>;
  files: Prepared;
  time: string;
}

// The prepared data file of each item of a dataset (paths of the
// workspace), by key
export function dataFiles(kind: Kind, options: Record<string, unknown>, project: string): Record<string, string> {
  const draft = draftOfDataset(kind, options, project);
  return Object.fromEntries(
    Object.entries(draft.items).flatMap(([key, item]) => (item.files.data ? [[key, item.files.data.file]] : [])),
  );
}

// The draft of a dataset of prepared files, from the record of their
// preparation: the original files, and the edits that made them. The
// dataset gives the items (of moment maps, which map each file is).
export function draftOfPreparation(kind: Kind, dataset: Record<string, unknown>, record: Preparation, project: string): Draft {
  const { options } = record;
  const files = dataFiles(kind, dataset, project);
  const prepared = record.files.file_d;
  // (the index of each item in the lists of prep)
  const index = (key: string) => (Array.isArray(prepared) ? prepared.indexOf(files[key]) : 0);
  const at = (value: unknown, i: number) => (Array.isArray(value) && kind === 'moments' ? value[i] : value);
  const base = (path: string) => path;
  const saved = draftOfDataset(kind, dataset, project);
  const items: Record<string, Item> = {};
  const values: Record<string, ValueEdits> = {};
  for (const key of Object.keys(files)) {
    const i = index(key);
    if (i < 0) continue;
    const item: Item = { files: {}, errorValue: null, errorKind: errorKindOption(options.error_kind) };
    for (const [role, option] of [['data', 'file_d'], ['error', 'file_e'], ['mask', 'file_m']] as const) {
      const file = fileArrayOf(at(options[option], i), base);
      if (file !== undefined) item.files[role] = file;
    }
    const error = saved.items[key]?.errorValue ?? null;
    if (item.files.error === undefined && error !== null) item.errorValue = error;
    items[key] = item;
    const text = (value: unknown) => (typeof value === 'number' ? String(value) : '');
    values[key] = { below: text(at(options.clip_min, i)), above: text(at(options.clip_max, i)), sigma: text(at(options.sclip_sigma, i)) };
  }
  const range = (value: unknown, from: number): Range =>
    Array.isArray(value) && typeof value[from] === 'number' && typeof value[from + 1] === 'number'
      ? [String(value[from]), String(value[from + 1] - 1)]
      : ['', ''];
  const text = (value: unknown) => (typeof value === 'number' ? String(value) : '');
  return {
    items,
    spectral: spectralOfRest(options.velocity_rest, options.velocity, options.spectral_type),
    edits: {
      x: range(options.roi_spat, 0),
      y: range(options.roi_spat, 2),
      channels: range(options.roi_spec, 0),
      values,
      regions: text(options.ccl_lcount),
      pixels: text(options.ccl_pcount),
      ratio: text(options.ccl_lratio),
      minify: options.minify === true,
      pad: text(options.nanpad),
      noise: options.noise_error === true,
    },
  };
}

// The edits of a draft in words, one line each (for the record of how a
// dataset was made)
export function editsInWords(kind: Kind, draft: Draft): string[] {
  const { edits } = draft;
  const lines: string[] = [];
  const range = (r: Range) => `${r[0] || '0'}–${r[1] || 'end'}`;
  const crops = [
    edits.x.some(Boolean) && `x ${range(edits.x)}`,
    edits.y.some(Boolean) && `y ${range(edits.y)}`,
    edits.channels.some(Boolean) && `channels ${range(edits.channels)}`,
  ].filter(Boolean);
  if (crops.length) lines.push(`Cropped to ${crops.join(', ')}`);
  if (edits.noise) lines.push('Errors from the noise of each channel');
  for (const slot of presentSlots(kind, draft)) {
    const own = edits.values[slot.key] ?? NO_VALUE_EDITS;
    const of = kind === 'moments' ? ` of ${slot.title.toLowerCase()}` : '';
    if (own.below || own.above) lines.push(`Kept the values${of} from ${own.below || '−∞'} to ${own.above || '∞'}`);
    if (own.sigma) lines.push(`Left out the values${of} beyond ${own.sigma}σ`);
  }
  const regions = [
    edits.regions && `the ${edits.regions} largest`,
    edits.pixels && `of ${edits.pixels} pixels or more`,
    edits.ratio && `of ${edits.ratio}× the largest or more`,
  ].filter(Boolean);
  if (regions.length) lines.push(`Kept the regions ${regions.join(', ')}`);
  if (edits.minify) lines.push('Cropped to the pixels left');
  if (edits.pad) lines.push(`Padded by ${edits.pad} pixels`);
  const rest = restOf(draft.spectral);
  if (rest) lines.push(`Velocities of the line at ${rest}`);
  if (draft.spectral.convention) lines.push(`Velocities read as ${draft.spectral.convention}`);
  const said = SPECTRAL_TYPES.find(({ type }) => type === draft.spectral.type);
  if (said) lines.push(`Spectral axis read as ${said.text.toLowerCase()} (${said.type})`);
  return lines;
}

// The draft of the files that prep wrote for a draft (of its items, in
// their order): their headers state velocities, and they have no edits
export function draftOfPrepared(kind: Kind, draft: Draft, prepared: Prepared): Draft {
  const items: Record<string, Item> = {};
  presentSlots(kind, draft).forEach((slot, i) => {
    const of = (key: string) => {
      const value = prepared[key];
      return (Array.isArray(value) ? value[i] : value) ?? null;
    };
    const files: Item['files'] = {};
    for (const [role, key] of [['data', 'file_d'], ['error', 'file_e'], ['mask', 'file_m']] as const) {
      const file = of(key);
      if (file !== null) files[role] = { file, array: 0 };
    }
    const own = draft.items[slot.key];
    items[slot.key] = { files, errorValue: files.error ? null : own.errorValue, errorKind: 'sigma' };
  });
  return { items, spectral: NO_SPECTRAL, edits: NO_EDITS };
}

// A name for a dataset of a file (e.g. NGC2403 of NGC2403.cube.fits), not
// one of those taken
export function suggestName(file: string, taken: string[]): string {
  const stem = (file.split('/').pop() ?? '')
    .replace(/\.(fits|fit|fts)(\.gz)?$/i, '')
    .replace(/^prep_/, '')
    .replace(/[^A-Za-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  const base = stem || 'dataset';
  if (!taken.includes(base)) return base;
  for (let n = 2; ; n++) if (!taken.includes(`${base}_${n}`)) return `${base}_${n}`;
}
