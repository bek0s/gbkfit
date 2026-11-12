// The model of an observation of the Studio, without React: its options
// (those of the library: see gbkfit.observation), what each stage of it
// is at a glance, and the checks of it against its data.
import type { ArrayInfo } from '../../api/client';
import { membersOf, propertiesOf, typeOf, type Defs, type Options, type Schema } from '../../components/SchemaForm';

// The stages of an observation on the sky, from the galaxies to the data,
// as the dock has them (by the id of their section)
export const STAGES = [
  { key: 'light', title: 'Galaxies', section: 'ob-light' },
  { key: 'line_of_sight', title: 'Line of sight', section: 'ob-sight' },
  { key: 'instrument', title: 'Instrument', section: 'ob-instrument' },
  { key: 'observable', title: 'Observable', section: 'ob-observable' },
  { key: 'data', title: 'Data', section: 'ob-data' },
] as const;

// The options of a sky observation that its sections set by hand; the
// others are in its forms of more options
export const LIGHT_KEYS = ['galaxies', 'components', 'lines', 'band'];
export const HANDLED_KEYS = ['type', 'data', 'observable', 'instrument', 'line_of_sight', ...LIGHT_KEYS];

// The observables (and datasets) whose data have a spectral axis
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

// What a stage of an observation is, at a glance
export function glanceOf(key: (typeof STAGES)[number]['key'], options: Options, titleOf: (family: string, type: unknown) => string): string {
  if (key === 'light') {
    const galaxies = options.galaxies;
    return Array.isArray(galaxies) && galaxies.length ? galaxies.join(', ') : 'All of the setup';
  }
  if (key === 'line_of_sight') return options.line_of_sight == null ? 'Not chosen' : titleOf('line_of_sight', objectAt(options, 'line_of_sight').type).replace(/ \(.*\)$/, '');
  if (key === 'instrument') {
    const instrument = objectAt(options, 'instrument');
    const parts = (['psf', 'lsf'] as const).flatMap((part) => {
      const value = instrument[part];
      if (value == null) return [];
      return [`${part.toUpperCase()} ${typeof value === 'string' ? value : titleOf(part, (value as Options).type)}`];
    });
    return parts.join(' · ') || 'A perfect one';
  }
  if (key === 'observable') return titleOf('observable', objectAt(options, 'observable').type);
  return typeof options.data === 'string' ? options.data : options.data ? 'Given in place' : 'None: a model';
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
  titleOf,
}: {
  options: Options;
  datasets: string[];
  // The type of the dataset named (undefined: not read, or none)
  datasetType: string | undefined;
  data: ArrayInfo | undefined;
  psf: Ellipse | null;
  lsfWidth: number | null;
  titleOf: (family: string, type: unknown) => string;
}): Check[] {
  const checks: Check[] = [];
  const observable = objectAt(options, 'observable');
  const instrument = objectAt(options, 'instrument');
  const named = typeof options.data === 'string' ? options.data : null;
  if (options.data == null) checks.push({ ok: null, text: 'No data: the observation makes a model (or mock data).' });
  else if (named !== null && !datasets.includes(named)) checks.push({ ok: false, text: `The project has no dataset ${named}.` });
  else if (datasetType !== undefined) {
    checks.push(
      datasetType === observable.type
        ? { ok: true, text: `The observable and the data are both ${titleOf('observable', observable.type)}.` }
        : { ok: false, text: `The observable is ${titleOf('observable', observable.type)}, but the data are ${titleOf('observable', datasetType)}.` },
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
  if (isSpectral(observable.type)) {
    if (instrument.lsf == null) checks.push({ ok: null, text: 'No LSF: a perfect spectrograph.' });
    else if (lsfWidth !== null && width !== null) {
      checks.push({ ok: null, text: `The LSF is ${fixed(lsfWidth)} km/s across (FWHM): ${fixed(lsfWidth / width)} channels of the data.` });
    }
  }
  if (data?.step) {
    checks.push({ ok: null, text: `The data have pixels of ${fixed(Math.abs(data.step[0]))}″${width !== null ? ` and channels of ${fixed(width)} km/s` : ''}.` });
  }
  return checks;
}

function fixed(value: number): string {
  return Number(value.toPrecision(3)).toString();
}

// The member of the observable of a type (see the schema): that of an
// observation with data (the data give its grid) or without (it gives its
// own: its size, or its regions, its coverage...)
export function observableMember(defs: Defs, type: unknown, withData: boolean): Schema | undefined {
  const members = membersOf(defs.observable ?? {}, defs).filter((m) => typeOf(m) === type);
  const gives = (m: Schema) => Array.isArray(m.required) && (m.required as string[]).some((key) => key === 'size' || key === 'coverage' || key === 'regions');
  return members.find((m) => gives(m) !== withData) ?? members[0];
}

// The options of the observable that its member for data does not take
// (its own grid, which the data give)
export function gridOptions(defs: Defs, observable: Options): string[] {
  const member = observableMember(defs, observable.type, true);
  if (!member) return [];
  const allowed = Object.keys(propertiesOf(member));
  return Object.keys(observable).filter((key) => !allowed.includes(key));
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
  const observable = objectAt(options, 'observable');
  const withData = options.data != null;
  if (typeof options.data === 'string' && !datasets.includes(options.data)) needs.push({ section: 'ob-data', text: `The project has no dataset ${options.data}` });
  if (observable.type == null) needs.push({ section: 'ob-observable', text: 'Choose what the data measure (the observable)' });
  else {
    if (withData && datasetType !== undefined && datasetType !== observable.type) {
      needs.push({ section: 'ob-observable', text: `The observable is not of the type of the data (${titleOf('observable', datasetType)})` });
    }
    const extra = withData ? gridOptions(defs, observable) : [];
    if (extra.length) needs.push({ section: 'ob-observable', text: `The data give the grid: the observable cannot have its ${extra.join(', ')}` });
    const member = observableMember(defs, observable.type, withData);
    const missing = ((member?.required as string[] | undefined) ?? []).filter((key) => key !== 'type' && observable[key] == null);
    if (missing.length) needs.push({ section: 'ob-observable', text: `Give the observable its ${missing.map((key) => NAMES[key] ?? key).join(', ')}` });
    else if (member && 'spectral' in propertiesOf(member) && isSpectral(observable.type) && !(withData && dataHaveChannels(datasetType))) {
      const spectral = objectAt(observable, 'spectral');
      if (spectral.size == null || spectral.step == null) needs.push({ section: 'ob-observable', text: 'Give the channels of the model (their number and width)' });
    }
  }
  if (options.line_of_sight == null) needs.push({ section: 'ob-sight', text: 'Choose the line of sight' });
  return needs;
}
