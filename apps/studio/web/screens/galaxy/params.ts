// The parameters of a galaxy as the library describes them (see the
// server's POST /parameters), and their properties as a params block holds
// them (by key: a parameter's name, or its name and a subscript, e.g.
// 'disk_posa[0]'): a number fixes it (every element of a vector), a list
// gives each element its own, a string ties it to an expression, a dict
// frees it ({value, min, max, ...}; its keys that start with '*' are
// lists, one item per element), and null leaves it to a transform.
//
// The names and sizes of the parameters follow the options of the galaxy
// (e.g. its components' names, the place of a trait in its list, the
// number of rings), so that its properties are kept in step with them as
// they change (see transition).
import { getAt, isObject, startsWith, type Options, type Path } from './options';

export interface Param {
  name: string;
  // Its name and what it is, for people (e.g. 'Rotation speed')
  title?: string | null;
  desc?: string | null;
  // Its number of elements: 1 (a scalar), or more (e.g. one per ring)
  size: number;
  default: number | null;
  unit: string | null;
  // Its valid range, as the library writes it (e.g. '[0, 90]', '(0, inf)')
  range: string;
  period: number | null;
  // The path of the option that makes it in the galaxy (e.g.
  // ['components', 0, 'vptraits', 1]; ['components', 0]:
  // the component itself)
  owner: Path;
}

export type Properties = Record<string, unknown>;

// The name of the parameter of a key ('disk_posa[0:3]': 'disk_posa')
export function keyName(key: string): string {
  const at = key.indexOf('[');
  return (at < 0 ? key : key.slice(0, at)).trim();
}

// --- What a property sets ----------------------------------------------

// How a property sets a parameter, as the screen edits it: fixed or free
// values (one per element, or one for all), tied to an expression, or
// otherwise (e.g. solved, or left to a transform; kept as it is)
export type Setting =
  | { kind: 'fixed'; values: number[] }
  | { kind: 'free'; values: number[]; min: number | null; max: number | null }
  | { kind: 'tied'; expression: string }
  | { kind: 'other' };

// The setting of a parameter by its properties (that of its name; keys
// with subscripts are not edited here: 'other'), its values one per
// element (or one for all)
export function settingOf(properties: Properties, param: Param): Setting {
  const subscripted = Object.keys(properties).some((key) => key !== param.name && keyName(key) === param.name);
  const property = properties[param.name];
  if (subscripted) return { kind: 'other' };
  if (property === undefined) return { kind: 'fixed', values: param.default === null ? [] : [param.default] };
  if (typeof property === 'number') return { kind: 'fixed', values: [property] };
  if (typeof property === 'string') return { kind: 'tied', expression: property };
  if (Array.isArray(property)) {
    if (property.every((v) => typeof v === 'number')) return { kind: 'fixed', values: property as number[] };
    // (free elements, each its own dict, or a mix: free if every one is)
    if (property.every((v) => isObject(v) && typeof v.value === 'number')) {
      const items = property as Options[];
      return { kind: 'free', values: items.map((v) => v.value as number), min: numberOr(items[0].min), max: numberOr(items[0].max) };
    }
    return { kind: 'other' };
  }
  if (isObject(property)) {
    const values = Array.isArray(property['*value']) ? (property['*value'] as unknown[]) : [property.value];
    if (values.every((v) => typeof v === 'number')) return { kind: 'free', values: values as number[], min: numberOr(property.min), max: numberOr(property.max) };
  }
  return { kind: 'other' };
}

function numberOr(value: unknown): number | null {
  return typeof value === 'number' ? value : null;
}

// The value of element i of a setting's values (one for all, or one each)
export function valueAt(values: number[], i: number): number | undefined {
  return values.length === 1 ? values[0] : values[i];
}

// The values of every element of a parameter (its size), of a setting's
// values: one for all spread, or its own (resampled if of another size)
export function elementValues(values: number[], size: number): number[] {
  if (values.length === 0) return [];
  if (values.length === 1 || values.length === size) return values.length === size ? values : Array(size).fill(values[0]);
  return resample(values, size);
}

// The property of a parameter with its values (one for all: a number;
// else one each), its other properties kept; of a free one, its range
// grown to hold them (a value is never refused for it)
export function withValues(property: unknown, param: Param, values: number[]): unknown {
  const one = values.every((v) => v === values[0]);
  const scalar = param.size === 1 || one;
  if (isObject(property)) {
    const { value: _, '*value': __, ...rest } = property;
    const [low, high] = [Math.min(...values), Math.max(...values)];
    if (typeof rest.min === 'number' && low < rest.min) rest.min = low;
    if (typeof rest.max === 'number' && high > rest.max) rest.max = high;
    return scalar ? { ...rest, value: values[0] } : { ...rest, '*value': values };
  }
  if (Array.isArray(property) && property.every(isObject)) {
    // (free elements of their own: each given its value)
    return elementValues(values, param.size).map((v, i) => ({ ...(property[i] as Options | undefined ?? property[0]), value: v }));
  }
  return scalar ? values[0] : values;
}

// The property of a parameter freed (its value kept, its range the given
// one), or fixed (at its value)
export function withFreedom(property: unknown, param: Param, setting: Setting, free: boolean, range: [number, number]): unknown {
  if (setting.kind !== 'fixed' && setting.kind !== 'free') return property;
  const values = setting.values.length ? setting.values : [param.default ?? 0];
  const scalar = values.length === 1;
  if (!free) return scalar ? values[0] : values;
  const [min, max] = range;
  return scalar ? { value: values[0], min, max } : { '*value': values, min, max };
}

// The property of a free parameter with its range set
export function withRange(property: unknown, min: number, max: number): unknown {
  return isObject(property) ? { ...property, min, max } : property;
}

// --- Ranges ----------------------------------------------------------------

// The valid range of a parameter, of its text (e.g. '[0, 90]', '(0, inf)')
export function validRange(param: Param): [number, number] {
  const match = /^[[(]\s*([^,]+),\s*([^\])]+)[\])]$/.exec(param.range);
  const read = (text: string | undefined, infinite: number) => {
    const value = Number(text?.trim().replace(/^(-?)inf$/, '$1Infinity'));
    return Number.isFinite(value) ? value : infinite;
  };
  return [read(match?.[1], -Infinity), read(match?.[2], Infinity)];
}

// The reach of the slider of a parameter: its valid range where it has
// one; else a reach of its unit (an angle, a velocity, a length within
// the field of the view), or about its default; never beyond its valid
// range. Its step is round, about 1/200 of it.
export function sliderReach(param: Param, field: number): { min: number; max: number; step: number } {
  const [low, high] = validRange(param);
  const around = Math.abs(param.default ?? 0) || 1;
  const reach: [number, number] =
    param.unit === 'deg'
      ? [low >= 0 ? 0 : -(param.period ?? 360) / 2, param.period ?? 90]
      : param.unit === 'km / s'
        ? param.name.endsWith('vsys')
          ? [-500, 500]
          : [low >= 0 ? 0 : -400, 400]
        : param.unit === 'arcsec'
          ? [low >= 0 ? 0 : -field, field]
          : // (a logarithm, e.g. log_m: a few decades about its default)
            /(^|_)log_/.test(param.name)
            ? [(param.default ?? 0) - 3, (param.default ?? 0) + 3]
            : [low >= 0 ? 0 : -4 * around, 4 * around];
  const min = Math.max(low, reach[0]);
  const max = Math.min(high, Number.isFinite(high) && !Number.isFinite(reach[1]) ? high : Math.max(reach[1], min + 1e-9));
  const span = max - min || 1;
  const step = 10 ** Math.floor(Math.log10(span / 200));
  return { min, max, step };
}

// A default range of a parameter freed for a fit, about its value: its
// slider's reach
export function freeRange(param: Param, field: number): [number, number] {
  const { min, max } = sliderReach(param, field);
  return [min, max];
}

// --- Names -------------------------------------------------------------

// The name of a parameter in the part that owns it: without the prefixes
// of its component, its trait (or the name of a trait, of a line), or its
// mass component: e.g. 'vt' of 'disk_vpt2_vt', 'log_m' of 'mass_halo_log_m'
export function shortName(param: Param, options: Options, siblings: Param[]): string {
  const prefix = commonPrefix(siblings.filter((p) => ownerKey(p.owner) === ownerKey(param.owner)).map((p) => p.name));
  const [section, index] = param.owner;
  let rest = param.name;
  if (section === 'components' && typeof index === 'number') {
    const name = componentNameAt(options, index);
    if (name !== null && rest.startsWith(`${name}_`)) rest = rest.slice(name.length + 1);
    // (of a line, its ratio: 'nii6583 ratio')
    if (param.owner[2] === 'emission') return rest.replace(/_ratio$/, ' ratio');
    // (of a trait, its prefix: of its kind and place, or its kind and
    // name; that of all the parameters of the trait, up to an underscore,
    // if it has several, else its kind, its place and its name)
    if (param.owner.length > 3) {
      const own = prefix.length && prefix.endsWith('_') ? prefix : '';
      if (own && param.name.startsWith(own) && param.name.length > own.length) return param.name.slice(own.length);
      const trait = getAt(options, param.owner);
      const traitName = isObject(trait) && typeof trait.name === 'string' ? trait.name : null;
      rest = rest.replace(/^[a-z]+\d*_/, '');
      if (traitName !== null && rest.startsWith(`${traitName}_`)) rest = rest.slice(traitName.length + 1);
      return rest;
    }
  } else if (section === 'mass' && typeof index === 'number') {
    const owner = options.mass;
    const item = Array.isArray(owner) ? owner[index] : owner;
    const name = isObject(item) ? item.name : undefined;
    if (typeof name === 'string' && rest.startsWith(`mass_${name}_`)) rest = rest.slice(`mass_${name}_`.length);
  } else if (section === 'gravity') rest = rest.replace(/^gravity_/, '');
  return rest;
}

// The longest prefix of names ending at an underscore (several names: their
// common one; one: none)
function commonPrefix(names: string[]): string {
  if (names.length < 2) return '';
  let prefix = names[0];
  for (const name of names) {
    while (!name.startsWith(prefix)) prefix = prefix.slice(0, -1);
  }
  return prefix.slice(0, prefix.lastIndexOf('_') + 1);
}

function componentNameAt(options: Options, index: number): string | null {
  const components = options.components;
  const component = Array.isArray(components) ? components[index] : index === 0 ? components : undefined;
  return isObject(component) && typeof component.name === 'string' ? component.name : null;
}

export function ownerKey(owner: Path): string {
  return owner.join('/');
}

// The parameters an option makes: those it owns, or (exact) only those of
// the option itself (e.g. the position of a component, not its traits)
export function ownedBy(params: Param[], option: Path, exact = false): Param[] {
  return params.filter((p) => (exact ? ownerKey(p.owner) === ownerKey(option) : startsWith(p.owner, option)));
}

// --- Keeping the properties in step with the galaxy ------------------------

// The values of a list resampled to another size: linearly by the places
// of their elements (their radii, if given for both; else evenly), beyond
// the ends the end values; numbers only (others: the nearest)
export function resample(values: unknown[], size: number, from?: number[], to?: number[]): number[] {
  const n = values.length;
  if (n === 0 || size <= 0) return [];
  const x0 = from && from.length === n ? from : values.map((_, i) => (n === 1 ? 0 : i / (n - 1)));
  const x1 = to && to.length === size && from && from.length === n ? to : Array.from({ length: size }, (_, i) => (size === 1 ? 0 : i / (size - 1)));
  return x1.map((x) => {
    if (x <= x0[0]) return values[0] as number;
    if (x >= x0[n - 1]) return values[n - 1] as number;
    let j = 1;
    while (x0[j] < x) j++;
    const [a, b] = [values[j - 1], values[j]];
    if (typeof a !== 'number' || typeof b !== 'number') return (x - x0[j - 1] < x0[j] - x ? a : b) as number;
    const t = (x - x0[j - 1]) / (x0[j] - x0[j - 1]);
    return a + t * (b - a);
  });
}

// A property of a parameter of another size: its lists (of values, of
// free elements, or its '*' lists) resampled; one of a scalar, its first
function resized(property: unknown, size: number, from?: number[], to?: number[]): unknown {
  const fit = (list: unknown[]) => (size === 1 ? list[0] : list.length === size ? list : resampleAny(list, size, from, to));
  if (Array.isArray(property)) return fit(property);
  if (isObject(property)) {
    const copy: Options = {};
    for (const [key, value] of Object.entries(property)) {
      if (key.startsWith('*') && Array.isArray(value)) {
        if (size === 1) copy[key.slice(1)] = value[0];
        else copy[key] = fit(value);
      } else copy[key] = value;
    }
    return copy;
  }
  return property;
}

// A list resampled (see resample), its items not numbers the nearest
function resampleAny(list: unknown[], size: number, from?: number[], to?: number[]): unknown[] {
  if (list.every((v) => typeof v === 'number')) return resample(list, size, from, to);
  const places = resample(
    list.map((_, i) => i),
    size,
    from,
    to,
  );
  return places.map((p) => list[Math.round(p)]);
}

// How the parameters of a galaxy changed with an edit of its options:
// those of an option moved (e.g. the second velocity trait, now the first,
// as the first was removed) by a map of the paths of options (null: gone)
export type OwnerMap = (owner: Path) => Path | null;

// The properties of a galaxy in step with a change of its parameters (of
// one valid description to the next): the properties of parameters moved
// (by the map of their owners; matched in the order of the library among
// those of the same owner) renamed, of those gone removed, and of those of
// another size resampled (of a component, by the radii of its rings, if
// given before and after). Keys the parameters before did not have are
// kept as they are.
export function transition(
  properties: Properties,
  before: Param[],
  after: Param[],
  moved: OwnerMap | null,
  radii: { before: (component: number) => number[]; after: (component: number) => number[] },
): Properties {
  const known = new Map(after.map((p) => [p.name, p]));
  const renames = new Map<string, Param>();
  if (moved) {
    // (by owner: a parameter of the same name, if it has one; else, if the
    // owner has as many parameters as before (e.g. renamed), the one in
    // its place)
    const group = (params: Param[], key: (p: Param) => string | null) => {
      const groups = new Map<string, Param[]>();
      for (const p of params) {
        const k = key(p);
        if (k !== null) groups.set(k, [...(groups.get(k) ?? []), p]);
      }
      return groups;
    };
    const newByOwner = group(after, (p) => ownerKey(p.owner));
    const oldByOwner = group(before, (p) => {
      const owner = moved(p.owner);
      return owner === null ? null : ownerKey(owner);
    });
    for (const [owner, olds] of oldByOwner) {
      const news = newByOwner.get(owner) ?? [];
      const names = new Set(news.map((p) => p.name));
      olds.forEach((p, i) => {
        if (names.has(p.name)) renames.set(p.name, known.get(p.name)!);
        else if (olds.length === news.length && !olds.some((o) => names.has(o.name))) renames.set(p.name, news[i]);
      });
    }
  }
  const was = new Map(before.map((p) => [p.name, p]));
  const result: Properties = {};
  // (the keys of renamed parameters go first, so that a rename into a name
  // that is also renamed away does not lose either)
  const entries = Object.entries(properties);
  for (const [key, value] of entries) {
    const name = keyName(key);
    const old = was.get(name);
    if (old === undefined) {
      if (!(key in result)) result[key] = value;
      continue;
    }
    // (moved: to its match, or, its option in place, to its own name if it
    // still has it; gone with its option, or without a match elsewhere)
    let target: Param | undefined;
    if (moved) {
      const owner = moved(old.owner);
      target = owner === null ? undefined : (renames.get(name) ?? (ownerKey(owner) === ownerKey(old.owner) ? known.get(name) : undefined));
    } else target = known.get(name);
    if (target === undefined) continue;
    const newKey = target.name + key.slice(name.length);
    let next = value;
    if (target.size !== old.size) {
      const component = typeof target.owner[1] === 'number' && target.owner[0] === 'components' ? target.owner[1] : null;
      const oldComponent = typeof old.owner[1] === 'number' && old.owner[0] === 'components' ? old.owner[1] : null;
      const from = oldComponent === null ? undefined : radii.before(oldComponent);
      const to = component === null ? undefined : radii.after(component);
      next = resized(value, target.size, from?.length === old.size ? from : undefined, to?.length === target.size ? to : undefined);
    }
    result[newKey] = next;
  }
  return result;
}

// The properties to render a galaxy at: those of its parameters alone, of
// their sizes (a list of another size resampled), without keys of
// subscripts out of their place (the rest are for the library to read)
export function forRender(properties: Properties, params: Param[]): Properties {
  const known = new Map(params.map((p) => [p.name, p]));
  const result: Properties = {};
  for (const [key, value] of Object.entries(properties)) {
    const param = known.get(keyName(key));
    if (!param) continue;
    if (key !== param.name) {
      if (param.size > 1) result[key] = value;
      continue;
    }
    result[key] = resized(value, param.size);
  }
  return result;
}
