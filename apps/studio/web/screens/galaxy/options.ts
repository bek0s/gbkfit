// The options of a galaxy as plain data (as its block holds them), and how
// the screen reads and changes them: by paths of keys and indices (e.g.
// ['components', 0, 'vptraits', 1]), each change a copy.
// Several options of the library are one item or a list of them (e.g.
// components, ptraits, the lines of an emission): read as lists, and
// written back as one item when one is left.
import type { Options } from '../../components/SchemaForm';

export type { Options };
export type Path = (string | number)[];

export function isObject(value: unknown): value is Options {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// A value as a list of options: a list, one item, or none
export function asList(value: unknown): Options[] {
  if (Array.isArray(value)) return value.filter(isObject);
  return isObject(value) ? [value] : [];
}

// Items as an option that is one item or a list: none (undefined), the
// one, or the list
export function asOneOrList<T>(items: T[]): T | T[] | undefined {
  return items.length === 0 ? undefined : items.length === 1 ? items[0] : items;
}

// The value at a path (undefined if there is none). An index into one item
// (not a list) reads that item at 0, as the library reads it.
export function getAt(options: unknown, path: Path): unknown {
  let value = options;
  for (const key of path) {
    if (typeof key === 'number') {
      if (Array.isArray(value)) value = value[key];
      else value = key === 0 && isObject(value) ? value : undefined;
    } else value = isObject(value) ? value[key] : undefined;
    if (value === undefined) return undefined;
  }
  return value;
}

// Options with the value at a path replaced (undefined: removed); the rest
// as it is. An item removed from a list leaves the others in their order
// (and none: the option removed). An index into one item (not a list) is
// into that item at 0, which stays one item until a second is added.
export function setAt(options: Options, path: Path, value: unknown): Options {
  return setIn(options, path, value) as Options;
}

function setIn(node: unknown, path: Path, value: unknown): unknown {
  if (path.length === 0) return value;
  const [key, ...rest] = path;
  if (typeof key === 'number') {
    const list = Array.isArray(node) ? [...node] : node === undefined ? [] : [node];
    const next = setIn(list[key], rest, value);
    if (next === undefined) list.splice(key, 1);
    else list[key] = next;
    return Array.isArray(node) ? (list.length ? list : undefined) : asOneOrList(list);
  }
  const copy: Options = isObject(node) ? { ...node } : {};
  const next = setIn(copy[key], rest, value);
  if (next === undefined) delete copy[key];
  else copy[key] = next;
  return copy;
}

// The components of a galaxy
export function componentsOf(options: Options): Options[] {
  return asList(options.components);
}

// The name of a component (that of its options; else of its place)
export function componentName(component: Options | undefined, index: number): string {
  return typeof component?.name === 'string' && component.name ? component.name : `component ${index + 1}`;
}

// A name not taken by others, of a base (base, base2, base3, ...)
export function freshName(taken: Iterable<unknown>, base: string): string {
  const names = new Set(taken);
  for (let i = 1; ; i++) {
    const name = i === 1 ? base : `${base}${i}`;
    if (!names.has(name)) return name;
  }
}

// Whether a path starts with another
export function startsWith(path: Path, start: Path): boolean {
  return start.length <= path.length && start.every((key, i) => path[i] === key);
}

// The lines of an emission as a list: names (of the catalogue), or lines
// of options ({name, rest})
export function lineList(value: unknown): unknown[] {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}

// The name of a line: its name in the catalogue, or that of its options
export function lineName(line: unknown): string {
  return typeof line === 'string' ? line : String((line as Options | null)?.name ?? '?');
}

// The galaxy with the coefficients of the opacity of the lines of each
// component (self-absorption) in step with its lines: of one unnamed line,
// one set of traits; of named lines, the traits of each by its name (kept
// where it had them; else those of the first, or uniform).
export function withOpacitiesInStep(options: Options): Options {
  if (options.components === undefined) return options;
  let changed = false;
  const next = componentsOf(options).map((c) => {
    const opacity = c.opacity;
    if (!isObject(opacity) || opacity.type !== 'line') return c;
    const names = lineList(isObject(c.emission) ? c.emission.lines : undefined).map(lineName);
    const given = opacity.coefficients;
    const byLine = isObject(given) && !('type' in given);
    const single = byLine ? Object.values(given)[0] : given;
    const fallback = single ?? { type: 'uniform' };
    const coefficients =
      names.length === 0 ? fallback : Object.fromEntries(names.map((name) => [name, (byLine ? given[name] : undefined) ?? fallback]));
    if (JSON.stringify(coefficients) === JSON.stringify(given)) return c;
    changed = true;
    return { ...c, opacity: { ...opacity, coefficients } };
  });
  return changed ? { ...options, components: Array.isArray(options.components) ? next : next[0] } : options;
}

// The radii of rings of any form (see the library's Nodes): every sep from
// 0 to the first at or beyond max, len of them from 0 to max, or those
// given
export function ringRadii(value: unknown): number[] {
  if (Array.isArray(value)) return value.map(Number).filter(Number.isFinite);
  if (!isObject(value)) return [];
  const max = Number(value.max);
  const sep = Number(value.sep);
  const len = Number(value.len);
  if (!(max > 0)) return [];
  if (sep > 0 && sep <= max) {
    const n = Math.ceil(max / sep - 1e-9);
    return Array.from({ length: n + 1 }, (_, i) => i * sep);
  }
  if (len >= 2) return Array.from({ length: len }, (_, i) => (i * max) / (len - 1));
  return [];
}

// A round step near a value (1, 2 or 5 times a power of 10)
export function roundStep(value: number): number {
  const power = 10 ** Math.floor(Math.log10(value));
  const lead = value / power;
  return (lead < 1.5 ? 1 : lead < 3.5 ? 2 : lead < 7.5 ? 5 : 10) * power;
}
