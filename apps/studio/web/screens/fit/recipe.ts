// The recipe of a fit: a setup of the project, as its screen edits it.
//
// Its galaxies, lenses and observations are items: a piece of the project
// by name, the piece under a name of the fit ({name, use}: with several
// galaxies, each needs one, which prefixes its parameters), or options in
// place. Its section params is a list of layers, a later one replacing
// what an earlier one gives: the values of each galaxy and lens, kept with
// its piece by the Studio's convention (params/<piece>, in its own names,
// as its screen writes them), which the screen lays out from the items (a
// piece for one galaxy without a name, else {of: <name>, use: <piece>});
// the layers the user wrote, kept as they are; and the fit's own layer,
// the last one in place (its properties, in the names of the fit), where
// the edits of the screen go. Its section datasets pairs its observations
// with data by position: at the place of each, a dataset of the project by
// name, one in place, or null (none); no section, no data at all. Its
// section seen says what each observation sees, by position too: names of
// galaxies and of components ('galaxy.component', or the component alone
// of one galaxy), or null (everything); no section, everything for all.
import { isObject, type Options } from '../galaxy/options';

export type { Options };

export interface Item {
  // (its name in the fit, if any; the piece it is, if not in place; its
  // options, if in place)
  name: string | null;
  piece: string | null;
  options: Options | null;
}

export type Section = 'galaxies' | 'lenses' | 'observations';

// The items of a section (a list, or one item)
export function itemsOf(setup: Options, section: Section): Item[] {
  const value = setup[section];
  const list = Array.isArray(value) ? value : value === undefined || value === null ? [] : [value];
  return list.map((item): Item => {
    if (typeof item === 'string') return { name: null, piece: item, options: null };
    if (isObject(item) && typeof item.use === 'string') return { name: typeof item.name === 'string' ? item.name : null, piece: item.use, options: null };
    const options = isObject(item) ? item : {};
    return { name: typeof options.name === 'string' ? options.name : null, piece: null, options };
  });
}

function itemValue(item: Item): unknown {
  if (item.piece === null) return item.options ?? {};
  return item.name === null ? item.piece : { name: item.name, use: item.piece };
}

// The layer of the values of a galaxy or a lens of a piece that has them
// (see the head); none for one without
function itemLayer(item: Item, valued: Set<string>): unknown {
  if (item.piece === null || !valued.has(item.piece)) return undefined;
  return item.name === null ? item.piece : { of: item.name, use: item.piece };
}

function layersOf(setup: Options): unknown[] {
  const params = setup.params;
  return Array.isArray(params) ? params : params === undefined || params === null ? [] : [params];
}

// The fit's own layer: the last layer in place that is of no item
function isOwn(layer: unknown): layer is Options {
  return isObject(layer) && !('of' in layer) && !('use' in layer);
}

export function ownProperties(setup: Options): Options {
  const own = layersOf(setup).findLast(isOwn);
  return isObject(own?.properties) ? own.properties : {};
}

// The setup with a property of the fit's own layer (none: taken out, the
// value of the layers before it showing again)
export function withOwnProperty(setup: Options, key: string, property: unknown): Options {
  const layers = [...layersOf(setup)];
  let at = layers.findLastIndex(isOwn);
  if (at < 0) {
    if (property === undefined) return setup;
    layers.push({ properties: {} });
    at = layers.length - 1;
  }
  const own = layers[at] as Options;
  const properties = { ...(isObject(own.properties) ? own.properties : {}) };
  if (property === undefined) delete properties[key];
  else properties[key] = property;
  layers[at] = { ...own, properties };
  return { ...setup, params: layers };
}

// The setup with the items of a section, and the layers of the values of
// its galaxies and lenses laid out anew from them (those of the items
// before go, those of the items now come first; the others stay)
export function withItems(setup: Options, section: Section, items: Item[], valued: Set<string>): Options {
  const next = { ...setup, [section]: items.map(itemValue) };
  const before = valuedLayers(setup, valued).map((layer) => JSON.stringify(layer));
  const kept = layersOf(setup).filter((layer) => !before.includes(JSON.stringify(layer)));
  return { ...next, params: [...valuedLayers(next, valued), ...kept] };
}

function valuedLayers(setup: Options, valued: Set<string>): unknown[] {
  return (['galaxies', 'lenses'] as const).flatMap((section) => itemsOf(setup, section).map((item) => itemLayer(item, valued))).filter((layer) => layer !== undefined);
}

// The setup with an item of a section renamed (or named, from none), and
// what names it: the layers of its values, the keys of the fit's own
// layer of its parameters (their prefix, given the names of its
// parameters before), and the depth order
export function withItemName(setup: Options, section: Section, index: number, name: string | null, parameters: string[], valued: Set<string>): Options {
  const items = itemsOf(setup, section);
  const old = items[index].name;
  const prefix = (n: string | null) => (n === null ? '' : `${n}_`);
  items[index] = { ...items[index], name, options: items[index].options && { ...items[index].options, name: name ?? undefined } };
  let next = withItems(setup, section, items, valued);
  const own = ownProperties(next);
  for (const key of Object.keys(own)) {
    const base = key.split('[', 1)[0].trim();
    if (!parameters.includes(base)) continue;
    const renamed = prefix(name) + key.slice(prefix(old).length);
    next = withOwnProperty(withOwnProperty(next, key, undefined), renamed, own[key]);
  }
  if (Array.isArray(next.depth_order) && old !== null) next = { ...next, depth_order: next.depth_order.map((n) => (n === old ? name : n)) };
  if (section === 'galaxies') next = withSeenRenamed(next, old, name);
  return next;
}

// The setup with what its observations see of a galaxy renamed: its name,
// and the galaxy of its components ('galaxy.component'); a galaxy named
// from none (the one of the fit, as a second comes) qualifies the names
// of its components, and one that loses its name (the only one left)
// takes its own off them (the galaxy whole is then everything)
function withSeenRenamed(setup: Options, old: string | null, name: string | null): Options {
  const rename = (entry: string): string | null => {
    const [galaxy, component] = entry.includes('.') ? entry.split('.', 2) : [null, entry];
    if (old === null) return name !== null && galaxy === null ? `${name}.${component}` : entry;
    if (galaxy === old) return name === null ? component : `${name}.${component}`;
    return entry !== old ? entry : name;
  };
  const renamed = (names: string[]) => names.map(rename).filter((entry) => entry !== null);
  return withSeens(setup, seensOf(setup).map((names) => (names && renamed(names).length ? renamed(names) : null)));
}

// A name for an item of a piece in the fit: the last part of its name, as
// a name of parameters takes it (letters, digits and _), not one of those
// taken
export function nameFor(piece: string, taken: string[]): string {
  let base = (piece.split('/').pop() ?? piece).replace(/[^A-Za-z0-9_]/g, '_').replace(/^(\d)/, '_$1') || 'galaxy';
  if (base === 'mass') base = 'galaxy';
  let name = base;
  for (let i = 2; taken.includes(name); i++) name = `${base}${i}`;
  return name;
}

// The section datasets: the data of the observations, by position (a
// list, or one item; none: no section)
function datasetsOf(setup: Options): unknown[] {
  const value = setup.datasets;
  return Array.isArray(value) ? value : value === undefined ? [] : [value];
}

// The data the observation at an index is paired with (null: none)
export function datasetOf(setup: Options, index: number): unknown {
  return datasetsOf(setup)[index] ?? null;
}

// The setup with its section datasets of the data of its observations, by
// position: none if no observation has any
function withDatasets(setup: Options, datasets: unknown[]): Options {
  const { datasets: _old, ...rest } = setup;
  return datasets.some((dataset) => dataset !== null) ? { ...rest, datasets } : rest;
}

// The data of every observation, at its place (null: none), for a number
// of them
function datasetsFor(setup: Options, count: number): unknown[] {
  const datasets = datasetsOf(setup);
  return Array.from({ length: count }, (_, i) => datasets[i] ?? null);
}

// The setup with the observation at an index paired with a dataset of the
// project (by name; null: with none)
export function withDataset(setup: Options, index: number, dataset: string | null): Options {
  const datasets = datasetsFor(setup, itemsOf(setup, 'observations').length);
  datasets[index] = dataset;
  return withDatasets(setup, datasets);
}

// The section seen: what each observation sees, by position (names, or
// null: everything; a list, or none: no section)
function seensOf(setup: Options): (string[] | null)[] {
  const value = setup.seen;
  if (!Array.isArray(value)) return [];
  return value.map((item) => (typeof item === 'string' ? [item] : Array.isArray(item) ? item.map(String) : null));
}

// What the observation at an index sees (null: everything)
export function seenOf(setup: Options, index: number): string[] | null {
  return seensOf(setup)[index] ?? null;
}

// The setup with its section seen: none if every observation sees
// everything
function withSeens(setup: Options, seens: (string[] | null)[]): Options {
  const { seen: _old, ...rest } = setup;
  return seens.some((names) => names !== null) ? { ...rest, seen: seens } : rest;
}

// What every observation sees, at its place (null: everything), for a
// number of them
function seensFor(setup: Options, count: number): (string[] | null)[] {
  const seens = seensOf(setup);
  return Array.from({ length: count }, (_, i) => seens[i] ?? null);
}

// The setup with what the observation at an index sees (names of
// galaxies and components; none or null: everything)
export function withSeen(setup: Options, index: number, names: string[] | null): Options {
  const seens = seensFor(setup, itemsOf(setup, 'observations').length);
  seens[index] = names && names.length ? names : null;
  return withSeens(setup, seens);
}

// The setup with an observation of a piece added, without data, seeing
// everything
export function withObservation(setup: Options, piece: string, valued: Set<string>): Options {
  const items = itemsOf(setup, 'observations');
  const next = withItems(setup, 'observations', [...items, { name: null, piece, options: null }], valued);
  return withSeens(withDatasets(next, datasetsFor(setup, items.length + 1)), seensFor(setup, items.length + 1));
}

// The setup without the observation at an index, and without its data
// and what it sees
export function withoutObservation(setup: Options, index: number, valued: Set<string>): Options {
  const items = itemsOf(setup, 'observations');
  const next = withItems(setup, 'observations', items.filter((_, i) => i !== index), valued);
  const kept = <T,>(list: T[]) => list.filter((_, i) => i !== index);
  return withSeens(withDatasets(next, kept(datasetsFor(setup, items.length))), kept(seensFor(setup, items.length)));
}

// What gives a parameter its properties, for people: the fit's own layer
// (the setup itself), the values of a piece, or none (its default)
export function sourceOf(source: string | null | undefined, setupFile: string): { label: string; own: boolean } | null {
  if (!source) return null;
  if (source === setupFile) return { label: 'this fit', own: true };
  const match = /^params\/(.+)\.yaml$/.exec(source);
  return { label: match ? match[1] : source, own: false };
}
