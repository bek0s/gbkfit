// The schema of the library (see the server's /schema) as the galaxy
// screen reads it: the types of a family (e.g. the velocity traits), what
// each is, the options a new item of a type starts with, and the options
// of an item that the screen leaves to a form of the schema.
import { choicesOf, membersOf, resolve, typeOf, type Defs, type Schema } from '../../components/SchemaForm';
import { ITEM_STARTS, NUMBER_STARTS } from './catalogue';
import { freshName, isObject, type Options } from './options';

export interface TypeChoice {
  value: string;
  label: string;
  description?: string;
}

// The member of a family of a type (its schema)
export function memberOf(defs: Defs | undefined, family: string, type: unknown): Schema | undefined {
  if (!defs?.[family] || typeof type !== 'string') return undefined;
  return membersOf(defs[family], defs).find((m) => typeOf(m) === type);
}

// The types of a family, with the title of each and what it is (the
// first sentence of its description); of those given alone, if given
// (e.g. those a cone takes)
export function typesOf(defs: Defs | undefined, family: string, only?: string[] | null): TypeChoice[] {
  if (!defs?.[family]) return [];
  // (each type once: a family can have two members of one type, e.g. of
  // options or of a file)
  const types = new Map<string, TypeChoice>();
  for (const m of membersOf(defs[family], defs)) {
    const type = typeOf(m)!;
    if ((only && !only.includes(type)) || types.has(type)) continue;
    types.set(type, {
      value: type,
      label: typeof m.title === 'string' ? m.title : type.replaceAll('_', ' '),
      description: typeof m.description === 'string' ? firstSentence(m.description) : undefined,
    });
  }
  return [...types.values()];
}

// The first sentence of a text (not cut at e.g. or i.e.)
export function firstSentence(text: string): string {
  const ends = /\.(?=\s+[A-Z(]|\s*$)/g;
  for (let end = ends.exec(text); end !== null; end = ends.exec(text)) {
    const sentence = text.slice(0, end.index + 1);
    if (!ABBREVIATION.test(sentence)) return sentence.trim();
  }
  return text.trim();
}

// The abbreviations whose periods do not end a sentence (e.g. 'Epinat et
// al. (2008)', 'e.g. Voronoi bins')
const ABBREVIATION = /\b(?:e\.g|i\.e|et al|cf|vs)\.$/i;

// The properties of a schema of an object, and those it requires
export function propertiesOf(node: Schema | undefined, defs: Defs): { properties: Record<string, Schema>; required: string[] } {
  const resolved = node ? resolve(node, defs) : {};
  return {
    properties: (resolved.properties ?? {}) as Record<string, Schema>,
    required: (resolved.required as string[] | undefined) ?? [],
  };
}

// The options of an object of the schema with those it requires: at their
// first choice (e.g. the direction of a velocity trait), of the first type
// of a family (with what it requires), or a name not taken (names)
function filled(defs: Defs, node: Schema, given: Options, names: string[]): Options {
  const { properties, required } = propertiesOf(node, defs);
  const result: Options = { ...given };
  for (const key of required) {
    if (key in result) continue;
    if (key === 'name') {
      result.name = freshName(names, String(result.type ?? 'item'));
      continue;
    }
    const property = resolve(properties[key] ?? {}, defs);
    const branches = ((property.anyOf as Schema[] | undefined) ?? [property]).map((b) => resolve(b, defs)).filter((b) => b.type !== 'null');
    const first = branches[0];
    if (!first) continue;
    const choices = choicesOf(first);
    if (choices) result[key] = choices[0].value;
    else if (first.type === 'number' || first.type === 'integer') {
      const start = NUMBER_STARTS[key] ?? property.default ?? first.default ?? first.minimum ?? first.exclusiveMinimum;
      if (typeof start === 'number') result[key] = start;
    } else if (membersOf(first, defs).length) {
      const member = membersOf(first, defs)[0];
      result[key] = filled(defs, member, { type: typeOf(member) }, []);
    } else if (first.type === 'object') result[key] = filled(defs, first, {}, []);
  }
  return result;
}

// A new item of a type of a family, with what it requires (see filled),
// named apart from the names given (those of its list)
export function startOf(defs: Defs | undefined, family: string, type: string, names: string[] = [], extra: Options = {}): Options {
  const member = memberOf(defs, family, type);
  const start = { type, ...(ITEM_STARTS[family]?.[type] ?? {}), ...extra };
  return defs && member ? filled(defs, member, start, names) : start;
}

// An item given another type: it keeps the options of the old that the
// new type has too (e.g. the zmax and name of a height trait), and starts
// the others (see startOf)
export function retyped(defs: Defs | undefined, family: string, item: unknown, type: string): Options {
  const member = memberOf(defs, family, type);
  const has = member && defs ? propertiesOf(member, defs).properties : {};
  const kept = isObject(item) ? Object.entries(item).filter(([key]) => key !== 'type' && key in has) : [];
  return { ...startOf(defs, family, type), ...Object.fromEntries(kept) };
}

// The options of a member that a form shows, without those the screen
// sets otherwise (e.g. the type, the name and the direction of a trait,
// the parts of a component): its schema, and whether it has any
export function restOf(defs: Defs | undefined, member: Schema | undefined, without: Set<string>): Schema | null {
  if (!defs || !member) return null;
  const resolved = resolve(member, defs);
  const { properties, required } = propertiesOf(resolved, defs);
  const kept = Object.entries(properties).filter(([key]) => !without.has(key) && key !== 'type');
  if (kept.length === 0) return null;
  return { ...resolved, properties: Object.fromEntries(kept), required: required.filter((key) => !without.has(key)) };
}

// The description of an option of a member (its first sentence)
export function describe(defs: Defs | undefined, member: Schema | undefined, key: string): string | undefined {
  if (!defs || !member) return undefined;
  const text = propertiesOf(member, defs).properties[key]?.description;
  return typeof text === 'string' ? firstSentence(text) : undefined;
}
