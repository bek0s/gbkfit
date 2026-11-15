// The types of the components of a galaxy as the library describes them
// (see the server's /components and /schema): the parts of each type in
// the order people read them (e.g. its rings, its density with its
// heights, its velocity), what each part is (one item of a family or a
// list of them, or options of their own, e.g. the rings), whether it can
// be left out, the types of its family it takes and the directions of its
// traits; its other options are its settings.
import type { Defs, Schema } from '../../components/SchemaForm';
import { asOneOrList, isObject, type Options } from './options';
import { memberOf, propertiesOf, retyped } from './schema';

// A type of component, as the server gives it
export interface ComponentType {
  parts: { option: string; heights?: string | null }[];
  traits: Record<string, string[]>;
  directions: string[];
  dispersion_directions: string[];
}

export type ComponentTypes = Record<string, ComponentType>;

// A part of a component (or the height traits paired with one)
export interface PartInfo {
  // Its option, its title and what it is
  option: string;
  title: string;
  description?: string;
  // The family of its items (e.g. 'vpt'), or the definition of its options
  // of their own (e.g. 'Nodes', 'Clouds')
  family: string | null;
  def: string | null;
  // A list of items (else one), whether it can be left out, and whether
  // an item of its list can be none (e.g. a velocity that does not change
  // with height)
  many: boolean;
  optional: boolean;
  nullable: boolean;
  // The types of its family that the component takes (null: every one),
  // and the directions of its traits (null: they have none)
  types: string[] | null;
  directions: string[] | null;
  // The height traits paired with its traits, if any
  heights: PartInfo | null;
}

// The options of every component that the screen sets in its head
export const HEAD_OPTIONS = new Set(['type', 'name']);

// The part of an option of a type of component
function partOf(defs: Defs, member: Schema, option: string, kind: ComponentType | undefined, heights: string | null): PartInfo | null {
  const { properties, required } = propertiesOf(member, defs);
  const property = properties[option];
  if (!property) return null;
  const branches = ((property.anyOf as Schema[] | undefined) ?? [property]).filter((b) => b.type !== 'null');
  const list = branches.find((b) => b.type === 'array');
  const items = list?.items as Schema | undefined;
  const itemBranches = (items?.anyOf as Schema[] | undefined) ?? (items ? [items] : []);
  const ref = [...branches, ...itemBranches].map((b) => b.$ref).find((r): r is string => typeof r === 'string');
  const name = ref?.replace('#/$defs/', '') ?? null;
  // (a family: its definition is a choice of members, each of a type)
  const isFamily = name !== null && isFamilyDef(defs, name);
  const family = isFamily ? name : null;
  const directions = family === 'vpt' ? (kind?.directions ?? null) : family === 'dpt' ? (kind?.dispersion_directions ?? null) : null;
  return {
    option,
    title: typeof property.title === 'string' ? property.title : option,
    description: typeof property.description === 'string' ? property.description : undefined,
    family,
    def: isFamily ? null : name,
    // (a list of items of a family: rings given as radii are not items)
    many: list !== undefined && family !== null,
    optional: !required.includes(option),
    nullable: itemBranches.some((b) => b.type === 'null'),
    types: family ? (kind?.traits[family] ?? null) : null,
    directions,
    heights: heights ? partOf(defs, member, heights, kind, null) : null,
  };
}

// Whether a definition is a family: its members have types
function isFamilyDef(defs: Defs, name: string): boolean {
  const node = defs[name];
  const members = (node?.oneOf as Schema[] | undefined) ?? [];
  return members.some((m) => {
    const forms = (m.oneOf as Schema[] | undefined) ?? [m];
    return forms.some((f) => (f.properties as Record<string, Schema> | undefined)?.type?.const !== undefined);
  });
}

// The parts of a type of component, in order
export function partsOfType(defs: Defs | undefined, types: ComponentTypes | undefined, type: unknown): PartInfo[] {
  if (!defs || !types || typeof type !== 'string') return [];
  const member = memberOf(defs, 'component', type);
  const kind = types[type];
  if (!member || !kind) return [];
  return kind.parts.map((p) => partOf(defs, member, p.option, kind, p.heights ?? null)).filter((p): p is PartInfo => p !== null);
}

// The settings of a type of component: its options that are not its
// parts, their heights, or set in its head
export function settingsOfType(defs: Defs | undefined, types: ComponentTypes | undefined, type: unknown): string[] {
  if (!defs || !types || typeof type !== 'string') return [];
  const member = memberOf(defs, 'component', type);
  if (!member) return [];
  const parts = new Set((types[type]?.parts ?? []).flatMap((p) => [p.option, ...(p.heights ? [p.heights] : [])]));
  return Object.keys(propertiesOf(member, defs).properties).filter((key) => !parts.has(key) && !HEAD_OPTIONS.has(key));
}

// A component given another type (e.g. a thin disk made thick): it keeps
// the items of its parts that the new type takes too (see retyped), and
// starts the parts it requires that are left without any
export function retypedComponent(defs: Defs | undefined, types: ComponentTypes | undefined, component: Options, type: string): Options {
  const parts = partsOfType(defs, types, type);
  // (an item of a type it takes; of heights, none too, if they can be)
  const takes = (part: PartInfo, item: unknown) =>
    item === null ? part.nullable : part.types === null || (isObject(item) && part.types.includes(String(item.type)));
  const kept: Options = { ...component };
  for (const part of parts) {
    for (const p of [part, part.heights]) {
      const value = p ? kept[p.option] : undefined;
      if (p === null || value === undefined || p.family === null) continue;
      const items = (Array.isArray(value) ? value : [value]).filter((item) => takes(p, item));
      const next = p.many ? asOneOrList(items) : items[0];
      if (next === undefined || next === null) delete kept[p.option];
      else kept[p.option] = next;
    }
  }
  return retyped(defs, 'component', kept, type);
}

// The title of an option of a type of a family (e.g. 'Ring width'), or
// the option
export function optionTitle(defs: Defs | undefined, family: string, type: unknown, option: string): string {
  const property = defs ? propertiesOf(memberOf(defs, family, type), defs).properties[option] : undefined;
  return typeof property?.title === 'string' ? property.title : option;
}

// What the options of an item name, by option (e.g. 'components' for the
// component that a mass component follows; see the library's
// config_references)
export function referencesOf(defs: Defs | undefined, family: string, type: unknown): Record<string, string> {
  if (!defs) return {};
  const { properties } = propertiesOf(memberOf(defs, family, type), defs);
  return Object.fromEntries(
    Object.entries(properties)
      .filter(([, p]) => typeof p.references === 'string')
      .map(([key, p]) => [key, p.references as string]),
  );
}

// The name a new component of a type is given (made unique): the last
// word of its type (a disk, thin or thick; a spheroid of any kind)
export function nameBase(type: string): string {
  return type.split('_').pop() ?? type;
}

// The extent of a component, for its line: its rings or shells, or how
// far a spheroid is drawn
export function extentOf(component: Options, radii: number[]): string {
  if (radii.length > 1) {
    const unit = component.type === 'cone' ? 'shells' : 'rings';
    return `${radii.length} ${unit} · 0–${Number(radii[radii.length - 1].toPrecision(4))}″`;
  }
  if (typeof component.rmax === 'number') return `to ${component.rmax}″`;
  return '';
}

// The family of an option of a type of a family (e.g. 'halo_profile', of
// the profile of a halo), if its value is a member of one
export function familyOfOption(defs: Defs | undefined, family: string, type: unknown, option: string): string | null {
  const property = (memberOf(defs, family, type)?.properties as Record<string, Schema> | undefined)?.[option];
  if (!property) return null;
  const branches = (property.anyOf as Schema[] | undefined) ?? [property];
  const ref = branches.map((b) => b.$ref).find((r): r is string => typeof r === 'string');
  return ref ? ref.replace('#/$defs/', '') : null;
}

// The options of a galaxy with what refers by name to one of its
// components or parts of its mass (what: 'components' or 'mass') renamed
// from one name to another: the options that reference them (see
// referencesOf; e.g. the component a traced disk follows, at any depth of
// the options of a type, or the parts of the mass a gravity screens)
export function withReferencesRenamed(defs: Defs | undefined, options: Options, what: 'components' | 'mass', from: string, to: string): Options {
  const renamed = (value: unknown) => (value === from ? to : value);
  const walk = (family: string, value: unknown): unknown => {
    if (Array.isArray(value)) return value.map((item) => walk(family, item));
    if (!isObject(value) || typeof value.type !== 'string') return value;
    const refs = referencesOf(defs, family, value.type);
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => {
        if (refs[key] === what) return [key, Array.isArray(item) ? item.map(renamed) : renamed(item)];
        const inner = familyOfOption(defs, family, value.type, key);
        return [key, inner ? walk(inner, item) : item];
      }),
    );
  };
  const next = { ...options };
  for (const [key, family] of [['components', 'component'], ['mass', 'mass_component'], ['gravity', 'gravity']] as const) {
    if (next[key] !== undefined) next[key] = walk(family, next[key]);
  }
  return next;
}
