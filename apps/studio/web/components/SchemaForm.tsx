// A form of options made from the JSON Schema of the configurations of
// GBKFIT (see the server's /schema, written from the signatures of the
// library): a field for each option, of its type, range, unit, default
// and description. A family (e.g. a PSF: one of its members, chosen by
// its option type) is a choice of the member and the fields of its
// options; a list of a family (e.g. the PSFs of a sum) is a list of such
// forms. A file is one of the project (imported or uploaded), and its
// array, if it has several. What the form cannot edit (e.g. a table along
// the spectral axis) is shown as it is, and kept.
import { Autocomplete, NumberInput, Slider } from '@mantine/core';
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { useDescription } from '../api/queries';
import { FileSources } from './FileSources';
import { Button, Close, TextField } from './ui/basics';
import { joinPath } from './FilePicker';
import { Choice } from './ui/inputs';

// A node of a JSON Schema (as the library writes them), and its
// definitions (its $defs)
export type Schema = { [key: string]: unknown };
export type Defs = Record<string, Schema>;
export type Options = Record<string, unknown>;

// What the fields share: the definitions of the schema, the project (the
// files are its own, relative to it), and the options a member of a
// family starts with, by family and type (e.g. a Gaussian PSF of 1.5")
interface FormContext {
  defs: Defs;
  project: string;
  starts: Record<string, Record<string, Options>>;
  // The blocks of the project by folder (their names), and how one opens
  blocks: Record<string, string[]>;
  openBlock?: (folder: string, name: string) => void;
  // Whether numbers have sliders too (see sliderOf)
  sliders: boolean;
}

// The families whose options can be a block of the project instead, by
// its name, and the folders of those blocks (see gbkfit.project)
const BLOCK_FOLDERS: Record<string, string> = { dataset: 'datasets', psf: 'psfs', lsf: 'lsfs', noise: 'noise' };

const Context = createContext<FormContext | null>(null);

function useForm(): FormContext {
  const form = useContext(Context);
  if (form === null) throw new Error('a field of a schema is used out of a SchemaForm');
  return form;
}

// The form of a definition of the schema (e.g. 'psf'), or of a node of it
// (e.g. the density of a disk), of its options
export function SchemaForm({
  defs,
  def,
  node: given,
  project,
  value,
  onChange,
  starts = {},
  blocks = {},
  openBlock,
  sliders = false,
}: {
  defs: Defs;
  def?: string;
  node?: Schema;
  project: string;
  value: Options;
  onChange: (value: Options) => void;
  starts?: Record<string, Record<string, Options>>;
  blocks?: Record<string, string[]>;
  openBlock?: (folder: string, name: string) => void;
  sliders?: boolean;
}) {
  const node = given ?? (def === undefined ? undefined : defs[def]);
  if (node === undefined) return <p className="error">The schema has no {def}.</p>;
  return (
    <Context.Provider value={{ defs, project, starts, blocks, openBlock, sliders }}>
      <div className="schema-form">
        <Field node={node} family={def} value={value} onChange={(next) => onChange(next as Options)} />
      </div>
    </Context.Provider>
  );
}

// A node of the schema, its references followed
export function resolve(node: Schema, defs: Defs): Schema {
  const ref = node.$ref;
  if (typeof ref === 'string' && ref.startsWith('#/$defs/')) return resolve(defs[ref.slice(8)] ?? {}, defs);
  return node;
}

// The members of a family: the objects of its choice (oneOf, nested),
// each of a type (its const) and with the title of its type, in the order
// of the schema
export function membersOf(node: Schema, defs: Defs): Schema[] {
  const resolved = resolve(node, defs);
  const choices = resolved.oneOf as Schema[] | undefined;
  if (!choices) return typeOf(resolved) === undefined ? [] : [resolved];
  // (the forms of a member, e.g. a beam by its axes or by a file, have its
  // title and description; their own, of their constructors, is kept as
  // form)
  return choices.flatMap((choice) =>
    membersOf(choice, defs).map((member) => ({
      ...member,
      ...(resolved.title === undefined ? {} : { title: resolved.title }),
      ...(resolved.description === undefined ? {} : { description: resolved.description, form: member.description }),
    })),
  );
}

// The title of a type of a family (that of its members), or the type
export function titleOf(defs: Defs | undefined, family: string, type: string): string {
  const member = defs?.[family] ? membersOf(defs[family], defs).find((m) => typeOf(m) === type) : undefined;
  return typeof member?.title === 'string' ? member.title : type;
}

// The choices of a node of the schema, each with its title: those of a
// choice of constants (oneOf of {const, title}), or of an enum (titled by
// their values); null if it is not a choice
export function choicesOf(node: Schema | undefined): { value: unknown; title: string }[] | null {
  if (!node) return null;
  if (Array.isArray(node.enum)) return (node.enum as unknown[]).map((value) => ({ value, title: String(value) }));
  const options = node.oneOf as Schema[] | undefined;
  if (!Array.isArray(options) || options.length === 0 || !options.every((o) => 'const' in o)) return null;
  return options.map((o) => ({ value: o.const, title: typeof o.title === 'string' ? o.title : String(o.const) }));
}

// The type of a member of a family (the const of its option type)
export function typeOf(node: Schema): string | undefined {
  const type = (node.properties as Record<string, Schema> | undefined)?.type;
  return typeof type?.const === 'string' ? type.const : undefined;
}

// The name of a family a node is (that of its definition), if it is one
function familyOf(node: Schema): string | undefined {
  const ref = node.$ref;
  return typeof ref === 'string' && ref.startsWith('#/$defs/') ? ref.slice(8) : undefined;
}

// The field of a node of the schema, of its value (undefined: none)
function Field({
  node,
  value,
  onChange,
  family,
  nullable = false,
  slider,
}: {
  node: Schema;
  value: unknown;
  onChange: (value: unknown) => void;
  // The family it is (e.g. 'psf'), for the options its members start with
  family?: string;
  // Whether it can be none
  nullable?: boolean;
  // The slider of a number, if it has one
  slider?: SliderSpec | null;
}) {
  const { defs } = useForm();
  const resolved = resolve(node, defs);
  const name = family ?? familyOf(node);
  if (typeof value === 'string' && name && BLOCK_FOLDERS[name]) return <BlockField family={name} value={value} onChange={onChange} />;
  if (membersOf(resolved, defs).length > 0 && resolved.oneOf) {
    return <MemberField members={membersOf(resolved, defs)} family={name ?? ''} nullable={nullable} value={value} onChange={onChange} />;
  }
  const branches = resolved.anyOf as Schema[] | undefined;
  if (branches) return <AnyField branches={branches} node={resolved} value={value} onChange={onChange} slider={slider} />;
  return <PlainField node={resolved} value={value} onChange={onChange} slider={slider} />;
}

// A block of the project named in place of options (e.g. the dataset of
// an observation): its name, which can be another, or options in place
// instead; it opens
function BlockField({ family, value, onChange }: { family: string; value: string; onChange: (value: unknown) => void }) {
  const { blocks, starts, openBlock } = useForm();
  const folder = BLOCK_FOLDERS[family];
  const names = blocks[folder] ?? [];
  const first = Object.entries(starts[family] ?? {})[0];
  return (
    <div className="schema-block">
      <Choice
        items={[
          ...(names.includes(value) ? [] : [{ value, label: `${value} (missing)` }]),
          ...names.map((name) => ({ value: name, label: name })),
          { value: '\u0000', label: 'Options here instead…' },
        ]}
        value={value}
        label={`The ${family} of the project`}
        onChange={(next) => onChange(next === '\u0000' ? (first ? { type: first[0], ...first[1] } : {}) : next)}
      />
      {openBlock && names.includes(value) && (
        <Button kind="small" title={`Open the ${family} ${value}`} onClick={() => openBlock(folder, value)}>
          open →
        </Button>
      )}
    </div>
  );
}

// A member of a family: the choice of its type, and the fields of its
// options; a new type keeps the options of the old that it has too
function MemberField({
  members,
  family,
  nullable,
  value,
  onChange,
}: {
  members: Schema[];
  family: string;
  nullable: boolean;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const { starts, blocks } = useForm();
  // (a family of blocks of the project can name one instead)
  const named = BLOCK_FOLDERS[family] ? (blocks[BLOCK_FOLDERS[family]] ?? []) : [];
  const options = isObject(value) ? value : {};
  const type = typeof options.type === 'string' ? options.type : undefined;
  // (a type of several members, e.g. of options or of a file: that whose
  // options it has, the one chosen of those that fit, e.g. before any of
  // its options are given, or the first)
  const [chosen, setChosen] = useState(0);
  const ofType = members.filter((m) => typeOf(m) === type);
  const fit = ofType.filter((m) => Object.keys(options).every((key) => key === 'type' || key in propertiesOf(m)));
  const member = fit.includes(ofType[chosen]) ? ofType[chosen] : (fit[0] ?? ofType[0]);
  // (in the order of their texts, the common ones first; others after)
  const order = TYPE_ORDER;
  const rank = (t: string) => (order.includes(t) ? order.indexOf(t) : order.length);
  const types = [...new Set(members.map((m) => typeOf(m)!))].sort((a, b) => rank(a) - rank(b));
  // (a form chosen keeps the options it has too, and starts the others as
  // the type does)
  const chooseForm = (index: number) => {
    const target = ofType[index];
    setChosen(index);
    const start = Object.entries(starts[family]?.[type!] ?? {}).filter(([key]) => key in propertiesOf(target));
    const kept = Object.entries(options).filter(([key]) => key !== 'type' && key in propertiesOf(target));
    onChange({ type, ...Object.fromEntries(start), ...Object.fromEntries(kept) });
  };
  const choose = (next: string) => {
    if (next === '') return onChange(undefined);
    if (next.startsWith('block:')) return onChange(next.slice(6));
    const start = starts[family]?.[next] ?? {};
    const target = members.find((m) => typeOf(m) === next);
    const kept = Object.fromEntries(Object.entries(options).filter(([key]) => key !== 'type' && target && key in propertiesOf(target)));
    onChange({ type: next, ...start, ...kept });
  };
  return (
    <div className="schema-member">
      <div className="schema-row">
        <span className="schema-label">type</span>
        <Choice
          items={[
            ...(nullable ? [{ value: '', label: 'None' }] : type === undefined ? [{ value: '', label: 'Choose…' }] : []),
            ...(type !== undefined && !types.includes(type) ? [{ value: type, label: type }] : []),
            ...types.map((t) => {
              const of = members.find((m) => typeOf(m) === t);
              return { value: t, label: typeof of?.title === 'string' ? of.title : t, description: of?.description as string | undefined };
            }),
            ...named.map((name) => ({ value: `block:${name}`, label: name, description: `A block of the project (its ${family})` })),
          ]}
          value={type ?? ''}
          label={`The type of the ${family}`}
          onChange={choose}
        />
      </div>
      {ofType.length > 1 && member && (
        <div className="schema-row">
          <span className="schema-label text">Given by</span>
          <Choice
            items={ofType.map((m, i) => ({ value: String(i), label: formLabel(m), description: m.form as string | undefined }))}
            value={String(ofType.indexOf(member))}
            label={`How the ${family} is given`}
            onChange={(next) => chooseForm(Number(next))}
          />
        </div>
      )}
      {member && <ObjectFields node={member} value={options} onChange={onChange} />}
    </div>
  );
}

// The order of the types of the instruments in their menus: the common
// ones first (others after them; their titles and what each is come from
// the library)
const TYPE_ORDER = ['point', 'gauss', 'moffat', 'ggauss', 'lorentz', 'hanning', 'image', 'images', 'sum', 'convolution'];

// A form of a member, by its options (e.g. bmaj, bmin, bpa; file)
function formLabel(form: Schema): string {
  const keys = Object.keys(propertiesOf(form)).filter((key) => key !== 'type');
  return keys.length ? keys.join(', ') : 'nothing';
}

export function propertiesOf(node: Schema): Record<string, Schema> {
  return (node.properties as Record<string, Schema> | undefined) ?? {};
}

function isObject(value: unknown): value is Options {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// The fields of the options of an object (but its type), each in a row
function ObjectFields({ node, value, onChange }: { node: Schema; value: Options; onChange: (value: Options) => void }) {
  const required = (node.required as string[] | undefined) ?? [];
  const properties = Object.entries(propertiesOf(node)).filter(([key]) => key !== 'type');
  const set = (key: string, next: unknown) => {
    const { [key]: _, ...rest } = value;
    onChange(next === undefined ? rest : { ...rest, [key]: next });
  };
  // (options it has that the schema has not: kept, and shown)
  const unknown = Object.keys(value).filter((key) => key !== 'type' && !(key in propertiesOf(node)));
  const { defs, sliders } = useForm();
  return (
    <>
      {properties.map(([key, property]) => {
        const slider = sliders ? sliderOf(key, property, defs, value[key]) : null;
        // (an optional object of options of its own, not given: added on
        // demand, as its fields are many; given, it can go)
        const optional = !required.includes(key) && isPlainObject(property, defs);
        return (
          <Row key={key} name={key} node={property} required={required.includes(key)}>
            {optional && value[key] === undefined ? (
              <Button kind="small" title={typeof property.description === 'string' ? property.description : undefined} onClick={() => set(key, {})}>
                + Set
              </Button>
            ) : (
              <Field node={property} value={value[key]} onChange={(next) => set(key, next)} slider={slider} />
            )}
            {optional && value[key] !== undefined && <Close label={`Remove ${key}`} onClick={() => set(key, undefined)} />}
          </Row>
        );
      })}
      {unknown.map((key) => (
        <Row key={key} name={key} node={{ description: 'An option this type does not take: the configuration is invalid with it.' }} required={false}>
          <span className="error">{JSON.stringify(value[key])}</span>
          <Close label={`Remove ${key}`} onClick={() => set(key, undefined)} />
        </Row>
      ))}
    </>
  );
}

// Whether a node is options of their own (an object of properties, not a
// family), or none
function isPlainObject(node: Schema, defs: Defs): boolean {
  const resolved = resolve(node, defs);
  const branches = ((resolved.anyOf as Schema[] | undefined) ?? [resolved]).map((b) => resolve(b, defs)).filter((b) => b.type !== 'null');
  return branches.length === 1 && branches[0].type === 'object' && branches[0].properties !== undefined && branches[0].oneOf === undefined;
}

// A row of an option: its name (marked if it must be given), its field,
// and its unit, with its description as the title
function Row({ name, node, required, children }: { name: string; node: Schema; required: boolean; children: ReactNode }) {
  const unit = typeof node.unit === 'string' ? node.unit : null;
  const description = typeof node.description === 'string' ? node.description : undefined;
  return (
    <div className="schema-row" title={description}>
      <span className="schema-label">
        {name}
        {required && <span className="schema-required" title="It must be given">*</span>}
      </span>
      <div className="schema-field">
        {children}
        {/* (a column of units in every row, so that the fields line up) */}
        <span className="schema-unit">{unit ?? ''}</span>
      </div>
    </div>
  );
}

// A choice of schemas (anyOf), by the value: e.g. a number or a table
// along the spectral axis; a filename or a filename and its HDU; a value
// or none (null: its default)
function AnyField({
  branches,
  node,
  value,
  onChange,
  slider,
}: {
  branches: Schema[];
  node: Schema;
  value: unknown;
  onChange: (value: unknown) => void;
  slider?: SliderSpec | null;
}) {
  const { defs } = useForm();
  const resolved = branches.map((b) => resolve(b, defs));
  const nullable = resolved.some((b) => b.type === 'null');
  const kinds = resolved.map((_, i) => i).filter((i) => resolved[i].type !== 'null');
  const find = (test: (b: Schema) => boolean) => kinds.find((i) => test(resolved[i]));
  const field = (i: number) => <Field node={branches[i]} nullable={nullable} value={value} onChange={onChange} />;
  // (a file: a path, or a path and its HDU)
  if (resolved.some((b) => b.format === 'path')) return <FileField value={value} onChange={onChange} />;
  // (a quantity, as its text: e.g. a width of the light, or a table of it)
  const quantity = find((b) => b.format === 'quantity');
  if (quantity !== undefined && (value === undefined || typeof value === 'string')) {
    return <QuantityField node={resolved[quantity]} value={value} onChange={onChange} />;
  }
  // (a name: of a block of the project, or a choice)
  if (typeof value === 'string' || value === undefined) {
    const family = kinds.map((i) => familyOf(branches[i])).find((f) => f && BLOCK_FOLDERS[f]);
    if (typeof value === 'string' && family) return <BlockField family={family} value={value} onChange={onChange} />;
    const choice = find((b) => choicesOf(b) !== null && (value === undefined || choicesOf(b)!.some((c) => c.value === value)));
    if (choice !== undefined) return <PlainField node={{ ...resolved[choice], default: node.default }} value={value} onChange={onChange} />;
    const text = find((b) => b.type === 'string' && b.format !== 'path');
    if (typeof value === 'string' && text !== undefined) return <PlainField node={resolved[text]} value={value} onChange={onChange} />;
  }
  const number = find((b) => b.type === 'number' || b.type === 'integer');
  if (number !== undefined && (value === undefined || value === null || typeof value === 'number')) {
    return <NumberValue node={{ ...resolved[number], default: node.default }} value={value as number | undefined} onChange={onChange} slider={slider} />;
  }
  const flag = find((b) => b.type === 'boolean');
  if (flag !== undefined && (value === undefined || typeof value === 'boolean')) {
    return <PlainField node={{ ...resolved[flag], default: node.default }} value={value} onChange={onChange} />;
  }
  const plain = find((b) => b.type === 'array' && resolve((b.items as Schema) ?? {}, defs).type === 'number');
  if (plain !== undefined && (value === undefined || value === null || (Array.isArray(value) && value.every((v) => typeof v === 'number')))) {
    return <NumbersField value={value as number[] | undefined} onChange={onChange} placeholder={nullable ? 'None' : undefined} />;
  }
  const list = find((b) => b.type === 'array');
  if (Array.isArray(value) && list !== undefined) return field(list);
  const options = find((b) => b.type !== 'array' && (b.oneOf !== undefined || b.type === 'object'));
  if ((value === undefined || isObject(value)) && options !== undefined) return field(options);
  if (kinds.length === 1) return field(kinds[0]);
  return <Kept value={value} onChange={onChange} />;
}

// The field of a node of one type
function PlainField({
  node,
  value,
  onChange,
  slider,
}: {
  node: Schema;
  value: unknown;
  onChange: (value: unknown) => void;
  slider?: SliderSpec | null;
}) {
  const { defs } = useForm();
  if (node.type === 'number' || node.type === 'integer') {
    return typeof value === 'number' || value === undefined ? (
      <NumberValue node={node} value={value} onChange={onChange} slider={slider} />
    ) : (
      <Kept value={value} onChange={onChange} />
    );
  }
  if (node.type === 'boolean') {
    return (
      <Choice
        items={[
          { value: '', label: node.default === undefined ? '—' : `default (${String(node.default)})` },
          { value: 'true', label: 'Yes' },
          { value: 'false', label: 'No' },
        ]}
        value={value === undefined ? '' : String(value)}
        label="yes or no"
        onChange={(next) => onChange(next === '' ? undefined : next === 'true')}
      />
    );
  }
  const choices = choicesOf(node);
  if (choices) {
    const titled = (v: unknown) => choices.find((c) => c.value === v)?.title ?? String(v);
    return (
      <Choice
        items={[{ value: '', label: node.default === undefined ? '—' : `default (${titled(node.default)})` }, ...choices.map((c) => ({ value: String(c.value), label: c.title }))]}
        value={value === undefined ? '' : String(value)}
        label="a choice"
        onChange={(next) => onChange(next === '' ? undefined : choices.find((c) => String(c.value) === next)?.value)}
      />
    );
  }
  if (node.type === 'string' && node.format === 'quantity') {
    return typeof value === 'string' || value === undefined ? (
      <QuantityField node={node} value={value} onChange={onChange} />
    ) : (
      <Kept value={value} onChange={onChange} />
    );
  }
  if (node.type === 'string') {
    return <TextField value={typeof value === 'string' ? value : ''} placeholder={node.default != null ? String(node.default) : undefined} label="text" onChange={(next) => onChange(next === '' ? undefined : next)} />;
  }
  if (node.type === 'array') {
    const items = resolve((node.items as Schema) ?? {}, defs);
    if (items.type === 'number' || items.type === 'integer') {
      return <NumbersField value={Array.isArray(value) ? (value as number[]) : undefined} onChange={onChange} />;
    }
    if (membersOf(items, defs).length > 0) return <ListField node={node.items as Schema} value={value} onChange={onChange} />;
  }
  if (node.type === 'object' && node.properties) {
    return (
      <div className="schema-object">
        <ObjectFields node={node} value={isObject(value) ? value : {}} onChange={onChange} />
      </div>
    );
  }
  return <Kept value={value} onChange={onChange} />;
}

// A quantity of one value, as its text (e.g. '2.5 Angstrom'; the library
// reads it): its number, and its unit, with the units of its kinds as
// suggestions (any other can be typed)
function QuantityField({ node, value, onChange }: { node: Schema; value: string | undefined; onChange: (value: unknown) => void }) {
  const [number, unit] = splitQuantity(value ?? '');
  const kinds = (node.unit_kinds as Record<string, string[]> | undefined) ?? {};
  const units = Object.entries(kinds).map(([kind, items]) => ({ group: kind, items }));
  const join = (n: string, u: string) => {
    const text = `${n.trim()} ${u.trim()}`.trim();
    return text === '' ? undefined : text;
  };
  return (
    <div className="schema-quantity">
      <TextField value={number} placeholder="value" label="value" onChange={(next) => onChange(join(next, unit))} mono />
      <Autocomplete
        className="schema-quantity-unit"
        value={unit}
        data={units}
        placeholder="unit"
        aria-label="unit"
        size="xs"
        onChange={(next) => onChange(join(number, next))}
      />
    </div>
  );
}

// The number and the unit of the text of a quantity (e.g. '2.5 Angstrom');
// a text that does not start with a number is all unit
function splitQuantity(text: string): [string, string] {
  const match = /^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$/.exec(text);
  return match ? [match[1], match[2].trim()] : ['', text.trim()];
}

// The slider of a number: from min to max by step; or, logarithmic (of a
// width, above 0 without an end), from min to max, each step the same
// fraction of the value
export interface SliderSpec {
  min: number;
  max: number;
  step: number;
  log?: boolean;
}

// The slider of an option, by its name, its range and its unit, if it has
// one: a range of its own (e.g. an axis ratio, in (0, 1]), an angle (0 to
// 180 degrees), the slope of a profile (beta), or a width above 0 (arcsec:
// from a fine seeing to a radio beam; km/s); none for others, or a value that is not a number (e.g. a table)
export function sliderOf(name: string, property: Schema, defs: Defs, value: unknown): SliderSpec | null {
  if (value !== undefined && typeof value !== 'number') return null;
  const resolved = resolve(property, defs);
  const branches = ((resolved.anyOf as Schema[] | undefined) ?? [resolved]).map((b) => resolve(b, defs));
  const number = branches.find((b) => b.type === 'number');
  if (number === undefined) return null;
  const unit = resolved.unit;
  const low = (number.exclusiveMinimum ?? number.minimum) as number | undefined;
  const high = (number.exclusiveMaximum ?? number.maximum) as number | undefined;
  if (low !== undefined && high !== undefined) {
    const step = niceBelow((high - low) / 100);
    return {
      min: number.exclusiveMinimum !== undefined ? low + step : low,
      max: number.exclusiveMaximum !== undefined ? high - step : high,
      step,
    };
  }
  if (unit === 'deg') return { min: 0, max: 180, step: 1 };
  if (name === 'beta') return low === 1 ? { min: 1.05, max: 10, step: 0.05 } : { min: 0.2, max: 8, step: 0.05 };
  if (low !== 0) return null;
  if (unit === 'arcsec') return { min: 0.01, max: 100, step: LOG_STEP, log: true };
  if (unit === 'km/s') return { min: 0.1, max: 1000, step: LOG_STEP, log: true };
  return null;
}

// (the step of a logarithmic slider, of the logarithm of the value: about
// 2% of the value; 400 steps over the 4 decades of its range)
const LOG_STEP = 0.01;

// The round number (1, 2 or 5 times a power of 10) at or below a positive
// number
function niceBelow(x: number): number {
  const power = 10 ** Math.floor(Math.log10(x));
  return ([5, 2, 1].find((m) => m * power <= x * (1 + 1e-9)) ?? 1) * power;
}

// A number, as typed (blank: none, its default shown); out of its range,
// marked; with a slider beside it, if it has one (see sliderOf)
function NumberValue({
  node,
  value,
  onChange,
  slider,
}: {
  node: Schema;
  value: number | undefined | null;
  onChange: (value: unknown) => void;
  slider?: SliderSpec | null;
}) {
  const [text, setText] = useState(value == null ? '' : String(value));
  // (a value changed elsewhere, e.g. the type, is shown)
  useEffect(() => {
    if ((value ?? null) !== (text.trim() === '' ? null : Number(text))) setText(value == null ? '' : String(value));
  }, [value]);
  const at = typeof value === 'number' ? value : typeof node.default === 'number' ? node.default : null;
  const low = (node.exclusiveMinimum ?? node.minimum) as number | undefined;
  const high = (node.exclusiveMaximum ?? node.maximum) as number | undefined;
  const n = text.trim() === '' ? null : Number(text);
  const out =
    n !== null &&
    ((typeof node.exclusiveMinimum === 'number' && n <= node.exclusiveMinimum) ||
      (typeof node.minimum === 'number' && n < node.minimum) ||
      (typeof node.exclusiveMaximum === 'number' && n >= node.exclusiveMaximum) ||
      (typeof node.maximum === 'number' && n > node.maximum));
  const range = low !== undefined || high !== undefined ? `${node.exclusiveMinimum !== undefined ? '(' : '['}${low ?? '−∞'}, ${high ?? '∞'}${node.exclusiveMaximum !== undefined ? ')' : ']'}` : undefined;
  const typed = (next: string) => {
    setText(next);
    if (next.trim() === '') onChange(undefined);
    else if (Number.isFinite(Number(next))) onChange(Number(next));
  };
  const input = (
    <NumberInput
      value={text}
      onChange={(next) => typed(String(next))}
      hideControls
      placeholder={node.default != null ? String(node.default) : undefined}
      aria-label="a number"
      title={range && `In ${range}`}
      error={out ? `out of ${range}` : undefined}
      size="xs"
      className={slider ? 'ui-number schema-number compact' : 'ui-number schema-number'}
    />
  );
  if (!slider) return input;
  // (a logarithmic slider moves along the logarithm of the value, and gives
  // it to 3 significant digits; a value beyond its ends puts it at one. As
  // it is dragged, the value has as many decimals as its step, or, of a
  // logarithmic one, as its order of magnitude has, so that its digits
  // stay still)
  const log = slider.log === true;
  const min = log ? Math.log10(slider.min) : slider.min;
  const max = log ? Math.log10(slider.max) : slider.max;
  const decimals = (step: number) => Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  const valueOf = (x: number) => {
    if (!log) return x.toFixed(decimals(slider.step));
    const v = Number((10 ** x).toPrecision(3));
    return v.toFixed(Math.max(0, 2 - Math.floor(Math.log10(v) + 1e-9)));
  };
  const position = at === null || (log && !(at > 0)) ? min : log ? Math.log10(at) : at;
  return (
    <div className="schema-slide">
      <Slider
        min={min}
        max={max}
        step={slider.step}
        value={Math.min(max, Math.max(min, position))}
        onChange={(x) => typed(valueOf(x))}
        label={valueOf}
        thumbLabel="a number"
        size="sm"
        className="ui-slider schema-slider"
      />
      {input}
    </div>
  );
}

// Numbers, typed with commas between them (blank: none)
function NumbersField({ value, onChange, placeholder }: { value: number[] | undefined | null; onChange: (value: unknown) => void; placeholder?: string }) {
  const [text, setText] = useState(value ? value.join(', ') : '');
  useEffect(() => {
    if ((value ?? []).join(',') !== parseNumbers(text)?.join(',')) setText(value ? value.join(', ') : '');
  }, [value]);
  return (
    <TextField
      mono
      value={text}
      placeholder={placeholder ?? 'e.g. 1, 1'}
      label="numbers, with commas between them"
      onChange={(typed) => {
        setText(typed);
        const numbers = parseNumbers(typed);
        if (typed.trim() === '') onChange(undefined);
        else if (numbers) onChange(numbers);
      }}
    />
  );
}

function parseNumbers(text: string): number[] | null {
  const parts = text.split(/[,\s]+/).filter(Boolean);
  const numbers = parts.map(Number);
  return parts.length > 0 && numbers.every(Number.isFinite) ? numbers : null;
}

// A list of members of a family (e.g. the PSFs of a sum): a form each,
// which can go, and a button for another
function ListField({ node, value, onChange }: { node: Schema; value: unknown; onChange: (value: unknown) => void }) {
  const { starts } = useForm();
  const family = familyOf(node) ?? '';
  const items = Array.isArray(value) ? value : [];
  const first = Object.entries(starts[family] ?? {})[0];
  return (
    <div className="schema-list">
      {items.map((item, i) => (
        <div key={i} className="schema-item">
          <div className="schema-item-head">
            <span className="eyebrow">
              {family} {i + 1}
            </span>
            <Close label={`Remove ${family} ${i + 1}`} onClick={() => onChange(items.filter((_, j) => j !== i))} />
          </div>
          <Field node={node} family={family} value={item} onChange={(next) => onChange(items.map((old, j) => (j === i ? next : old)))} />
        </div>
      ))}
      <Button kind="small" onClick={() => onChange([...items, first ? { type: first[0], ...first[1] } : {}])}>
        + Add a {family}
      </Button>
    </div>
  );
}

// A file of the project (relative to it): imported or uploaded, and its
// array, if it has several (by name, or by index)
function FileField({ value, onChange }: { value: unknown; onChange: (value: unknown) => void }) {
  const { project } = useForm();
  const { file, hdu } = (typeof value === 'string' ? { file: value, hdu: undefined } : isObject(value) ? value : { file: undefined, hdu: undefined }) as {
    file?: unknown;
    hdu?: unknown;
  };
  const path = typeof file === 'string' ? file : null;
  const arrays = useDescription(path === null ? null : joinPath(project, path)).data ?? [];
  const prefix = project === '.' ? '' : `${project}/`;
  const set = (nextFile: string | null, nextHdu: unknown) =>
    onChange(nextFile === null ? undefined : nextHdu === undefined ? nextFile : { file: nextFile, hdu: nextHdu });
  if (path === null) {
    return (
      <div className="slot empty">
        <FileSources project={project} onFile={(chosen) => set(chosen.startsWith(prefix) ? chosen.slice(prefix.length) : chosen, undefined)} />
      </div>
    );
  }
  const chosen = arrays.find((a) => a.name === hdu || a.index === hdu) ?? arrays[0];
  return (
    <div className="slot">
      <span className="slot-file" title={path}>
        {path.split('/').pop()}
      </span>
      {arrays.length > 1 && (
        <Choice
          items={arrays.map((a) => ({ value: String(a.index), label: `${a.name || `array ${a.index}`} · ${a.shape.filter((n) => n > 1).join(' × ')}` }))}
          value={String(chosen?.index ?? '')}
          label="The array of the file"
          onChange={(index) => {
            const array = arrays.find((a) => a.index === Number(index));
            if (array) set(path, array.name && arrays.filter((a) => a.name === array.name).length === 1 ? array.name : array.index);
          }}
        />
      )}
      <Close label="Remove the file" onClick={() => set(null, undefined)} />
    </div>
  );
}

// A value the form cannot edit (e.g. a table along the spectral axis):
// shown as it is, and kept; it can go
function Kept({ value, onChange }: { value: unknown; onChange: (value: unknown) => void }) {
  if (value === undefined) return <span className="note">Edit it in the file</span>;
  return (
    <span className="schema-kept" title="The Studio cannot edit this yet; it is kept as it is">
      <code>{JSON.stringify(value)}</code>
      <Close label="Remove it" onClick={() => onChange(undefined)} />
    </span>
  );
}
