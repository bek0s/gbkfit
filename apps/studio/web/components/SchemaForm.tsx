// A form of options made from the JSON Schema of the configurations of
// GBKFIT (see the server's /schema, written from the signatures of the
// library): a field for each option, of its type, range, unit, default
// and description. A family (e.g. a PSF: one of its members, chosen by
// its option type) is a choice of the member and the fields of its
// options; a list of a family (e.g. the PSFs of a sum) is a list of such
// forms. A file is one of the project (imported or uploaded), and its
// array, if it has several. What the form cannot edit (e.g. a table along
// the spectral axis) is shown as it is, and kept.
import { NumberInput } from '@mantine/core';
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { useDescription } from '../api/queries';
import { FileSources } from '../screens/items';
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
}

const Context = createContext<FormContext | null>(null);

function useForm(): FormContext {
  const form = useContext(Context);
  if (form === null) throw new Error('a field of a schema is used out of a SchemaForm');
  return form;
}

// The form of a definition of the schema (e.g. 'psf'), of its options
export function SchemaForm({
  defs,
  def,
  project,
  value,
  onChange,
  starts = {},
}: {
  defs: Defs;
  def: string;
  project: string;
  value: Options;
  onChange: (value: Options) => void;
  starts?: Record<string, Record<string, Options>>;
}) {
  const node = defs[def];
  if (node === undefined) return <p className="error">The schema has no {def}.</p>;
  return (
    <Context.Provider value={{ defs, project, starts }}>
      <div className="schema-form">
        <Field node={node} family={def} value={value} onChange={(next) => onChange((next ?? {}) as Options)} />
      </div>
    </Context.Provider>
  );
}

// The members of a family: the objects of its choice (oneOf, nested),
// each of a type (its const), in the order of the schema
export function membersOf(node: Schema, defs: Defs): Schema[] {
  const resolved = resolve(node, defs);
  const choices = resolved.oneOf as Schema[] | undefined;
  if (!choices) return typeOf(resolved) === undefined ? [] : [resolved];
  return choices.flatMap((choice) => membersOf(choice, defs));
}

// The type of a member of a family (the const of its option type)
function typeOf(node: Schema): string | undefined {
  const type = (node.properties as Record<string, Schema> | undefined)?.type;
  return typeof type?.const === 'string' ? type.const : undefined;
}

function resolve(node: Schema, defs: Defs): Schema {
  const ref = node.$ref;
  if (typeof ref === 'string' && ref.startsWith('#/$defs/')) return resolve(defs[ref.slice(8)] ?? {}, defs);
  return node;
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
}: {
  node: Schema;
  value: unknown;
  onChange: (value: unknown) => void;
  // The family it is (e.g. 'psf'), for the options its members start with
  family?: string;
}) {
  const { defs } = useForm();
  const resolved = resolve(node, defs);
  const name = family ?? familyOf(node);
  if (membersOf(resolved, defs).length > 0 && resolved.oneOf) {
    return <MemberField members={membersOf(resolved, defs)} family={name ?? ''} value={value} onChange={onChange} />;
  }
  const branches = resolved.anyOf as Schema[] | undefined;
  if (branches) return <AnyField branches={branches.map((b) => resolve(b, defs))} node={resolved} value={value} onChange={onChange} />;
  return <PlainField node={resolved} value={value} onChange={onChange} />;
}

// A member of a family: the choice of its type, and the fields of its
// options; a new type keeps the options of the old that it has too
function MemberField({
  members,
  family,
  value,
  onChange,
}: {
  members: Schema[];
  family: string;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const { starts } = useForm();
  const options = isObject(value) ? value : {};
  const type = typeof options.type === 'string' ? options.type : undefined;
  // (a type of several members, e.g. of options or of a file: that whose
  // options it has, or the first)
  const ofType = members.filter((m) => typeOf(m) === type);
  const member =
    ofType.find((m) => Object.keys(options).every((key) => key === 'type' || key in propertiesOf(m))) ?? ofType[0];
  const types = [...new Set(members.map((m) => typeOf(m)!))];
  const choose = (next: string) => {
    const start = starts[family]?.[next] ?? {};
    const target = members.find((m) => typeOf(m) === next);
    const kept = Object.fromEntries(Object.entries(options).filter(([key]) => key !== 'type' && target && key in propertiesOf(target)));
    onChange({ type: next, ...start, ...kept });
  };
  return (
    <div className="schema-member">
      <label className="schema-row">
        <span className="schema-label">type</span>
        <Choice
          items={[
            ...(type === undefined || !types.includes(type) ? [{ value: type ?? '', label: type ?? 'choose…' }] : []),
            ...types.map((t) => ({ value: t, label: typeTitle(t) })),
          ]}
          value={type ?? ''}
          label={`The type of the ${family}`}
          onChange={choose}
        />
      </label>
      {ofType.length > 1 && member && (
        <p className="note schema-note">
          Also given as: {ofType.filter((m) => m !== member).map((m) => Object.keys(propertiesOf(m)).filter((k) => k !== 'type').join(', ')).join('; ')}
        </p>
      )}
      {member && <ObjectFields node={member} value={options} onChange={onChange} />}
    </div>
  );
}

// The titles of the types of the families of instruments (others: their
// names)
const TYPE_TITLES: Record<string, string> = {
  point: 'Point (none)',
  gauss: 'Gaussian',
  ggauss: 'Generalised Gaussian',
  moffat: 'Moffat',
  lorentz: 'Lorentzian',
  hanning: 'Hanning',
  gauss_beam: 'Gaussian beam (FWHM)',
  image: 'Image',
  images: 'Images, along the spectrum',
  sum: 'Sum',
  convolution: 'Convolution',
};

export function typeTitle(type: string): string {
  return TYPE_TITLES[type] ?? type;
}

function propertiesOf(node: Schema): Record<string, Schema> {
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
  return (
    <>
      {properties.map(([key, property]) => (
        <Row key={key} name={key} node={property} required={required.includes(key)}>
          <Field node={property} value={value[key]} onChange={(next) => set(key, next)} />
        </Row>
      ))}
      {unknown.map((key) => (
        <Row key={key} name={key} node={{ description: 'An option this type does not take: the configuration is invalid with it.' }} required={false}>
          <span className="error">{JSON.stringify(value[key])}</span>
          <Close label={`Remove ${key}`} onClick={() => set(key, undefined)} />
        </Row>
      ))}
    </>
  );
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
        {unit && <span className="schema-unit">{unit}</span>}
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
}: {
  branches: Schema[];
  node: Schema;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const kinds = branches.filter((b) => b.type !== 'null');
  const nullable = kinds.length < branches.length;
  // (a file: a path, or a path and its HDU)
  if (kinds.some((b) => b.format === 'path')) return <FileField value={value} onChange={onChange} />;
  const number = kinds.find((b) => b.type === 'number' || b.type === 'integer');
  if (number && (value === undefined || value === null || typeof value === 'number')) {
    return <NumberValue node={{ ...number, default: node.default }} value={value as number | undefined} onChange={onChange} />;
  }
  const plain = kinds.find((b) => b.type === 'array' && resolve((b.items as Schema) ?? {}, {}).type === 'number');
  if (plain && (value === undefined || value === null || Array.isArray(value))) {
    return <NumbersField value={value as number[] | undefined} onChange={onChange} placeholder={nullable ? 'from the header' : undefined} />;
  }
  if (kinds.length === 1) return <Field node={kinds[0]} value={value} onChange={onChange} />;
  return <Kept value={value} onChange={onChange} />;
}

// The field of a node of one type
function PlainField({ node, value, onChange }: { node: Schema; value: unknown; onChange: (value: unknown) => void }) {
  const { defs } = useForm();
  if (node.type === 'number' || node.type === 'integer') {
    return typeof value === 'number' || value === undefined ? (
      <NumberValue node={node} value={value} onChange={onChange} />
    ) : (
      <Kept value={value} onChange={onChange} />
    );
  }
  if (node.type === 'boolean') {
    return (
      <Choice
        items={[
          { value: '', label: node.default === undefined ? '—' : `default (${String(node.default)})` },
          { value: 'true', label: 'yes' },
          { value: 'false', label: 'no' },
        ]}
        value={value === undefined ? '' : String(value)}
        label="yes or no"
        onChange={(next) => onChange(next === '' ? undefined : next === 'true')}
      />
    );
  }
  if (node.type === 'string' && Array.isArray(node.enum)) {
    return (
      <Choice
        items={[{ value: '', label: node.default === undefined ? '—' : `default (${String(node.default)})` }, ...(node.enum as string[]).map((e) => ({ value: e, label: e }))]}
        value={typeof value === 'string' ? value : ''}
        label="a choice"
        onChange={(next) => onChange(next === '' ? undefined : next)}
      />
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

// A number, as typed (blank: none, its default shown); out of its range,
// marked
function NumberValue({ node, value, onChange }: { node: Schema; value: number | undefined | null; onChange: (value: unknown) => void }) {
  const [text, setText] = useState(value == null ? '' : String(value));
  // (a value changed elsewhere, e.g. the type, is shown)
  useEffect(() => {
    if ((value ?? null) !== (text.trim() === '' ? null : Number(text))) setText(value == null ? '' : String(value));
  }, [value]);
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
  return (
    <NumberInput
      value={text}
      onChange={(next) => {
        const typed = String(next);
        setText(typed);
        if (typed.trim() === '') onChange(undefined);
        else if (Number.isFinite(Number(typed))) onChange(Number(typed));
      }}
      hideControls
      placeholder={node.default != null ? String(node.default) : undefined}
      aria-label="a number"
      title={range && `In ${range}`}
      error={out ? `out of ${range}` : undefined}
      size="xs"
      className="ui-number schema-number"
    />
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
  if (value === undefined) return <span className="note">(edit in the file)</span>;
  return (
    <span className="schema-kept" title="The Studio cannot edit this yet; it is kept as it is">
      <code>{JSON.stringify(value)}</code>
      <Close label="Remove it" onClick={() => onChange(undefined)} />
    </span>
  );
}
