// The options of an observation as rows of its sheet (see ui/rows), each
// by its schema (that of the library): a choice in a menu, a switch, a
// number (typed; set once it is a valid one, as the schema bounds it), a
// pair of numbers of the sky (x and y side by side, linked or not), or a
// few values (a box each, e.g. one for each axis; see AxesInput), a list
// of numbers (typed, e.g. '2 × 2 × 1'), a quantity (its number in the
// box, its unit a menu beside it), a text, a choice or a value of its own
// (e.g. an extent that spans the galaxies, or a number), a group of
// options (e.g. the depth axis: its rows, indented), a typed item (its
// type a menu, its options rows under it) or a list of them, or, of any
// other form (e.g. a file, regions, a table), a form of the schema across
// the row. Each row is the home of the problems of its path (see
// Problems), its title the library's, and what it is in its tooltip.
import { useEffect, useState, type ReactNode } from 'react';
import { ProblemField } from '../../components/Problems';
import { choicesOf, membersOf, propertiesOf, resolve, SchemaForm, typeOf, type Defs, type Options, type Schema } from '../../components/SchemaForm';
import { splitQuantity } from '../../components/ui/inputs';
import { DropMenu } from '../../components/ui/overlays';
import { ChoiceBox, NumberInput, Row } from '../../components/ui/rows';
import { unitText } from '../../components/units';
import { firstSentence, retyped, typesOf } from '../galaxy/schema';
import { formsOf, LINKED_PAIRS } from './model';

// What the rows of options share: the schema, the project (the files of a
// form are its own), its blocks by folder and how one opens
export interface RowsContext {
  defs: Defs;
  project: string;
  blocks: Record<string, string[]>;
  openBlock: (folder: string, name: string) => void;
  // The units a quantity starts in while none is given, before the first
  // it takes (e.g. that of the channels of the observation, km/s)
  units?: string[];
}

// The form of an option, as its row shows it
type Shape = 'choice' | 'switch' | 'integer' | 'number' | 'pair' | 'values' | 'numbers' | 'names' | 'quantity' | 'text' | 'choiceValue' | 'record' | 'typed' | 'typedList' | 'form';

// The alternatives of a node of the schema, but none (null), its
// references followed, nested choices flattened
function alternatives(node: Schema, defs: Defs): Schema[] {
  const resolved = resolve(node, defs);
  const any = (resolved.anyOf ?? null) as Schema[] | null;
  if (any) return any.flatMap((each) => alternatives(each, defs));
  return resolved.type === 'null' ? [] : [resolved];
}

// The alternatives of a node as written, but none (null): its references
// kept (e.g. to a family, whose name a typed item needs)
function written(node: Schema): Schema[] {
  const any = (node.anyOf ?? null) as Schema[] | null;
  if (any) return any.flatMap(written);
  return node.type === 'null' ? [] : [node];
}

// The family of the schema a node refers to (a choice of types, each of
// its options: e.g. a weight function), or of the items of a list of
// them; else null
function familyOf(node: Schema, defs: Defs): string | null {
  const target = node.type === 'array' && isObject(node.items) ? (node.items as Schema) : node;
  if (typeof target.$ref !== 'string') return null;
  const name = target.$ref.replace('#/$defs/', '');
  return defs[name] && membersOf(defs[name], defs).some((m) => typeOf(m) !== undefined) ? name : null;
}

// The node of an option without its null (that of its one alternative,
// else a choice of them)
export function bareNode(node: Schema, defs: Defs): Schema {
  const all = alternatives(node, defs);
  return all.length === 1 ? { ...all[0], ...pick(node, ['title', 'description', 'unit', 'default']) } : { ...node, anyOf: all };
}

// (a number, or of types of which one is a number: e.g. a level that can be
// a map, a number while it is one)
const isNumeric = (node: Schema) => node.type === 'number' || node.type === 'integer' || (Array.isArray(node.type) && node.type.includes('number'));
const isNumbers = (node: Schema) => node.type === 'array' && isObject(node.items) && isNumeric(node.items as Schema);
// (the number of values of a list of numbers of a given length, else
// null)
const lengthOf = (node: Schema) => (typeof node.minItems === 'number' && node.minItems === node.maxItems ? node.minItems : null);

// The shape of an option of its node, as its value is (e.g. a number that
// can be a table along the spectral axis is a number while it is one)
export function shapeOf(node: Schema, defs: Defs, value: unknown): Shape {
  const all = alternatives(node, defs);
  if (all.length === 0) return 'form';
  if (choicesOf(all.length === 1 ? all[0] : { oneOf: all })) return 'choice';
  // (a typed item, or a list of them, of a family: e.g. weight functions)
  const raw = written(node);
  const families = raw.map((n) => familyOf(n, defs));
  if (families.length && families.every((f) => f !== null && f === families[0])) {
    const listed = raw.some((n) => n.type === 'array');
    if (listed && (value === undefined || value === null || isObject(value) || Array.isArray(value))) return 'typedList';
    if (!listed && (value === undefined || value === null || isObject(value))) return 'typed';
  }
  // (a group of options: e.g. the depth axis)
  if (all.length === 1 && all[0].type === 'object' && isObject(all[0].properties) && (value === undefined || value === null || isObject(value))) return 'record';
  // (a choice, or a value of its own: e.g. an extent that spans the
  // galaxies, or a number)
  const choiceNodes = all.filter((n) => choicesOf(n));
  const others = all.filter((n) => !choicesOf(n));
  if (choiceNodes.length === 1 && others.length === 1 && (isNumeric(others[0]) || isNumbers(others[0]))) return 'choiceValue';
  const numeric = all.filter(isNumeric);
  if (numeric.length && all.some(isNumbers) && all.every((n) => isNumeric(n) || isNumbers(n))) {
    const numbers = value === undefined || value === null || typeof value === 'number' || (Array.isArray(value) && value.every((v) => typeof v === 'number'));
    if (!numbers) return 'form';
    // (one value, or a list of one: of a grid of one axis, e.g. along a
    // slit, a number)
    if (all.filter(isNumbers).every((n) => n.maxItems === 1)) return numeric.every((n) => n.type === 'integer') ? 'integer' : 'number';
    return 'pair';
  }
  if (numeric.length) {
    if (value !== undefined && value !== null && typeof value !== 'number') return 'form';
    return numeric.every((n) => n.type === 'integer') ? 'integer' : 'number';
  }
  // (a list of numbers of a few values, e.g. one for each axis: a box
  // each; else typed)
  if (all.length === 1 && isNumbers(all[0])) {
    const count = lengthOf(all[0]);
    return count !== null && count >= 2 && count <= 4 ? 'values' : 'numbers';
  }
  if (all.length === 1 && all[0].type === 'array' && isObject(all[0].items) && (all[0].items as Schema).type === 'string') return 'names';
  if (all.length === 1 && all[0].type === 'boolean') return 'switch';
  // (a text, or a quantity, that can be given otherwise too: e.g. a band
  // by its name or in full, a width that can be a table): as text while
  // it is one
  const texts = all.filter((n) => n.type === 'string');
  if (texts.length === 1 && (value === undefined || value === null || typeof value === 'string')) {
    return texts[0].format === 'quantity' ? 'quantity' : texts[0].format === undefined ? 'text' : 'form';
  }
  return 'form';
}

// Whether a number is one an option takes, as its schema bounds it
function withinBounds(value: number, nodes: Schema[]): boolean {
  if (!Number.isFinite(value)) return false;
  return nodes.some((n) => {
    if (n.type === 'integer' && !Number.isInteger(value)) return false;
    if (typeof n.minimum === 'number' && value < n.minimum) return false;
    if (typeof n.exclusiveMinimum === 'number' && value <= n.exclusiveMinimum) return false;
    if (typeof n.maximum === 'number' && value > n.maximum) return false;
    if (typeof n.exclusiveMaximum === 'number' && value >= n.exclusiveMaximum) return false;
    return true;
  });
}

// The title of an option (the library's, else its key), and what it is
export function optionTitle(node: Schema | undefined, key: string): string {
  return typeof node?.title === 'string' ? node.title : key.replaceAll('_', ' ');
}

export function optionAbout(node: Schema | undefined): string | undefined {
  return typeof node?.description === 'string' ? firstSentence(node.description) : undefined;
}

// An option as a row (see the module): its node, its value (undefined:
// none, its default), what sets it (undefined: removed), the path of its
// problems, and what is said under its title (e.g. what it makes: '684″
// across'; kept as a line while it can be, so that the row keeps its
// height as it comes and goes: hold)
export function OptionRow({
  ctx,
  name,
  node,
  value,
  onChange,
  path,
  title,
  sub,
  hold = false,
  label,
  choices,
  none = 'all',
}: {
  ctx: RowsContext;
  name: string;
  node: Schema;
  value: unknown;
  onChange: (value: unknown) => void;
  path: (string | number)[];
  title?: string;
  sub?: ReactNode;
  hold?: boolean;
  // (its name in the list of problems; by default its title)
  label?: string;
  // (of names: those to choose from, else typed; and what none means)
  choices?: string[];
  none?: string;
}) {
  const { defs } = ctx;
  const shape = shapeOf(node, defs, value);
  const all = alternatives(node, defs);
  const heading = title ?? optionTitle(node, name);
  const about = optionAbout(node);
  const unit = typeof node.unit === 'string' ? node.unit : (all.find((n) => typeof n.unit === 'string')?.unit as string | undefined);
  const shownUnit = unit ? unitText(unit) : '';
  const fallback = node.default ?? all.find((n) => n.default !== undefined)?.default;
  const line = sub ?? (hold ? ' ' : undefined);
  const row = (box: ReactNode, extra: { unit?: ReactNode; wide?: boolean; wideBox?: boolean } = {}) => (
    <ProblemField path={path} label={label ?? heading}>
      <Row title={heading} titleProps={{ title: about }} sub={line} box={box} unit={extra.unit ?? shownUnit} wide={extra.wide} wideBox={extra.wideBox} />
    </ProblemField>
  );
  if (shape === 'choice') {
    const choices = choicesOf(all.length === 1 ? all[0] : { oneOf: all }) ?? [];
    const current = choices.find((c) => c.value === value);
    const byDefault = choices.find((c) => c.value === fallback);
    const text = current ? current.title : value === undefined || value === null ? (byDefault ? byDefault.title : 'by default') : String(value);
    const long = choices.some((c) => c.title.length > 11);
    return row(
      <ChoiceBox
        label={heading}
        text={<span data-default={current ? undefined : true}>{text}</span>}
        title={about}
        width={260}
        entries={[
          ...(value !== undefined && value !== null && !required(node) ? [{ label: byDefault ? `By default (${byDefault.title})` : 'By default', onClick: () => onChange(undefined) }] : []),
          ...choices.map((c) => ({ label: c.title, current: c.value === value, onClick: () => onChange(c.value) })),
        ]}
      />,
      { wideBox: long },
    );
  }
  if (shape === 'switch') {
    const on = value === undefined || value === null ? fallback === true : value === true;
    return row(
      <ChoiceBox
        label={heading}
        text={<span data-default={value === undefined || value === null || undefined}>{on ? 'on' : 'off'}</span>}
        title={about}
        width={160}
        entries={[
          { label: 'Off', current: !on, onClick: () => onChange(fallback === false || fallback === undefined || fallback === null ? undefined : false) },
          { label: 'On', current: on, onClick: () => onChange(fallback === true ? undefined : true) },
        ]}
      />,
    );
  }
  if (shape === 'integer' || shape === 'number') {
    return row(
      <NumberInput
        value={typeof value === 'number' ? value : Array.isArray(value) && typeof value[0] === 'number' ? value[0] : undefined}
        label={heading}
        placeholder={typeof fallback === 'number' ? String(fallback) : 'none'}
        valid={(v) => withinBounds(v, all.filter(isNumeric))}
        onChange={onChange}
        onClear={() => onChange(undefined)}
      />,
    );
  }
  if (shape === 'pair' || shape === 'values' || shape === 'numbers') {
    const items = all.filter(isNumbers).map((n) => n.items as Schema);
    const valid = (v: number) => withinBounds(v, [...all.filter(isNumeric), ...items]);
    if (shape === 'pair') return row(<AxesInput value={value} fallback={fallback} count={2} label={heading} linkable={LINKED_PAIRS.includes(name)} valid={valid} onChange={onChange} />);
    if (shape === 'values') return row(<AxesInput value={value} fallback={fallback} count={lengthOf(all[0]) ?? 2} label={heading} linkable={false} valid={valid} onChange={onChange} />);
    // (a list of long numbers, across the unit too)
    return row(<ListInput value={value} label={heading} valid={valid} onChange={onChange} />, { wideBox: pairText(value).length > 11 });
  }
  if (shape === 'names') {
    const names = Array.isArray(value) ? value.map(String) : [];
    const text = names.length ? names.join(', ') : none;
    if (choices) {
      const toggled = (name: string) => (names.includes(name) ? names.filter((n) => n !== name) : [...names, name]);
      return row(
        <ChoiceBox
          label={heading}
          text={<span data-default={names.length ? undefined : true}>{text}</span>}
          title={about}
          width={260}
          entries={[
            { label: `All (${none})`, current: names.length === 0, onClick: () => onChange(undefined) },
            ...[...new Set([...choices, ...names])].map((name) => ({ label: name, current: names.includes(name), onClick: () => onChange(toggled(name).length ? toggled(name) : undefined) })),
          ]}
        />,
        { wideBox: true },
      );
    }
    return row(<TextBox value={names.join(', ')} label={heading} placeholder={none} onChange={(typed) => onChange(namesOf(typed))} />, { wideBox: true });
  }
  if (shape === 'quantity') {
    const kinds = all.find((n) => isObject(n.unit_kinds))?.unit_kinds as Record<string, string[]> | undefined;
    const [number, valueUnit] = typeof value === 'string' ? splitQuantity(value) : ['', ''];
    const units = Object.values(kinds ?? {}).flat();
    const squash = (u: string) => u.replace(/\s+/g, '');
    const preferred = units.find((u) => (ctx.units ?? []).some((p) => squash(p) === squash(u)));
    const chosen = valueUnit || preferred || units[0] || '';
    return row(
      <NumberInput
        value={number === '' ? undefined : Number(number)}
        label={heading}
        placeholder="none"
        valid={Number.isFinite}
        onChange={(v) => onChange(`${v} ${chosen}`.trim())}
        onClear={() => onChange(undefined)}
      />,
      {
        unit: kinds ? (
          <DropMenu
            label={`The unit of ${heading.toLowerCase()}`}
            width={200}
            entries={Object.entries(kinds).flatMap(([kind, list]) => [{ section: kind }, ...list.map((u) => ({ label: unitText(u), current: u === chosen, onClick: () => onChange(number === '' ? undefined : `${number} ${u}`) }))])}
            target={
              <button type="button" className="ob-unit-pick" title="Its unit">
                {unitText(chosen)}
              </button>
            }
          />
        ) : (
          chosen
        ),
      },
    );
  }
  if (shape === 'choiceValue') {
    return <ChoiceValueRow name={name} node={node} all={all} heading={heading} about={about} line={line} label={label ?? heading} path={path} value={value} fallback={fallback} onChange={onChange} />;
  }
  if (shape === 'record') {
    const record = all[0];
    return (
      <ProblemField path={path} label={label ?? heading}>
        <GroupHead heading={heading} about={about} line={line} />
        <div className="ob-group">
          <OptionRows ctx={ctx} node={record} keys={Object.keys(propertiesOf(record))} value={isObject(value) ? value : {}} path={path} onChange={(next) => onChange(Object.keys(next).length ? next : undefined)} />
        </div>
      </ProblemField>
    );
  }
  if (shape === 'typed') {
    return <TypedRows ctx={ctx} family={familyOf(written(node)[0], defs)!} heading={heading} about={about} line={line} label={label ?? heading} path={path} value={value} optional={!required(node)} onChange={onChange} />;
  }
  if (shape === 'typedList') {
    const family = familyOf(written(node)[0], defs)!;
    const items = Array.isArray(value) ? value : isObject(value) ? [value] : [];
    const types = typesOf(defs, family);
    const set = (next: unknown[]) => onChange(next.length ? next : undefined);
    return (
      <ProblemField path={path} label={label ?? heading}>
        <GroupHead heading={heading} about={about} line={line} />
        <div className="ob-group">
          {items.map((item, i) => (
            <TypedRows
              key={i}
              ctx={ctx}
              family={family}
              heading={`${i + 1}`}
              label={`${heading} ${i + 1}`}
              path={[...path, i]}
              value={item}
              optional={false}
              onChange={(next) => set(items.map((x, k) => (k === i ? next : x)))}
              onRemove={() => set(items.filter((_, k) => k !== i))}
            />
          ))}
          {types.length > 0 && (
            <button type="button" className="ob-add" onClick={() => set([...items, retyped(defs, family, {}, types[0].value)])}>
              + Add one
            </button>
          )}
        </div>
      </ProblemField>
    );
  }
  if (shape === 'text') {
    return row(<TextBox value={typeof value === 'string' ? value : ''} label={heading} placeholder={typeof fallback === 'string' ? fallback : 'none'} onChange={(text) => onChange(text === '' ? undefined : text)} />, { wideBox: true });
  }
  // (any other form: the schema's own, across the row)
  return (
    <ProblemField path={path} label={label ?? heading}>
      <div className="ob-form-row">
        <Row title={heading} titleProps={{ title: about }} sub={line} />
        <div className="gm-form">
          {/* (the form of the option itself, its title the row's) */}
          <SchemaForm defs={defs} node={node} project={ctx.project} value={value as Options} onChange={onChange} blocks={ctx.blocks} openBlock={ctx.openBlock} />
        </div>
      </div>
    </ProblemField>
  );
}

// The rows of options of an object (e.g. the PSF of an instrument): each
// of its keys given, of its schema
export function OptionRows({
  ctx,
  node,
  keys,
  value,
  onChange,
  path,
  subs = {},
}: {
  ctx: RowsContext;
  // The schema of the object (its properties)
  node: Schema;
  keys: string[];
  value: Options;
  onChange: (value: Options) => void;
  path: (string | number)[];
  subs?: Record<string, ReactNode>;
}) {
  const properties = (resolve(node, ctx.defs).properties ?? {}) as Record<string, Schema>;
  return (
    <>
      {keys
        .filter((key) => key in properties)
        .map((key) => (
          <OptionRow
            key={key}
            ctx={ctx}
            name={key}
            node={properties[key]}
            value={value[key]}
            path={[...path, key]}
            sub={subs[key]}
            hold={key in subs}
            onChange={(next) => {
              const { [key]: _, ...rest } = value;
              onChange(next === undefined ? rest : { ...rest, [key]: next });
            }}
          />
        ))}
    </>
  );
}

// The values of an option of one for each of a few axes, side by side: a
// pair of the sky (x and y; e.g. its pixels: one number stands for both),
// or a list of a given length (count; e.g. an oversampling for each axis
// of a cube). Between the two of a pair that can be (linkable), the link
// that ties them: linked, one typed sets both, and a click sets them
// apart; apart, each is its own, and a click links them (y takes x). It
// opens linked if they are equal or none is given, and is set apart when
// they come to differ (e.g. copied from a dataset); the link is not
// saved. A value typed while none is given is that of every box (shown
// at once); either box cleared clears the option
function AxesInput({ value, fallback, count, label, linkable, valid, onChange }: { value: unknown; fallback?: unknown; count: number; label: string; linkable: boolean; valid: (value: number) => boolean; onChange: (value: unknown) => void }) {
  const values = Array.isArray(value) ? value : Array.from({ length: count }, () => value);
  const at = (k: number) => (typeof values[k] === 'number' ? (values[k] as number) : undefined);
  // (the default of each box, if the option has one: e.g. 1)
  const defaults = Array.isArray(fallback) ? fallback : Array.from({ length: count }, () => fallback);
  const byDefault = (k: number) => (typeof defaults[k] === 'number' ? (defaults[k] as number) : undefined);
  const equal = at(0) === at(1);
  const [linked, setLinked] = useState(equal);
  useEffect(() => {
    if (!equal) setLinked(false);
  }, [equal]);
  const tied = linkable && count === 2 && linked;
  const set = (k: number, v: number) => onChange(Array.from({ length: count }, (_, i) => (i === k || tied ? v : (at(i) ?? byDefault(i) ?? v))));
  const link = () => {
    const x = at(0);
    setLinked(!linked);
    if (!linked && x !== undefined && at(1) !== x) onChange([x, x]);
  };
  const axis = (k: number) => (count === 2 ? ['x', 'y'][k] : `axis ${k + 1}`);
  const box = (k: number) => <NumberInput key={k} value={at(k)} label={`${label}: ${axis(k)}`} placeholder={byDefault(k) !== undefined ? String(byDefault(k)) : 'none'} valid={valid} onChange={(v) => set(k, v)} onClear={() => onChange(undefined)} />;
  if (count !== 2) {
    return (
      <span className="sh-pair" data-count={count}>
        {Array.from({ length: count }, (_, k) => box(k))}
      </span>
    );
  }
  return (
    <span className="sh-pair">
      {box(0)}
      {linkable ? (
        <button
          type="button"
          className="sh-link"
          aria-pressed={tied}
          aria-label={tied ? `Set the x and y of ${label.toLowerCase()} apart` : `Link the x and y of ${label.toLowerCase()}`}
          title={tied ? 'Linked: one value for x and y. Click to set them apart' : 'Apart: x and y each its own. Click to link them (y takes x)'}
          onClick={link}
        >
          {tied ? (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M6.6 9.4l2.8-2.8" />
              <path d="M7.3 4.6l1.1-1.1a2.7 2.7 0 013.8 3.8l-1.1 1.1" />
              <path d="M8.7 11.4l-1.1 1.1a2.7 2.7 0 01-3.8-3.8l1.1-1.1" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M7.3 4.6l1.1-1.1a2.7 2.7 0 013.8 3.8l-1.1 1.1" />
              <path d="M8.7 11.4l-1.1 1.1a2.7 2.7 0 01-3.8-3.8l1.1-1.1" />
              <path d="M5.2 2.6v1.6M2.6 5.2h1.6M10.8 13.4v-1.6M13.4 10.8h-1.6" />
            </svg>
          )}
        </button>
      ) : (
        <span />
      )}
      {box(1)}
    </span>
  );
}

// The head of a group of options (see OptionRow: a group, a list of typed
// items): its title, and what is said under it
function GroupHead({ heading, about, line }: { heading: string; about?: string; line?: ReactNode }) {
  return (
    <div className="ob-group-head sh-title" title={about}>
      <b>{heading}</b>
      {line && <small>{line}</small>}
    </div>
  );
}

// An option of a choice or a value of its own (e.g. an extent that spans
// the galaxies, or a number): a menu of the choices and "A value of its
// own" while it is a choice; its value in the box, the same menu beside
// it (in the column of units), while it is a value
function ChoiceValueRow({
  name,
  node,
  all,
  heading,
  about,
  line,
  label,
  path,
  value,
  fallback,
  onChange,
}: {
  name: string;
  node: Schema;
  all: Schema[];
  heading: string;
  about?: string;
  line?: ReactNode;
  label: string;
  path: (string | number)[];
  value: unknown;
  fallback: unknown;
  onChange: (value: unknown) => void;
}) {
  const choices = choicesOf(all.find((n) => choicesOf(n))!) ?? [];
  const valueNode = all.find((n) => !choicesOf(n))!;
  const isValue = typeof value === 'number' || Array.isArray(value);
  // (a value of its own asked for, before one is typed)
  const [own, setOwn] = useState(false);
  useEffect(() => {
    if (!isValue && value !== undefined && value !== null) setOwn(false);
  }, [isValue, value]);
  const valued = isValue || own;
  const current = choices.find((c) => c.value === (value ?? fallback));
  const entries = [
    ...choices.map((c) => ({
      label: c.title,
      current: !valued && c === current,
      onClick: () => {
        setOwn(false);
        onChange(c.value === fallback ? undefined : c.value);
      },
    })),
    { label: 'A value of its own', current: valued, onClick: () => setOwn(true) },
  ];
  const unit = typeof node.unit === 'string' ? unitText(node.unit) : '';
  const valid = (v: number) => withinBounds(v, [valueNode, ...(isObject(valueNode.items) ? [valueNode.items as Schema] : [])]);
  const count = lengthOf(valueNode);
  const box = !valued ? (
    <ChoiceBox label={heading} text={<span data-default={value === undefined || value === null || undefined}>{current?.title ?? 'by default'}</span>} title={about} width={260} entries={entries} />
  ) : isNumeric(valueNode) ? (
    <NumberInput value={typeof value === 'number' ? value : undefined} label={heading} placeholder="none" valid={valid} onChange={onChange} onClear={() => onChange(undefined)} />
  ) : count !== null && count >= 2 && count <= 4 ? (
    <AxesInput value={isValue ? value : undefined} count={count} label={heading} linkable={false} valid={valid} onChange={onChange} />
  ) : (
    <ListInput value={isValue ? value : undefined} label={heading} valid={valid} onChange={onChange} />
  );
  const menu = (
    <DropMenu
      label={`Its ${heading.toLowerCase()}: a choice, or a value of its own`}
      width={260}
      entries={entries}
      target={
        <button type="button" className="ob-unit-pick" title="A choice, or a value of its own">
          {unit || '▾'}
        </button>
      }
    />
  );
  return (
    <ProblemField path={path} label={label}>
      <Row title={heading} titleProps={{ title: about }} sub={line} box={box} wideBox={!valued || undefined} unit={valued ? menu : undefined} data-option={name} />
    </ProblemField>
  );
}

// A typed item of a family (e.g. a weight function): its type, a menu (of
// none too, if it is optional), and its options, rows under it; of a
// list, a button that removes it
function TypedRows({
  ctx,
  family,
  heading,
  about,
  line,
  label,
  path,
  value,
  optional,
  onChange,
  onRemove,
}: {
  ctx: RowsContext;
  family: string;
  heading: string;
  about?: string;
  line?: ReactNode;
  label: string;
  path: (string | number)[];
  value: unknown;
  optional: boolean;
  onChange: (value: unknown) => void;
  onRemove?: () => void;
}) {
  const own = isObject(value) ? value : null;
  const types = typesOf(ctx.defs, family);
  const form = own ? formsOf(ctx.defs, family, own).form : undefined;
  const text = own ? (types.find((t) => t.value === own.type)?.label ?? String(own.type ?? '')) : 'none';
  return (
    <ProblemField path={path} label={label}>
      <Row
        title={heading}
        titleProps={{ title: about }}
        sub={line}
        wideBox
        box={
          <ChoiceBox
            label={label}
            text={<span data-default={own ? undefined : true}>{text}</span>}
            title={about}
            width={280}
            entries={[
              ...(optional ? [{ label: 'None', current: !own, onClick: () => onChange(undefined) }] : []),
              ...types.map((t) => ({ label: t.label, description: t.description, current: own?.type === t.value, onClick: () => onChange(retyped(ctx.defs, family, own ?? {}, t.value)) })),
            ]}
          />
        }
        pill={
          onRemove && (
            <button type="button" className="sh-icon" title="Remove it" aria-label={`Remove ${label.toLowerCase()}`} onClick={onRemove}>
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />
              </svg>
            </button>
          )
        }
      />
      {own && form && (
        <div className="ob-group">
          <OptionRows ctx={ctx} node={form} keys={Object.keys(propertiesOf(form)).filter((key) => key !== 'type')} value={own} path={path} onChange={(next) => onChange({ ...next, type: own.type })} />
        </div>
      )}
    </ProblemField>
  );
}

// A list of numbers as typed (e.g. '2 × 2 × 1'): a click selects it, to
// type another over it; set as it is typed once each of its numbers is
// valid
function ListInput({ value, label, valid, onChange }: { value: unknown; label: string; valid: (value: number) => boolean; onChange: (value: unknown) => void }) {
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <input
      className="sh-num"
      aria-label={label}
      value={typing ?? pairText(value)}
      placeholder="none"
      onFocus={(event) => event.currentTarget.select()}
      onChange={(event) => {
        setTyping(event.target.value);
        const text = event.target.value.trim();
        if (text === '') return onChange(undefined);
        const parts = text.split(/\s*(?:[×x*,]|\s)\s*/i).filter(Boolean).map(Number);
        if (parts.length === 0 || !parts.every(valid)) return;
        onChange(parts);
      }}
      onBlur={() => setTyping(null)}
      onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.blur()}
    />
  );
}

// A pair as written: one number, if both are the same (a list, of each)
export function pairText(value: unknown): string {
  const show = (v: unknown) => (typeof v === 'number' ? String(Number(v.toPrecision(6))) : '?');
  if (Array.isArray(value)) return value.length === 2 && value[0] === value[1] ? show(value[0]) : value.map(show).join(' × ');
  return value === undefined || value === null ? '' : show(value);
}

// Names as typed, apart by commas or spaces (none: undefined)
function namesOf(text: string): string[] | undefined {
  const names = text.split(/[\s,]+/).filter(Boolean);
  return names.length ? names : undefined;
}

// A text, as typed: set as it is left (or entered)
function TextBox({ value, label, placeholder, onChange }: { value: string; label: string; placeholder: string; onChange: (text: string) => void }) {
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <input
      className="sh-num ob-text-box"
      aria-label={label}
      value={typing ?? value}
      placeholder={placeholder}
      spellCheck={false}
      onChange={(event) => setTyping(event.target.value)}
      onBlur={() => {
        if (typing !== null && typing.trim() !== value) onChange(typing.trim());
        setTyping(null);
      }}
      onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.blur()}
    />
  );
}

function required(node: Schema): boolean {
  return node.default === undefined && !alternativesHaveNull(node);
}

function alternativesHaveNull(node: Schema): boolean {
  const any = node.anyOf as Schema[] | undefined;
  return Array.isArray(any) && any.some((n) => n.type === 'null');
}

function isObject(value: unknown): value is Options {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function pick(node: Schema, keys: string[]): Schema {
  return Object.fromEntries(keys.filter((key) => node[key] !== undefined).map((key) => [key, node[key]]));
}
