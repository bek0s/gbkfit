// Inputs (of Mantine; see theme.ts), with the conventions of the Studio:
// a number field whose value is its text (blank: none), clamped to its
// range as it is left; a quantity as its text, of a number and a unit; a
// choice of a few (segments) or of a list (a field whose list drops as the
// menus do), or of several of a list (a field of the same look); and a
// check.
import { Autocomplete, Checkbox, Combobox, defaultOptionsFilter, Input, NumberInput, SegmentedControl, Select, TextInput, type ComboboxItem, type ComboboxItemGroup, type OptionsFilter } from '@mantine/core';
import { useRef, useState } from 'react';
import { DropMenu } from './overlays';

// The most choices of a list it shows without a search: a longer one is
// searched, as it is typed into
const SEARCHED = 10;

// One of a list of choices (some can be off, and each can say what it
// is under its label, and be of a group, under its heading: consecutive
// choices of one group are), as a field whose list drops as the menus do
// (see overlays); searched, if long (see SEARCHED). A click on the field
// opens its list (and leaves it open as the field is searched, to place
// the caret); one on its chevron opens or closes it.
// Several of a list of choices, or all of them (none chosen), as a field
// of the look of Choice whose menu toggles each (those chosen marked);
// those chosen that the list has not stay, to be taken out
export function Choices({ items, value, onChange, label, all }: { items: string[]; value: string[]; onChange: (value: string[]) => void; label: string; all: string }) {
  const toggled = (item: string) => (value.includes(item) ? value.filter((v) => v !== item) : [...value, item]);
  const text = value.length ? value.join(', ') : all;
  return (
    <DropMenu
      label={label}
      entries={[
        { label: all.charAt(0).toUpperCase() + all.slice(1), current: value.length === 0, onClick: () => onChange([]) },
        ...[...new Set([...items, ...value])].map((item) => ({ label: item, current: value.includes(item), onClick: () => onChange(toggled(item)) })),
      ]}
      target={
        <Input component="button" type="button" size="xs" pointer className="ui-choice ui-choices" title={text} aria-label={label} rightSection={<Combobox.Chevron size="xs" />} rightSectionPointerEvents="none">
          <span data-default={value.length ? undefined : true}>{text}</span>
        </Input>
      }
    />
  );
}

export function Choice({
  items,
  value,
  onChange,
  label,
  title,
  placeholder,
  disabled = false,
}: {
  items: { value: string; label: string; disabled?: boolean; description?: string; group?: string }[];
  // (none of the items: the placeholder shows)
  value: string;
  onChange: (value: string) => void;
  label: string;
  title?: string;
  placeholder?: string;
  // (none can be chosen yet: e.g. until another field is)
  disabled?: boolean;
}) {
  const described = new Map(items.map((item) => [item.value, item.description]));
  // (the choices of no group, and the groups of the others, in order)
  const data: (ComboboxItem | ComboboxItemGroup<ComboboxItem, string>)[] = [];
  for (const { value, label, disabled, group } of items) {
    const item = { value, label, disabled };
    const last = data.at(-1);
    if (group === undefined) data.push(item);
    else if (last && 'group' in last && last.group === group) last.items.push(item);
    else data.push({ group, items: [item] });
  }
  const [opened, setOpened] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  // (whether it was open as its chevron was pressed: the press, out of
  // the field, closes it, before the click)
  const wasOpen = useRef(false);
  return (
    <Select
      ref={field}
      dropdownOpened={opened}
      onDropdownOpen={() => setOpened(true)}
      onDropdownClose={() => setOpened(false)}
      rightSectionPointerEvents="all"
      rightSectionProps={{
        onMouseDown: (event) => {
          event.preventDefault();
          wasOpen.current = opened;
        },
        // (opened as a click on the field opens it, the one chosen marked;
        // the field keeps the focus, or takes it, for the keys of the list)
        onClick: () => {
          if (disabled) return;
          if (wasOpen.current) setOpened(false);
          else {
            field.current?.focus();
            field.current?.click();
          }
        },
      }}
      data={data}
      searchable={items.length > SEARCHED}
      // (as it is searched, its first match is marked, which Enter takes)
      selectFirstOptionOnChange
      nothingFoundMessage="None matches"
      // (null for a value of none of the items: Mantine would keep showing
      // the item chosen before)
      value={items.some((item) => item.value === value) ? value : null}
      onChange={(next) => next !== null && onChange(next)}
      allowDeselect={false}
      checkIconPosition="right"
      maxDropdownHeight={340}
      comboboxProps={{ offset: 4, shadow: 'md' }}
      renderOption={({ option }) => {
        const description = described.get(option.value);
        return description ? (
          <div className="ui-choice-option">
            <span>{option.label}</span>
            <small title={description}>{description}</small>
          </div>
        ) : (
          option.label
        );
      }}
      aria-label={label}
      title={title}
      placeholder={placeholder}
      disabled={disabled}
      size="xs"
      className="ui-choice"
      classNames={{ dropdown: 'ui-menu ui-choice-dropdown' }}
    />
  );
}

// A number as typed (blank: none), clamped to min and max (if given) as
// the field is left
export function NumberField({
  value,
  onChange,
  min,
  max,
  placeholder,
  label,
  title,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  min?: number;
  max?: number;
  placeholder?: string;
  label: string;
  title?: string;
  disabled?: boolean;
}) {
  return (
    <NumberInput
      disabled={disabled}
      value={value}
      onChange={(next) => onChange(String(next))}
      min={min}
      max={max}
      clampBehavior={min === undefined && max === undefined ? 'none' : 'blur'}
      hideControls
      placeholder={placeholder}
      aria-label={label}
      title={title}
      size="xs"
      className="ui-number"
    />
  );
}

// A quantity as its text (e.g. '2.5 Angstrom', as the library reads it):
// its number and its unit, the units of its kinds suggested (by kind,
// e.g. wavelength: Angstrom, nm, ...; any other can be typed). The list of
// units shows them all (as tall as they are) until what is typed differs
// from the unit the field held as it opened, and then those that match
// it. Its value is blank of none; off, it shows one it is given.
export function QuantityInput({
  value,
  units,
  onChange,
  label,
  placeholders = ['value', 'unit'],
  disabled,
}: {
  value: string;
  units: Record<string, string[]>;
  onChange: (value: string) => void;
  // (what it is, which its parts are labelled by)
  label: string;
  placeholders?: [string, string];
  disabled?: boolean;
}) {
  const [number, unit] = splitQuantity(value);
  // (the unit the field held as its list opened)
  const [opened, setOpened] = useState<string | null>(null);
  // (Mantine's own filter, of options of any value, keeps some of those
  // given: of their type)
  const filter: OptionsFilter = (input) => (input.search === opened ? input.options : (defaultOptionsFilter(input) as typeof input.options));
  const join = (n: string, u: string) => `${n.trim()} ${u.trim()}`.trim();
  return (
    <div className="ui-quantity">
      <TextInput
        className="ui-quantity-number"
        disabled={disabled}
        size="xs"
        value={number}
        placeholder={placeholders[0]}
        aria-label={`${label}: its value`}
        inputMode="decimal"
        onChange={(event) => onChange(join(event.currentTarget.value, unit))}
      />
      <Autocomplete
        className="ui-quantity-unit"
        disabled={disabled}
        size="xs"
        value={unit}
        data={Object.entries(units).map(([group, items]) => ({ group, items }))}
        filter={filter}
        maxDropdownHeight="none"
        onDropdownOpen={() => setOpened(unit)}
        placeholder={placeholders[1]}
        aria-label={`${label}: its unit`}
        comboboxProps={{ width: 'max-content', position: 'bottom-start' }}
        onChange={(next) => onChange(join(number, next))}
      />
    </div>
  );
}

// The number and the unit of the text of a quantity (e.g. '2.5 Angstrom');
// a text that does not start with a number is all unit
export function splitQuantity(text: string): [string, string] {
  const match = /^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$/.exec(text);
  return match ? [match[1], match[2].trim()] : ['', text.trim()];
}

// One of a few choices, as segments (one can be off, and say why on
// hover); small, in a line of text (e.g. the head of a parameter)
export function Segments<T extends string>({
  items,
  value,
  onChange,
  label,
  small,
}: {
  items: { value: T; label: string; disabled?: boolean; title?: string }[];
  value: T | null;
  onChange: (value: T) => void;
  label: string;
  small?: boolean;
}) {
  return (
    <SegmentedControl
      data={items.map(({ value: each, label: text, disabled, title }) => ({ value: each, label: title ? <span title={title}>{text}</span> : text, disabled }))}
      value={value ?? ''}
      onChange={(next) => onChange(next as T)}
      aria-label={label}
      size="xs"
      className={small ? 'ui-segments ui-segments-small' : 'ui-segments'}
    />
  );
}

// A check, with its label
export function Check({
  checked,
  onChange,
  label,
  title,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  title?: string;
}) {
  return (
    <Checkbox
      checked={checked}
      onChange={(event) => onChange(event.currentTarget.checked)}
      label={label}
      title={title}
      size="xs"
      className="ui-check"
    />
  );
}
