// Inputs (of Mantine; see theme.ts), with the conventions of the Studio:
// a number field whose value is its text (blank: none), clamped to its
// range as it is left; a choice of a few (segments) or of a list (a
// field whose list drops as the menus do); and a check.
import { Checkbox, NumberInput, SegmentedControl, Select } from '@mantine/core';

// One of a list of choices (some can be off, and each can say what it
// is under its label), as a field whose list drops as the menus do (see
// overlays)
export function Choice({
  items,
  value,
  onChange,
  label,
  title,
}: {
  items: { value: string; label: string; disabled?: boolean; description?: string }[];
  value: string;
  onChange: (value: string) => void;
  label: string;
  title?: string;
}) {
  const described = new Map(items.map((item) => [item.value, item.description]));
  return (
    <Select
      data={items.map(({ value, label, disabled }) => ({ value, label, disabled }))}
      value={value}
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

// One of a few choices, as segments
export function Segments<T extends string>({
  items,
  value,
  onChange,
  label,
}: {
  items: { value: T; label: string; disabled?: boolean }[];
  value: T | null;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <SegmentedControl
      data={items}
      value={value ?? ''}
      onChange={(next) => onChange(next as T)}
      aria-label={label}
      size="xs"
      className="ui-segments"
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
