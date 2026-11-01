// Inputs (of Mantine; see theme.ts), with the conventions of the Studio:
// a number field whose value is its text (blank: none), clamped to its
// range as it is left; a choice of a few (segments) or of a list (the
// select of the browser); and a check.
import { Checkbox, NativeSelect, NumberInput, SegmentedControl } from '@mantine/core';

// One of a list of choices, as the select of the browser
export function Choice({
  items,
  value,
  onChange,
  label,
  title,
}: {
  items: { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
  label: string;
  title?: string;
}) {
  return (
    <NativeSelect
      data={items}
      value={value}
      onChange={(event) => onChange(event.currentTarget.value)}
      aria-label={label}
      title={title}
      size="xs"
      className="ui-choice"
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
