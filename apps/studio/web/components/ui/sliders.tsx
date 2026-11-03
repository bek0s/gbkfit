// Sliders (of Mantine; see theme.ts), with the conventions of the Studio:
// a range whose handle at an end of the slider means no limit there
// (null, an open range), and a slider of one value that may be none (at
// its low end); and a plain slider of a step (e.g. a channel).
import { RangeSlider, Slider } from '@mantine/core';

// A range of values from min to max: a handle at an end means no limit
// (null) at that end. Its ends are marked as open (−∞ and ∞), and so is a
// mark (e.g. 0); its values show, formatted, as a handle is moved.
export function ValueRange({
  min,
  max,
  step,
  value,
  onChange,
  format,
  mark,
  label,
}: {
  min: number;
  max: number;
  step: number;
  value: [number | null, number | null];
  onChange: (value: [number | null, number | null]) => void;
  format: (value: number) => string;
  mark?: number;
  label: string;
}) {
  const [low, high] = value;
  const clamp = (x: number) => Math.min(max, Math.max(min, x));
  return (
    <RangeSlider
      min={min}
      max={max}
      step={step}
      minRange={step}
      value={[low === null ? min : clamp(low), high === null ? max : clamp(high)]}
      onChange={([a, b]) => onChange([a <= min ? null : a, b >= max ? null : b])}
      label={(x) => (x <= min || x >= max ? 'no limit' : format(x))}
      marks={[
        { value: min, label: '−∞' },
        ...(mark === undefined ? [] : [{ value: mark, label: format(mark) }]),
        { value: max, label: '∞' },
      ]}
      thumbLabel={label}
      thumbFromLabel={`${label}: the low end`}
      thumbToLabel={`${label}: the high end`}
      size="md"
      className="ui-slider"
    />
  );
}

// One value from min to max, or none (at min). Its ends are marked (none,
// and max formatted); its value shows, formatted, as it is moved.
export function ValueSlider({
  min,
  max,
  step,
  value,
  onChange,
  format,
  label,
}: {
  min: number;
  max: number;
  step: number;
  value: number | null;
  onChange: (value: number | null) => void;
  format: (value: number) => string;
  label: string;
}) {
  return (
    <Slider
      min={min}
      max={max}
      step={step}
      value={value === null ? min : Math.min(max, Math.max(min, value))}
      onChange={(x) => onChange(x <= min ? null : x)}
      label={(x) => (x <= min ? 'None' : format(x))}
      marks={[
        { value: min, label: 'None' },
        { value: max, label: format(max) },
      ]}
      thumbLabel={label}
      size="md"
      className="ui-slider"
    />
  );
}

// A value from min to max by steps, as a slider (e.g. a channel), its
// value shown on its handle as it is moved (or always)
export function StepSlider({
  min,
  max,
  step = 1,
  value,
  onChange,
  format,
  label,
  always,
}: {
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  label: string;
  always?: boolean;
}) {
  return (
    <Slider
      min={min}
      max={max}
      step={step}
      value={Math.min(max, Math.max(min, value))}
      onChange={onChange}
      label={format ?? null}
      labelAlwaysOn={always}
      thumbLabel={label}
      size="md"
      className={always ? 'ui-slider ui-slider-labelled' : 'ui-slider'}
    />
  );
}
