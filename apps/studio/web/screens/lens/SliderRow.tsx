// A parameter of a lens as a row of its sheet (see ui/rows: its value
// typed, whether a fit frees it, and as it opens, a slider), whose span,
// of no fit range, is that of its valid range, or of what the picture
// shows (the field, of a position), or about its value.
import { ParamRow } from '../../components/ui/rows';
import { rangeOf } from '../../components/values';
import { unitText } from '../galaxy/ParamField';
import { isObject } from './draft';

// A parameter as the server describes it (see LensInfo): its name in the
// lens (e.g. main_theta_e) and in its piece (theta_e), and its piece
export interface LensParam {
  name: string;
  key: string;
  title?: string | null;
  unit: string | null;
  default: number | null;
  range?: string | null;
  period?: number | null;
  owner: [string, number];
}

// The value of a parameter: a number, or free in a range ({value, min,
// max}; see the params block)
export type ParamValue = number | { value?: number; min?: number | null; max?: number | null } | undefined;

// The number of a parameter's value (its default if none)
export function numberOf(param: LensParam, value: ParamValue): number {
  const v = isObject(value) ? value.value : value;
  return typeof v === 'number' ? v : (param.default ?? 0);
}

// The span of the slider of a parameter of no fit range (see the module):
// its period; its valid range, if bounded; of a position (arcsec, of no
// bounds), what the picture shows (field: how far it reaches from the
// origin), and its value; else about its value
export function naturalSpan(param: LensParam, value: number, field: number): [number, number] {
  if (param.period) return [0, param.period];
  const [lo, hi] = rangeOf(param.range);
  if (Number.isFinite(lo) && Number.isFinite(hi)) return [lo, hi];
  if (param.unit === 'arcsec' && !Number.isFinite(lo) && !Number.isFinite(hi)) return [Math.min(-field, value), Math.max(field, value)];
  const reach = Math.max(4 * Math.abs(value), 0.1);
  return [Number.isFinite(lo) ? lo : value - reach, Number.isFinite(hi) ? hi : value + reach];
}

export function SliderRow({
  param,
  value,
  field,
  open,
  onOpen,
  onChange,
}: {
  param: LensParam;
  value: ParamValue;
  field: number;
  // Whether its name and slider show, and what opens or closes it
  open: boolean;
  onOpen: () => void;
  onChange: (value: ParamValue) => void;
}) {
  const number = numberOf(param, value);
  const free = isObject(value) ? value : null;
  return (
    <ParamRow
      param={param}
      title={param.title ?? param.key}
      value={{ number, free: free !== null, min: free?.min, max: free?.max }}
      natural={naturalSpan(param, number, field)}
      unit={unitText(param.unit)}
      open={open}
      onOpen={onOpen}
      onValue={(v) => onChange(free ? { ...free, value: v } : v)}
      onFree={(on, span) => onChange(on ? { value: number, min: span[0], max: span[1] } : (free?.value ?? number))}
      onRange={(end, v) => free && onChange({ ...free, [end]: v })}
    />
  );
}
