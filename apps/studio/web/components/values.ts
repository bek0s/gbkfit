// The values of parameters as the sheets set them (the lens's, the
// galaxy's): their valid ranges as the library writes them, a value
// settled as it is set (wrapped into its period, kept in its valid
// range), and a value dragged, rounded.

// What a value is settled by: the valid range of its parameter, as the
// library writes it (e.g. '[0, inf)'), and its period, if it has one
// (e.g. a position angle)
export interface Bounds {
  range?: string | null;
  period?: number | null;
}

// The ends of a range as the library writes it (e.g. '[0, inf)', '(0,
// 90]'); none given, the whole line
export function rangeOf(range: string | null | undefined): [number, number] {
  const match = range?.match(/^[[(]\s*([^,]+),\s*([^\])]+)[\])]$/);
  const read = (text: string | undefined, infinite: number) => {
    const value = Number(text?.trim().replace(/^(-?)inf$/, '$1Infinity'));
    return Number.isNaN(value) ? infinite : value;
  };
  return [read(match?.[1], -Infinity), read(match?.[2], Infinity)];
}

// Whether a value is one a parameter can take (in its valid range)
export function isValid(value: number, bounds: Bounds): boolean {
  const [lo, hi] = rangeOf(bounds.range);
  return Number.isFinite(value) && value >= lo && value <= hi;
}

// A value as it is set, by a row or by a handle: wrapped into its period,
// kept in its valid range
export function settled(value: number, bounds: Bounds): number {
  const { period } = bounds;
  const wrapped = period ? ((value % period) + period) % period : value;
  const [lo, hi] = rangeOf(bounds.range);
  return Math.min(Math.max(wrapped, lo), hi);
}

// A value dragged over a span, rounded to a thousandth of the span (a
// power of ten: e.g. 0.001 of a redshift over 0 to 4): a drag says no more
// (a value typed is kept as typed)
export function snap(value: number, span: [number, number]): number {
  const step = 10 ** Math.floor(Math.log10(Math.abs(span[1] - span[0]) / 1000 || 1e-3));
  return Number((Math.round(value / step) * step).toFixed(Math.max(0, -Math.floor(Math.log10(step)))));
}
