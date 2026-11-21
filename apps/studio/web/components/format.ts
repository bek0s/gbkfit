// How the page writes numbers, and times.

// A value to a number of significant digits, without trailing zeros
export function significant(value: number, digits: number): string {
  // (very small or large, with an exponent, e.g. 4.91e-6: short enough for
  // a colour bar or a readout)
  const size = Math.abs(value);
  if (size !== 0 && Number.isFinite(size) && (size < 1e-3 || size >= 1e6)) {
    return value.toExponential(digits - 1).replace(/\.?0+e/, 'e').replace('e+', 'e');
  }
  return String(Number(value.toPrecision(digits)));
}

// A time (ISO 8601), short, e.g. in a list: of today, its hour; of this
// year, its day and hour; else its day and year (as given, if it is not
// a time)
export function when(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const today = new Date();
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (date.toDateString() === today.toDateString()) return `today ${time}`;
  if (date.getFullYear() === today.getFullYear()) return `${date.toLocaleDateString([], { day: 'numeric', month: 'short' })} ${time}`;
  return date.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' });
}

// A time (ISO 8601) in full, e.g. in a sentence (as given, if it is not
// a time)
export function whenInFull(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}
