// How the page writes numbers.

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
