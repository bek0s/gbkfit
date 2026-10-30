// How the page writes numbers.

// A value to a number of significant digits, without trailing zeros
export function significant(value: number, digits: number): string {
  return String(Number(value.toPrecision(digits)));
}
