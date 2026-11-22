// The arrays of a file (e.g. the HDUs of a FITS file, see ArrayInfo) as
// GBKFIT reads them: without their axes of length 1, by name or by index
import type { ArrayInfo } from '../api/client';

// The shape of an array without its axes of length 1, as GBKFIT reads it
export function kept(array: ArrayInfo): number[] {
  return array.shape.filter((size) => size > 1);
}

export function keptAxes(array: ArrayInfo): string[] {
  return array.axes.filter((_, i) => array.shape[i] > 1);
}

export function keptAxisUnits(array: ArrayInfo): string[] {
  return array.axis_units.filter((_, i) => array.shape[i] > 1);
}

export function sameShape(a: ArrayInfo, b: ArrayInfo): boolean {
  return kept(a).join() === kept(b).join();
}

// How a configuration names an array: by its name if no other array of
// its file has it, or else by its index
export function arrayRef(arrays: ArrayInfo[], array: ArrayInfo): string | number {
  return array.name && arrays.filter((a) => a.name === array.name).length === 1 ? array.name : array.index;
}

export function arrayLabel(array: ArrayInfo): string {
  return array.name || `array ${array.index}`;
}
