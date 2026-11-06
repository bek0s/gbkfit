// The statistics of a region of a map: of its pixels with a value, their
// number, sum, mean, median and scatter (the rms about the mean), and
// their range; as text, for a chip or a tag beside the map.
import { significant } from './format';
import type { Region } from './MapView';

export interface RegionStats {
  count: number;
  sum: number;
  mean: number;
  median: number;
  scatter: number;
  min: number;
  max: number;
}

// (values: ny by nx, the row y = 0 first; none if no pixel has a value)
export function regionStats(values: ArrayLike<number>, nx: number, [x0, x1, y0, y1]: Region): RegionStats | null {
  const found: number[] = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const value = values[y * nx + x];
      if (Number.isFinite(value)) found.push(value);
    }
  }
  if (found.length === 0) return null;
  found.sort((a, b) => a - b);
  const sum = found.reduce((total, value) => total + value, 0);
  const mean = sum / found.length;
  const scatter = Math.sqrt(found.reduce((total, value) => total + (value - mean) ** 2, 0) / found.length);
  const half = found.length >> 1;
  const median = found.length % 2 ? found[half] : (found[half - 1] + found[half]) / 2;
  return { count: found.length, sum, mean, median, scatter, min: found[0], max: found[found.length - 1] };
}

export function statsText(stats: RegionStats | null, unit: string): string {
  if (stats === null) return 'no values';
  const of = (value: number) => significant(value, 3);
  return `${stats.count} px · mean ${of(stats.mean)} · median ${of(stats.median)} · σ ${of(stats.scatter)}${unit ? ` ${unit}` : ''}`;
}
