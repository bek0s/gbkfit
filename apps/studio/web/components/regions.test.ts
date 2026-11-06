import { describe, expect, it } from 'vitest';
import { regionStats, statsText } from './regions';

describe('the statistics of a region', () => {
  // (3 by 2: the row y = 0 first)
  const values = [1, 2, NaN, 4, 5, 6];

  it('are of the pixels with a value in it', () => {
    const stats = regionStats(values, 3, [0, 1, 0, 1])!;
    expect(stats.count).toBe(4);
    expect(stats.sum).toBe(12);
    expect(stats.mean).toBe(3);
    expect(stats.median).toBe(3);
    expect(stats.scatter).toBeCloseTo(Math.sqrt(2.5));
    expect([stats.min, stats.max]).toEqual([1, 5]);
  });

  it('are none without values, and read as text', () => {
    expect(regionStats(values, 3, [2, 2, 0, 0])).toBeNull();
    expect(statsText(regionStats(values, 3, [0, 0, 0, 1]), 'km/s', 2)).toBe('mean 2.5 · median 2.5 · σ 1.5 km/s');
    expect(statsText(regionStats(values, 3, [0, 2, 0, 0]), '', 3)).toBe('mean 1.5 · median 1.5 · σ 0.5 (of 2 px with a value)');
  });
});
