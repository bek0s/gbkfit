import { describe, expect, it } from 'vitest';
import { toPixels, toSky } from './sky';

describe('regions of the sky in the pixels of a map', () => {
  const frame = { step: [2, 2] as [number, number], rpix: [5, 5] as [number, number] };

  it('go to the sky and back', () => {
    const box = toSky([4, 6, 5, 5], frame);
    expect(box).toEqual([-3, 3, -1, 1]);
    expect(toPixels(box, frame, [11, 11])).toEqual([4, 6, 5, 5]);
  });

  it('take the pixels of another map whose centres are in them', () => {
    // (pixels of 1 arcsec, measured from their middle)
    expect(toPixels([-3, 3, -1, 1], { step: [1, 1], rpix: [10, 10] }, [21, 21])).toEqual([7, 13, 9, 11]);
    // (beyond the map: none)
    expect(toPixels([100, 120, 0, 1], frame, [11, 11])).toBeNull();
  });

  it('keep their order with steps that run backwards (e.g. RA)', () => {
    const flipped = { step: [-2, 2] as [number, number], rpix: [5, 5] as [number, number] };
    expect(toPixels(toSky([2, 3, 1, 1], flipped), flipped, [11, 11])).toEqual([2, 3, 1, 1]);
  });
});
