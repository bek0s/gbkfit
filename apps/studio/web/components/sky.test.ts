import { describe, expect, it } from 'vitest';
import { frameOf } from './sky';

describe('the pixels of an array on the sky', () => {
  it('are known from its step and reference pixel', () => {
    expect(frameOf([-2, 2, 5], [3, 4, 0])).toEqual({ step: [-2, 2], rpix: [3, 4] });
  });

  it('are not known without them', () => {
    expect(frameOf(null, [3, 4])).toBeNull();
    expect(frameOf([2, 2], undefined)).toBeNull();
    expect(frameOf([0, 2], [3, 4])).toBeNull();
  });
});
