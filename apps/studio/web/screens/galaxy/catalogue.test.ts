import { describe, expect, it } from 'vitest';
import { lostIfThin, withThickness } from './catalogue';

const thick = {
  type: 'thick_disk',
  name: 'disk',
  density: { ptraits: [{ type: 'exponential' }, { type: 'gauss' }], htraits: [{ type: 'sech2', zmax: 6 }, { type: 'gauss', zmax: 6 }] },
  kinematics: { type: 'traits', vptraits: { type: 'arctan' }, vhtraits: null },
  zptraits: [{ type: 'nw_harmonic', order: 1, zmax: 6 }],
};

describe('the thickness of a disk', () => {
  it('names what a thick disk made thin loses, of what it has', () => {
    expect(lostIfThin(thick)).toEqual(['the heights of its density', 'the heights of its motions', 'its warps']);
  });

  it('makes a disk thin without what only a thick disk has', () => {
    expect(withThickness(thick, false)).toEqual({
      type: 'thin_disk',
      name: 'disk',
      density: { ptraits: [{ type: 'exponential' }, { type: 'gauss' }] },
      kinematics: { type: 'traits', vptraits: { type: 'arctan' } },
    });
  });

  it('makes a disk thick with a height for each trait of its density', () => {
    const thin = withThickness(thick, false);
    expect(withThickness(thin, true).density).toEqual({ ptraits: (thin.density as { ptraits: unknown }).ptraits, htraits: [{ type: 'sech2', zmax: 6 }, { type: 'sech2', zmax: 6 }] });
    expect(withThickness({ type: 'thin_disk', density: { ptraits: { type: 'exponential' } } }, true).density).toEqual({ ptraits: { type: 'exponential' }, htraits: { type: 'sech2', zmax: 6 } });
  });
});
