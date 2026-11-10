import { expect, test } from 'vitest';
import type { Path } from './options';
import { forRender, keyName, resample, settingOf, shortName, sliderReach, transition, validRange, withFreedom, withValues, type Param } from './params';

const param = (name: string, owner: Path, size = 1, extra: Partial<Param> = {}): Param => ({
  name,
  size,
  default: 1,
  unit: null,
  range: '(-inf, inf)',
  period: null,
  owner,
  ...extra,
});

test('keys name their parameters', () => {
  expect(keyName('disk_posa')).toBe('disk_posa');
  expect(keyName('disk_posa[0:3]')).toBe('disk_posa');
});

test('a property is a setting of the values of its parameter', () => {
  const p = param('a', [], 3);
  expect(settingOf({}, p)).toEqual({ kind: 'fixed', values: [1] });
  expect(settingOf({ a: 2 }, p)).toEqual({ kind: 'fixed', values: [2] });
  expect(settingOf({ a: [1, 2, 3] }, p)).toEqual({ kind: 'fixed', values: [1, 2, 3] });
  expect(settingOf({ a: { value: 2, min: 0, max: 5 } }, p)).toEqual({ kind: 'free', values: [2], min: 0, max: 5 });
  expect(settingOf({ a: { '*value': [1, 2, 3], min: 0, max: 5 } }, p)).toEqual({ kind: 'free', values: [1, 2, 3], min: 0, max: 5 });
  expect(settingOf({ a: 'b * 2' }, p)).toEqual({ kind: 'tied', expression: 'b * 2' });
  expect(settingOf({ 'a[0]': 2 }, p)).toEqual({ kind: 'other' });
});

test('values are written in the form of their property', () => {
  const p = param('a', [], 3);
  expect(withValues(undefined, p, [2, 2, 2])).toBe(2);
  expect(withValues(undefined, p, [1, 2, 3])).toEqual([1, 2, 3]);
  expect(withValues({ value: 1, min: 0, max: 5 }, p, [1, 2, 3])).toEqual({ '*value': [1, 2, 3], min: 0, max: 5 });
  expect(withValues({ '*value': [1, 2, 3], min: 0 }, p, [4, 4, 4])).toEqual({ value: 4, min: 0 });
  const free = withFreedom(2, param('b', []), { kind: 'fixed', values: [2] }, true, [0, 5]);
  expect(free).toEqual({ value: 2, min: 0, max: 5 });
  expect(withFreedom(free, param('b', []), { kind: 'free', values: [2], min: 0, max: 5 }, false, [0, 5])).toBe(2);
});

test('the ranges of sliders stay within the valid ranges', () => {
  expect(validRange(param('a', [], 1, { range: '[0, 90]' }))).toEqual([0, 90]);
  expect(validRange(param('a', [], 1, { range: '(-inf, inf)' }))).toEqual([-Infinity, Infinity]);
  const incl = sliderReach(param('incl', [], 1, { unit: 'deg', range: '[0, 90]' }), 20);
  expect([incl.min, incl.max]).toEqual([0, 90]);
  const posa = sliderReach(param('posa', [], 1, { unit: 'deg', range: '(-inf, inf)', period: 360 }), 20);
  expect([posa.min, posa.max]).toEqual([-180, 360]);
  const logm = sliderReach(param('mass_halo_log_m', [], 1, { default: 12 }), 20);
  expect([logm.min, logm.max]).toEqual([9, 15]);
  const s = sliderReach(param('s', [], 1, { range: '(0, inf)', default: 5 }), 20);
  expect([s.min, s.max]).toEqual([0, 20]);
});

test('short names drop the prefixes of their owners', () => {
  const options = { components: [{ name: 'disk', vptraits: [{}, {}, { name: 'fast' }] }], mass: [{ name: 'halo' }] };
  const params = [
    param('disk_vpt2_rt', ['components', 0, 'vptraits', 1]),
    param('disk_vpt2_vt', ['components', 0, 'vptraits', 1]),
    param('disk_vpt_fast_vt', ['components', 0, 'vptraits', 2]),
    param('disk_posa', ['components', 0]),
    param('disk_nii6583_ratio', ['components', 0, 'emission', 'lines', 1]),
    param('mass_halo_log_m', ['mass', 0]),
  ];
  expect(params.map((p) => shortName(p, options, params))).toEqual(['rt', 'vt', 'vt', 'posa', 'nii6583 ratio', 'log_m']);
});

test('resampling by places, or evenly', () => {
  expect(resample([0, 10], 3)).toEqual([0, 5, 10]);
  expect(resample([0, 10, 20], 2)).toEqual([0, 20]);
  expect(resample([0, 10, 20], 3, [0, 1, 2], [0, 0.5, 4])).toEqual([0, 5, 20]);
});

test('the properties follow the parameters of the galaxy', () => {
  const vpt = (i: number): Path => ['components', 0, 'vptraits', i];
  const before = [param('d_vpt_vt', vpt(0)), param('d_vpt2_vt', vpt(1)), param('d_posa', ['components', 0], 3)];
  const after = [param('d_vpt_vt', vpt(0)), param('d_posa', ['components', 0], 5)];
  const radii = { before: () => [0, 1, 2], after: () => [0, 0.5, 1, 1.5, 2] };
  const properties = { d_vpt_vt: 100, d_vpt2_vt: { value: 50, min: 0 }, d_posa: [0, 10, 20], other: 3 };
  // (the first trait removed: the second is the first now)
  const moved = (owner: Path) => (owner.length === 4 ? (owner[3] === 0 ? null : [...owner.slice(0, 3), (owner[3] as number) - 1]) : owner);
  expect(transition(properties, before, after, moved, radii)).toEqual({
    d_vpt_vt: { value: 50, min: 0 },
    d_posa: [0, 5, 10, 15, 20],
    other: 3,
  });
  // (without a map: by name; the parameters gone, gone)
  expect(transition(properties, before, after, null, radii)).toEqual({ d_vpt_vt: 100, d_posa: [0, 5, 10, 15, 20], other: 3 });
  // (to a scalar, the first)
  expect(transition({ d_posa: { '*value': [1, 2, 3], min: 0 } }, before, [param('d_posa', ['components', 0])], null, radii)).toEqual({
    d_posa: { value: 1, min: 0 },
  });
});

test('a render has the properties of the parameters alone', () => {
  const params = [param('a', [], 3), param('b', [])];
  expect(forRender({ a: [1, 2], b: 2, c: 3, 'a[0]': 1, 'b[0]': 1 }, params)).toEqual({ a: [1, 1.5, 2], b: 2, 'a[0]': 1 });
});
