import { expect, test } from 'vitest';
import { asList, freshName, getAt, ringRadii, roundStep, setAt, withOpacitiesInStep } from './options';

test('an option of one item or a list reads as a list', () => {
  expect(asList(undefined)).toEqual([]);
  expect(asList({ type: 'a' })).toEqual([{ type: 'a' }]);
  expect(asList([{ type: 'a' }, { type: 'b' }])).toHaveLength(2);
  // (an index into one item is that item)
  expect(getAt({ ptraits: { type: 'a' } }, ['ptraits', 0, 'type'])).toBe('a');
  expect(getAt({ ptraits: { type: 'a' } }, ['ptraits', 1])).toBeUndefined();
});

test('setAt changes a copy, and keeps the form of a list', () => {
  const options = { components: [{ name: 'a', density: { ptraits: { type: 'x' } } }, { name: 'b' }] };
  const next = setAt(options, ['components', 0, 'density', 'ptraits', 0, 'type'], 'y');
  expect(getAt(next, ['components', 0, 'density', 'ptraits'])).toEqual({ type: 'y' });
  expect(options.components[0].density?.ptraits.type).toBe('x');
  // (one item stays one, until a second is added: then a list)
  const two = setAt(next, ['components', 0, 'density', 'ptraits', 1], { type: 'z' });
  expect(getAt(two, ['components', 0, 'density', 'ptraits'])).toEqual([{ type: 'y' }, { type: 'z' }]);
  // (removed: the others in order, a list of one still a list; none: the
  // option gone)
  const one = setAt(options, ['components', 0], undefined);
  expect(one.components).toEqual([{ name: 'b' }]);
  expect(setAt(one, ['components', 0], undefined)).toEqual({});
  expect(setAt({ a: 1, b: 2 }, ['a'], undefined)).toEqual({ b: 2 });
});

test('fresh names are not taken', () => {
  expect(freshName([], 'disk')).toBe('disk');
  expect(freshName(['disk', 'disk2'], 'disk')).toBe('disk3');
});

test('the rings of every form', () => {
  expect(ringRadii({ max: 4, sep: 2 })).toEqual([0, 2, 4]);
  expect(ringRadii({ max: 5, sep: 2 })).toEqual([0, 2, 4, 6]);
  expect(ringRadii({ max: 4, len: 3 })).toEqual([0, 2, 4]);
  expect(ringRadii([0, 1, 3])).toEqual([0, 1, 3]);
  expect(ringRadii({ max: 0, sep: 1 })).toEqual([]);
  expect(roundStep(0.37)).toBeCloseTo(0.5);
  expect(roundStep(1.2)).toBe(1);
});

test('the coefficients of self-absorption follow the lines', () => {
  const galaxy = {
    components: [{ name: 'd', emission: { type: 'lines', lines: ['halpha', 'nii6583'] }, opacity: { type: 'line', coefficients: { type: 'uniform' } } }],
  };
  const next = withOpacitiesInStep(galaxy);
  expect(getAt(next, ['components', 0, 'opacity', 'coefficients'])).toEqual({ halpha: { type: 'uniform' }, nii6583: { type: 'uniform' } });
  expect(withOpacitiesInStep(next)).toBe(next);
});
