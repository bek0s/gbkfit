import { expect, test } from 'vitest';
import { address, parseAddress, type Screen } from './navigation';
import { latestFirst } from './remember';

test('every screen has an address that leads back to it', () => {
  const screens: Screen[] = [
    { kind: 'home' },
    { kind: 'fit' },
    { kind: 'block', folder: 'observations', name: 'ngc1313/hi #2' },
    { kind: 'new-dataset', data: 'cube' },
    { kind: 'draft', id: 'galaxy-cards', name: null },
    { kind: 'draft', id: 'galaxy-cards', name: 'ngc1313/disk' },
  ];
  for (const screen of screens) expect(parseAddress(address(screen))).toEqual(screen);
});

test('an address that is not a screen is the home', () => {
  for (const hash of ['', '#', '#runs', '#block/galaxies']) expect(parseAddress(hash)).toEqual({ kind: 'home' });
});

test('the latest item comes first, once, in a list cut to its length', () => {
  expect(latestFirst(['a', 'b', 'c'], 'c', 2)).toEqual(['c', 'a']);
});
