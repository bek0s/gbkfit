import { expect, test } from 'vitest';
import { kindLine, retypedObservation, withPath } from './model';
import { pairText } from './OptionRows';
import { testGalaxy } from './testGalaxy';

test('a kind is said in the first clause of its description', () => {
  expect(kindLine('Spectra on a grid of pixels: a spectral cube of the model. More.')).toBe('Spectra on a grid of pixels');
  expect(kindLine('Interferometric visibilities. Their transform.')).toBe('Interferometric visibilities');
  expect(kindLine(undefined)).toBe('');
});

test('the test galaxy is placed where it is asked to be', () => {
  const { options, properties } = testGalaxy({ inclination: 60, angle: 40, size: 50, rotation: 150 }, [1083, -741], 428, true);
  expect(properties).toMatchObject({ disk_xpos: 1083, disk_ypos: -741, disk_vsys: 428, disk_incl: 60, disk_posa: 40, disk_rpt_s: 50, disk_vpt_vt: 150 });
  const [disk] = options.components as Record<string, unknown>[];
  expect(disk.light).toEqual({ type: 'lines' });
  expect(testGalaxy({ inclination: 0, angle: 0, size: 1, rotation: 0 }, [0, 0], 0, false).options.components).toMatchObject([{ light: { type: 'continuum' } }]);
});

test('another kind keeps the options it has too', () => {
  const member = { properties: { type: {}, size: {}, step: {}, instrument: {} } };
  expect(retypedObservation({ type: 'pixel_spectra', size: 8, spectral: { size: 4 }, instrument: { psf: 'beam' } }, member, 'pixel_brightness')).toEqual({ size: 8, instrument: { psf: 'beam' }, type: 'pixel_brightness' });
});

test('options are set at a path, and emptied ones removed', () => {
  expect(withPath({}, ['offsets', 'position'], true)).toEqual({ offsets: { position: true } });
  expect(withPath({ offsets: { position: true } }, ['offsets', 'position'], undefined)).toEqual({});
});

test('a pair is one number if both are the same', () => {
  expect(pairText([6, 6])).toBe('6');
  expect(pairText([114, 100])).toBe('114 × 100');
  expect(pairText([2, 2, 1])).toBe('2 × 2 × 1');
  expect(pairText(undefined)).toBe('');
});
