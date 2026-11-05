// The round trip that editing a dataset rests on: the options of prep of
// a draft, recorded by the server, give back the draft (its files and its
// edits); and what a draft needs of its spectral axis.
import { describe, expect, it } from 'vitest';
import type { ArrayInfo } from '../../api/client';
import {
  arraysOf,
  draftOfPreparation,
  EMPTY_ITEM,
  needsOf,
  NO_EDITS,
  NO_SPECTRAL,
  preparedOptions,
  prepRequest,
  siblings,
  type Draft,
  type Kind,
  type Resolved,
} from './model';

function array(index: number, name: string, shape: number[], axes = ['RA---TAN', 'DEC--TAN', 'VRAD']): ArrayInfo {
  return { index, name, shape, axes: axes.slice(0, shape.length), unit: 'Jy/beam', step: null, beam: null, rms: null, ambiguous_velocity: false, problems: [] };
}

function resolve(kind: Kind, draft: Draft, files: Record<string, ArrayInfo[]>): Resolved {
  const describe = (path: string) => ({ arrays: files[path] });
  const ndim = kind === 'cube' ? 3 : 2;
  return { kind, draft, arrays: Object.fromEntries(Object.entries(draft.items).map(([key, item]) => [key, arraysOf(item, describe, ndim)])) };
}

describe('a draft prepared and edited again', () => {
  it('gives back the files and the edits of a cube', () => {
    const files = { 'p/data/cube.fits': [array(0, 'PRIMARY', [40, 30, 20]), array(1, 'STAT', [40, 30, 20])] };
    const draft: Draft = {
      items: { cube: { files: { data: { file: 'p/data/cube.fits' }, error: { file: 'p/data/cube.fits', array: 'STAT' } }, errorValue: null, errorKind: null } },
      spectral: { convention: 'radio', line: '', unit: '', redshift: '' },
      edits: { ...NO_EDITS, x: ['2', '30'], channels: ['5', '14'], values: { cube: { below: '0.001', above: '', sigma: '' } }, regions: '1', minify: true },
    };
    const request = prepRequest(resolve('cube', draft, files), 'p');
    expect(request.options).toMatchObject({ roi_spat: [2, 31, 0, 30], roi_spec: [5, 15], clip_min: 0.001, error_kind: 'variance', velocity: 'radio' });
    const prepared = { file_d: 'p/data/prep_cube.fits', file_e: 'p/data/prep_cube_error.fits' };
    const dataset = preparedOptions(resolve('cube', draft, files), prepared, 'p');
    const again = draftOfPreparation('cube', dataset, { ...request, files: prepared, time: '' }, 'p');
    expect(again.items.cube.files).toEqual({ data: { file: 'p/data/cube.fits', array: 'PRIMARY' }, error: { file: 'p/data/cube.fits', array: 'STAT' } });
    expect(again.items.cube.errorKind).toBe('variance');
    expect(again.edits).toEqual({ ...draft.edits, y: ['0', '29'] });
    expect(again.spectral.convention).toBe('radio');
  });

  it('finds each moment map in the lists of prep', () => {
    const map = (name: string) => [array(0, name, [10, 8])];
    const files = { 'data/m0.fits': map('M0'), 'data/m1.fits': map('M1') };
    const draft: Draft = {
      items: { moment0: { ...EMPTY_ITEM, files: { data: { file: 'data/m0.fits' } } }, moment1: { ...EMPTY_ITEM, files: { data: { file: 'data/m1.fits' } } } },
      spectral: NO_SPECTRAL,
      edits: { ...NO_EDITS, values: { moment1: { below: '', above: '300', sigma: '' } } },
    };
    const request = prepRequest(resolve('moments', draft, files), '.');
    expect(request.options).toMatchObject({ file_d: ['data/m0.fits', 'data/m1.fits'], clip_max: [null, 300], clip_min: null });
    const prepared = { file_d: ['data/prep_m0.fits', 'data/prep_m1.fits'] };
    const dataset = preparedOptions(resolve('moments', draft, files), prepared, '.');
    expect(dataset).toEqual({ type: 'pixel_moments', moment0: { data: 'data/prep_m0.fits' }, moment1: { data: 'data/prep_m1.fits' } });
    const again = draftOfPreparation('moments', dataset, { ...request, files: prepared, time: '' }, '.');
    expect(again.items.moment1.files.data?.file).toBe('data/m1.fits');
    expect(again.edits.values.moment1.above).toBe('300');
  });
});

describe('the spectral axis of a cube', () => {
  const draft = (line: string): Draft => ({
    items: { cube: { ...EMPTY_ITEM, files: { data: { file: 'ifu.fits' } } } },
    spectral: { ...NO_SPECTRAL, line, unit: 'Angstrom' },
    edits: NO_EDITS,
  });
  const ifu = { 'ifu.fits': [array(1, 'DATA', [80, 70, 240], ['RA---TAN', 'DEC--TAN', 'AWAV'])] };

  it('needs a line for channels of wavelengths, and not once it has one', () => {
    expect(needsOf(resolve('cube', draft(''), ifu)).map((need) => need.place)).toEqual(['spectral']);
    expect(needsOf(resolve('cube', draft('6562.8'), ifu))).toEqual([]);
  });
});

describe('the arrays offered for a mask', () => {
  it('leave out errors and quality flags', () => {
    const files = { 'muse.fits': [array(1, 'DATA', [8, 7, 6]), array(2, 'STAT', [8, 7, 6]), array(3, 'DQ', [8, 7, 6]), array(4, 'MASK', [8, 7, 6])] };
    const resolved = resolve('cube', { items: { cube: { ...EMPTY_ITEM, files: { data: { file: 'muse.fits' } } } }, spectral: NO_SPECTRAL, edits: NO_EDITS }, files);
    expect(siblings(resolved.arrays.cube, 'mask').map((a) => a.name)).toEqual(['MASK']);
    expect(siblings(resolved.arrays.cube, 'error').map((a) => a.name)).toEqual(['STAT', 'MASK']);
  });
});
