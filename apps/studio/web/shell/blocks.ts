// The blocks of a project (see gbkfit.project), one folder of each kind,
// in the sections of the tree: data, model and fit.

export interface Group {
  folder: string;
  // The name of the group, and what one of its blocks is
  title: string;
  kind: string;
}

export interface Section {
  key: 'data' | 'model' | 'fit';
  title: string;
  groups: Group[];
}

export const SECTIONS: Section[] = [
  {
    key: 'data',
    title: 'Data',
    groups: [
      { folder: 'datasets', title: 'Datasets', kind: 'Dataset' },
      { folder: 'psfs', title: 'PSFs', kind: 'PSF' },
      { folder: 'lsfs', title: 'LSFs', kind: 'LSF' },
    ],
  },
  {
    key: 'model',
    title: 'Model',
    groups: [
      { folder: 'galaxies', title: 'Galaxies', kind: 'Galaxy' },
      { folder: 'observations', title: 'Observations', kind: 'Observation' },
      { folder: 'noise', title: 'Noise', kind: 'Noise' },
    ],
  },
  {
    key: 'fit',
    title: 'Fit',
    groups: [
      { folder: 'params', title: 'Parameters', kind: 'Parameters' },
      { folder: 'fitters', title: 'Fitters', kind: 'Fitter' },
    ],
  },
];

// What a block of a folder is (e.g. 'Galaxy')
export function kindOf(folder: string): string {
  return SECTIONS.flatMap((section) => section.groups).find((group) => group.folder === folder)?.kind ?? folder;
}

// The kinds of data a dataset can hold, in groups, as the page names
// them; type is that of GBKFIT, and ready whether the page can add them
export interface DataKind {
  key: string;
  title: string;
  text: string;
  type: string;
  ready: boolean;
}

export const DATA_KINDS: { title: string; kinds: DataKind[] }[] = [
  {
    title: 'On a grid of pixels',
    kinds: [
      { key: 'cube', title: 'Cube', text: 'Spectra on a grid: an IFU or a radio cube', type: 'pixel_spectra', ready: true },
      { key: 'moments', title: 'Moment maps', text: 'Maps of the intensity, velocity and dispersion', type: 'pixel_moments', ready: true },
      { key: 'image', title: 'Flux map', text: 'An image of the light', type: 'pixel_brightness', ready: false },
    ],
  },
  {
    title: 'Slits and regions',
    kinds: [
      { key: 'slit', title: 'Slit spectra', text: 'Spectra along a slit', type: 'slit_spectra', ready: false },
      { key: 'slitless', title: 'Slitless spectra', text: 'A dispersed image', type: 'slitless_spectra', ready: false },
      { key: 'region-spectra', title: 'Spectra of regions', text: 'Spectra of bins or apertures', type: 'region_spectra', ready: false },
      { key: 'region-moments', title: 'Moments of regions', text: 'Moments of bins or apertures', type: 'region_moments', ready: false },
    ],
  },
  {
    title: 'Other',
    kinds: [
      { key: 'visibilities', title: 'Visibilities', text: 'The data of a radio interferometer', type: 'visibilities', ready: false },
      { key: 'profile', title: 'Profile', text: 'A table of values against radius', type: 'profile', ready: false },
    ],
  },
];

const KINDS_OF_DATA = DATA_KINDS.flatMap((group) => group.kinds);

export function dataKind(key: string): DataKind | undefined {
  return KINDS_OF_DATA.find((kind) => kind.key === key);
}

// The kind of data of a type of GBKFIT (e.g. that of a dataset)
export function dataKindOfType(type: unknown): DataKind | undefined {
  return KINDS_OF_DATA.find((kind) => kind.type === type);
}
