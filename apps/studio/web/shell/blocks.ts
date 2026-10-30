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
