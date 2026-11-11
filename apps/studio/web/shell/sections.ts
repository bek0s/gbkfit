// The sections of the Studio, as its rail shows them: the home of the
// project; the blocks it is built of (data, galaxies, observations), each
// a list of groups of one folder each; and the tools that use them
// (compare, fit). A block folds into its owner when it has no screen of
// its own: the parameters of a galaxy are part of the galaxy's screen.
import type { Screen } from './navigation';

export type SectionKey = 'home' | 'data' | 'galaxies' | 'observations' | 'compare' | 'fit';

export interface Group {
  folder: string;
  // The name of the group, and what one of its blocks is
  title: string;
  kind: string;
}

export interface Section {
  key: SectionKey;
  title: string;
  // Whether it is a tool (compare, fit) rather than blocks to build
  tool: boolean;
  groups: Group[];
}

export const SECTIONS: Section[] = [
  { key: 'home', title: 'Home', tool: false, groups: [] },
  { key: 'data', title: 'Data', tool: false, groups: [{ folder: 'datasets', title: 'Datasets', kind: 'Dataset' }] },
  { key: 'galaxies', title: 'Galaxies', tool: false, groups: [{ folder: 'galaxies', title: 'Galaxies', kind: 'Galaxy' }] },
  {
    key: 'observations',
    title: 'Observations',
    tool: false,
    groups: [
      { folder: 'observations', title: 'Observations', kind: 'Observation' },
      { folder: 'psfs', title: 'PSFs', kind: 'PSF' },
      { folder: 'lsfs', title: 'LSFs', kind: 'LSF' },
      { folder: 'noise', title: 'Noise', kind: 'Noise model' },
    ],
  },
  { key: 'compare', title: 'Compare', tool: true, groups: [] },
  {
    key: 'fit',
    title: 'Fit',
    tool: true,
    groups: [
      { folder: 'setups', title: 'Fits', kind: 'Fit' },
      { folder: 'fitters', title: 'Fitters', kind: 'Fitter' },
    ],
  },
];

// The kinds of the blocks of the folders that no list shows
const OTHER_KINDS: Record<string, string> = { params: 'Parameters' };

export function sectionNamed(key: SectionKey): Section {
  return SECTIONS.find((section) => section.key === key)!;
}

// The section of the blocks of a folder (the galaxies, for the parameters
// of a galaxy)
export function sectionOfFolder(folder: string): SectionKey {
  if (folder === 'params') return 'galaxies';
  return SECTIONS.find((section) => section.groups.some((group) => group.folder === folder))?.key ?? 'home';
}

// The section of a screen
export function sectionOf(screen: Screen): SectionKey {
  switch (screen.kind) {
    case 'home':
      return 'home';
    case 'section':
      return screen.key;
    case 'block':
      return sectionOfFolder(screen.folder);
    case 'new-dataset':
      return 'data';
    case 'compare':
      return 'compare';
    case 'fit':
      return 'fit';
  }
}

// What a block of a folder is (e.g. 'Galaxy')
export function kindOf(folder: string): string {
  return (
    SECTIONS.flatMap((section) => section.groups).find((group) => group.folder === folder)?.kind ??
    OTHER_KINDS[folder] ??
    folder
  );
}
