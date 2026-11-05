// Drafts of the screens of observations: layouts to compare before one is
// chosen, each reachable from the tree (its section Screen drafts). The
// blocks of observations open in the layout chosen for them (see
// useLayout), the first by default. (Galaxies have their screen, see
// screens/galaxy/GalaxyScreen.)

export interface Draft {
  id: string;
  folder: 'observations';
  // Its letter, its title and what it puts first
  letter: string;
  title: string;
  text: string;
}

export const DRAFTS: Draft[] = [
  {
    id: 'observation-pipeline',
    folder: 'observations',
    letter: 'A',
    title: 'Pipeline',
    text: 'The stages from the galaxy to the data (line of sight, instrument, observable, data, likelihood) as a row of cards; the stage chosen is edited under it.',
  },
  {
    id: 'observation-form',
    folder: 'observations',
    letter: 'B',
    title: 'Form and checks',
    text: 'All the options in one form, beside the checks of the observation against its data (types, pixels, channels, beam) and its footprint on the sky.',
  },
];

export function draft(id: string): Draft | undefined {
  return DRAFTS.find((d) => d.id === id);
}

export function draftsOf(folder: string): Draft[] {
  return DRAFTS.filter((d) => d.folder === folder);
}
