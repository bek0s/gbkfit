// Drafts of the screens of galaxies and observations: layouts to compare
// before one is chosen, each reachable from the tree (its section Screen
// drafts). The blocks of a folder open in the layout chosen for it (see
// useLayout), the first by default.

export interface Draft {
  id: string;
  folder: 'galaxies' | 'observations';
  // Its letter, its title and what it puts first
  letter: string;
  title: string;
  text: string;
}

export const DRAFTS: Draft[] = [
  {
    id: 'galaxy-outline',
    folder: 'galaxies',
    letter: 'A',
    title: 'Outline and inspector',
    text: 'The structure of the galaxy as an outline (its components and their parts); the part chosen is edited beside it, with the model at the right.',
  },
  {
    id: 'galaxy-cards',
    folder: 'galaxies',
    letter: 'B',
    title: 'Component cards',
    text: 'A card of each component that says what it is at a glance (its density, motions, light, rings), opened to edit; the radial profiles at the side.',
  },
  {
    id: 'galaxy-canvas',
    folder: 'galaxies',
    letter: 'C',
    title: 'Model first',
    text: 'The model takes the screen (its maps and profiles), with the components and the parameters in a drawer under it: a galaxy is shaped by looking at it.',
  },
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
