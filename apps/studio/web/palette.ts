// The colours of the Studio, in its two schemes: the one place where they
// are set. Each becomes a CSS variable of the page of its name (--bg,
// --accent, ...; see theme.ts), in the scheme in use, which styles.css and
// the maps (components/colours.ts) use; and Mantine is given them too (its
// accent and its shades of dark; see theme.ts).

// The tokens of a scheme: the background, the panels on it (and those
// raised over them), their lines; the text (and quieter text); the accent
// (and the text on it), success and error; a selection, a shadow; the
// dimming of the pixels out of a crop, and a hint over a map; and the ramps
// of the maps, sequential (seq0 to seq3) and diverging (map-neg2 to
// map-pos2)
export interface Scheme {
  bg: string;
  panel: string;
  panel2: string;
  raise: string;
  line: string;
  line2: string;
  fg: string;
  fg2: string;
  muted: string;
  faint: string;
  accent: string;
  'on-accent': string;
  ok: string;
  err: string;
  sel: string;
  shadow: string;
  dim: string;
  'hint-bg': string;
  'hint-fg': string;
  seq0: string;
  seq1: string;
  seq2: string;
  seq3: string;
  'map-neg2': string;
  'map-neg': string;
  'map-mid': string;
  'map-pos': string;
  'map-pos2': string;
}

export const dark: Scheme = {
  bg: '#0a0e15',
  panel: '#0f1520',
  panel2: '#141b29',
  raise: '#182032',
  line: '#212a3b',
  line2: '#2c374c',
  fg: '#e6eaf2',
  fg2: '#c4cad6',
  muted: '#96a0b5',
  faint: '#7b869c',
  accent: '#e8b04b',
  'on-accent': '#1d1405',
  ok: '#43d08f',
  err: '#ff7676',
  sel: 'rgba(232, 176, 75, .13)',
  shadow: 'rgba(0, 0, 0, .55)',
  dim: 'rgba(10, 14, 21, .55)',
  'hint-bg': 'rgba(10, 14, 21, .7)',
  'hint-fg': '#ffffff',
  seq0: '#0f1520',
  seq1: '#6b2f7a',
  seq2: '#e8724b',
  seq3: '#fbe3a0',
  'map-neg2': '#a7c8ff',
  'map-neg': '#3d7cf5',
  'map-mid': '#1a2233',
  'map-pos': '#ef4f5c',
  'map-pos2': '#ffb3aa',
};

export const light: Scheme = {
  bg: '#eef1f5',
  panel: '#ffffff',
  panel2: '#f5f7fa',
  raise: '#ffffff',
  line: '#dbe0e8',
  line2: '#c6cdd9',
  fg: '#141a26',
  fg2: '#2b3344',
  muted: '#566075',
  faint: '#6b7487',
  accent: '#a8700f',
  'on-accent': '#ffffff',
  ok: '#12804f',
  err: '#c62f37',
  sel: 'rgba(168, 112, 15, .1)',
  shadow: 'rgba(20, 30, 50, .22)',
  dim: 'rgba(10, 14, 21, .55)',
  'hint-bg': 'rgba(10, 14, 21, .7)',
  'hint-fg': '#ffffff',
  seq0: '#ffffff',
  seq1: '#f2c48d',
  seq2: '#c8503a',
  seq3: '#3b1240',
  'map-neg2': '#16357a',
  'map-neg': '#2f69d6',
  'map-mid': '#eef1f5',
  'map-pos': '#d6333c',
  'map-pos2': '#7d151b',
};

// The shades of the accent, from light to dark, that Mantine needs (ten):
// the accent of the dark scheme is the fifth, that of the light one the
// eighth (see theme.ts)
// The colour maps that can be chosen for maps of values (see
// components/colours.ts), besides that of the scheme (seq0 to seq3): the
// same in both schemes, from low to high. Each lightens steadily from end
// to end, so that it reads alike with colour blindness, and in grey.
// Viridis and cividis are those of matplotlib, at tenths (cividis made to
// look the same with the commonest colour blindness, red and green).
export const colourMaps = {
  grey: ['#000000', '#ffffff'],
  viridis: ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c', '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725'],
  cividis: ['#00224e', '#083370', '#35456c', '#4f576c', '#666970', '#7d7c78', '#948e77', '#aea371', '#c8b866', '#e5cf52', '#fee838'],
};

export const accentShades = [
  '#fff8e6',
  '#fdedc8',
  '#f8db94',
  '#f2c75c',
  dark.accent,
  '#d99a2c',
  '#c08420',
  light.accent,
  '#8a5b0b',
  '#6b4607',
];

// The CSS variables of a scheme (--bg, ...)
export function variables(scheme: Scheme): Record<string, string> {
  return Object.fromEntries(Object.entries(scheme).map(([name, value]) => [`--${name}`, value]));
}
