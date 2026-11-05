// The colours of the Studio: its themes, each in two schemes (dark and
// light), the one place where colours are set: in themes.json, made by the
// generator of the developers (python -m palette, in apps/studio; see
// palette/), or edited by hand, and held to the floors of its audit by the
// tests (tests/test_palette.py). Each token becomes a CSS variable of the
// page of its name (--bg, --accent, ...; see theme.ts), in the theme and
// scheme in use, which styles.css and the maps (components/colours.ts)
// use; and Mantine is given them too (see theme.ts).
import palette from './themes.json';

// The steps of a ramp (seq0 to seq16, div0 to div16)
type Steps<P extends string> = Record<`${P}${0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16}`, string>;

// The tokens of a scheme: the background of the window, the fields (the
// inputs, sunk into the panels), the panels on the background (and those
// raised over them), their lines; the text (and quieter text); the accent
// (what is chosen, focused, or done: as a mark, e.g. a line, an outline, a
// slider; as a fill, e.g. a button, a selection; as text; and the text on
// its fills; by day the marks and fills brighter than the text); success, a
// warning (what is needed, or wrong but not an error) and an error, each as
// text and as a fill or mark (e.g. a badge, a dot, a border), and the text
// on the fills of the warning and the error; the fills of the accent and
// the error as hovered; a
// selection (and as hovered), a shadow; the dimming of the
// pixels out of a crop, and a hint over a map; the ramps of the maps,
// sequential (seq0 to seq16: the colour map of the theme, from low to
// high) and diverging (div0 to div16, about a grey middle); and the colours
// of things told apart by colour
// alone (e.g. the components of a galaxy, their curves: cat1 to cat6)
export type Scheme = {
  bg: string;
  field: string;
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
  'accent-fill': string;
  'accent-text': string;
  'on-accent': string;
  ok: string;
  warn: string;
  err: string;
  'ok-fill': string;
  'warn-fill': string;
  'on-warn': string;
  'err-fill': string;
  'on-err': string;
  'accent-hover': string;
  'err-hover': string;
  sel: string;
  'sel-hover': string;
  shadow: string;
  dim: string;
  'hint-bg': string;
  'hint-fg': string;
  cat1: string;
  cat2: string;
  cat3: string;
  cat4: string;
  cat5: string;
  cat6: string;
} & Steps<'seq'> &
  Steps<'div'>;

export type Kind = 'dark' | 'light';

// A theme: its name, and its schemes; the greys of its light scheme, and
// the shades of its accent and error, as Mantine's ten (from light to dark:
// that of the dark scheme the fifth, that of the light one the eighth; see
// theme.ts)
export interface Theme {
  id: string;
  label: string;
  dark: Scheme;
  light: Scheme;
  greys: string[];
  shades: { accent: string[]; err: string[] };
}


// The themes (the first the default)
export const THEMES = palette.themes as Theme[];

// The number of steps of the ramps of the maps
export const STEPS = Object.keys(THEMES[0].dark).filter((name) => /^seq\d+$/.test(name)).length;

// The colour maps that can be chosen for maps of values (see
// components/colours.ts), besides that of the theme: published ones
// (Matplotlib's, Crameri's, cmocean's), each perceptually uniform, its
// lightness rising steadily, so that it reads alike with colour blindness
// and in grey: by their names, their titles and their colours, from dark
// to light, the same in every theme and scheme
export const colourMaps = palette.maps as Record<string, { title: string; colours: string[] }>;

// The CSS variables of a scheme (--bg, ...)
export function variables(scheme: Scheme): Record<string, string> {
  return Object.fromEntries(Object.entries(scheme).map(([name, value]) => [`--${name}`, value]));
}
