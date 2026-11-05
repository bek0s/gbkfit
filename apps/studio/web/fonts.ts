// The type of the Studio, the one place where it is set: the fonts (the
// faces loaded, and their families: words for people, sans; titles,
// display; literal things and data, mono), and the roles of its text (its
// type scale), each a weight, a size, a line height and a family. Every
// text of the page has a role, as the CSS variable --text-<role> (a value
// of the property font; e.g. font: var(--text-label), in styles.css), and
// Mantine's sizes are those of the roles (see theme.ts). Two weights, 400
// and 600 (some texts make their role bold, e.g. a name in a list); text in
// sentence case, but for the overlines, in capitals spaced by --caps. To
// change a font: its face here (the files of @fontsource), and its family;
// to change how a kind of text looks: its role.
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/600.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-sans-condensed/600.css';

export const FONTS = {
  sans: '"IBM Plex Sans", "Helvetica Neue", Helvetica, sans-serif',
  display: '"IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, Menlo, monospace',
};

type Role = { weight: 400 | 600; size: number; line: number; family: keyof typeof FONTS };

export const TYPE = {
  // the title of a screen (e.g. the name of a block)
  title: { weight: 600, size: 20, line: 1.25, family: 'display' },
  // the title of a dialog or of an empty stage, a step, the logo
  subtitle: { weight: 600, size: 16, line: 1.25, family: 'display' },
  // the title of a section (e.g. of a dock, a figure, a setting)
  heading: { weight: 600, size: 13, line: 1.3, family: 'display' },
  // text to read, the rows of lists and menus, tabs
  body: { weight: 400, size: 13, line: 1.45, family: 'sans' },
  // the text of controls (buttons, fields, choices), labels, help
  label: { weight: 400, size: 12, line: 1.45, family: 'sans' },
  // what is said of a thing beside it (e.g. its description, a note)
  caption: { weight: 400, size: 11, line: 1.35, family: 'sans' },
  // the name over a group, in capitals
  overline: { weight: 600, size: 11, line: 1.2, family: 'mono' },
  // literal things and values: names of files and keys, numbers
  data: { weight: 400, size: 12, line: 1.4, family: 'mono' },
  // values beside others: units, counts, ticks, readouts
  'data-small': { weight: 400, size: 11, line: 1.35, family: 'mono' },
} satisfies Record<string, Role>;

// The spacing of letters in capitals (the overlines, the logo)
export const CAPS = '0.08em';

// The CSS variables of the type: the families (--sans, --display,
// --mono), the roles (--text-<role>), and the spacing of capitals (--caps)
export function typeVariables(): Record<string, string> {
  return {
    ...Object.fromEntries(Object.entries(FONTS).map(([name, family]) => [`--${name}`, family])),
    ...Object.fromEntries(
      Object.entries(TYPE).map(([name, { weight, size, line, family }]) => [`--text-${name}`, `${weight} ${size}px/${line} var(--${family})`]),
    ),
    '--caps': CAPS,
  };
}
