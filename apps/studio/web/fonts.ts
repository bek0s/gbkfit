// The fonts of the Studio, the one place where they are set: the faces
// loaded, and the families of each role: words for people (sans), titles
// (display), and literal things and data (mono). They become the CSS
// variables of the page (--sans, --display, --mono), which every style of
// styles.css uses, and Mantine's fonts (see theme.ts). To change a font:
// its face here (the files of @fontsource), and its family.
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
