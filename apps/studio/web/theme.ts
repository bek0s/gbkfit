// The look of the components of Mantine (see components/ui), made that of
// the page: its fonts, and its colours (palette.ts) in the scheme in use:
// the tokens of the page as CSS variables, its accent as their primary
// colour, its shades of dark as those Mantine draws its dark scheme with
// (no grey of another palette shows), and the colours of its components
// those tokens.
import { createTheme, type CSSVariablesResolver, type MantineColorsTuple } from '@mantine/core';
import { accentShades, dark, light, variables } from './palette';

export const theme = createTheme({
  fontFamily: '"IBM Plex Sans", "Helvetica Neue", Helvetica, sans-serif',
  fontFamilyMonospace: '"IBM Plex Mono", ui-monospace, Menlo, monospace',
  headings: { fontFamily: '"IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif' },
  fontSizes: { xs: '11.5px', sm: '12.5px', md: '13px', lg: '15px', xl: '18px' },
  colors: {
    accent: accentShades as unknown as MantineColorsTuple,
    // (from the text of the dark scheme to its background)
    dark: [dark.fg, dark.fg2, dark.muted, dark.faint, dark.line2, dark.line, dark.raise, dark.panel2, dark.panel, dark.bg],
  },
  primaryColor: 'accent',
  primaryShade: { light: 7, dark: 4 },
  // (the text on a filled control is dark on a light fill, e.g. the accent
  // of the dark scheme, and white on a dark one)
  autoContrast: true,
  luminanceThreshold: 0.3,
  black: dark['on-accent'],
  defaultRadius: 'sm',
  cursorType: 'pointer',
});

// (the colours of the components are the tokens of the page)
const components = {
  '--mantine-color-body': 'var(--panel)',
  '--mantine-color-text': 'var(--fg)',
  '--mantine-color-dimmed': 'var(--faint)',
  '--mantine-color-default': 'var(--panel)',
  '--mantine-color-default-hover': 'var(--panel2)',
  '--mantine-color-default-color': 'var(--fg)',
  '--mantine-color-default-border': 'var(--line2)',
  '--mantine-color-placeholder': 'var(--faint)',
  '--mantine-color-anchor': 'var(--accent)',
};

export const resolver: CSSVariablesResolver = () => ({
  variables: {},
  dark: { ...variables(dark), ...components },
  light: { ...variables(light), ...components },
});
