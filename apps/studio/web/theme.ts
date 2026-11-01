// The look of the components of Mantine (see components/ui), made that of
// the page: its fonts, its accent as their primary colour, and its tokens
// (styles.css) as their colours, in the light and the dark scheme alike
// (the tokens change with the scheme of the system).
import { createTheme, type CSSVariablesResolver } from '@mantine/core';

export const theme = createTheme({
  fontFamily: '"IBM Plex Sans", "Helvetica Neue", Helvetica, sans-serif',
  fontFamilyMonospace: '"IBM Plex Mono", ui-monospace, Menlo, monospace',
  headings: { fontFamily: '"IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif' },
  fontSizes: { xs: '11.5px', sm: '12.5px', md: '13px', lg: '15px', xl: '18px' },
  // (the accent of the page, from light to dark: #e8b04b in the dark
  // scheme, #a8700f in the light one)
  colors: {
    accent: ['#fff8e6', '#fdedc8', '#f8db94', '#f2c75c', '#e8b04b', '#d99a2c', '#c08420', '#a8700f', '#8a5b0b', '#6b4607'],
  },
  primaryColor: 'accent',
  primaryShade: { light: 7, dark: 4 },
  defaultRadius: 'sm',
  cursorType: 'pointer',
});

// (the colours of the components are the tokens of the page)
const tokens = {
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

// (in the dark scheme, inputs and their borders take shades of dark)
export const resolver: CSSVariablesResolver = () => ({
  variables: {},
  light: tokens,
  dark: { ...tokens, '--mantine-color-dark-6': 'var(--panel2)', '--mantine-color-dark-4': 'var(--line2)' },
});
