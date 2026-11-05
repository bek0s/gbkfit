// The look of the components of Mantine (see components/ui), made that of
// the page in the theme chosen (palette.ts): the fonts (fonts.ts, also the
// CSS variables of the page: --sans, --display, --mono), and its colours in
// the scheme in use: the tokens of the page as CSS variables, its accent
// as their primary colour, its error as their red, its greys as those
// Mantine draws its dark scheme (dark) and its light one (gray) with (no
// grey of another palette shows), and the colours of its components those
// tokens. The theme chosen is remembered in this browser, and marks the
// root of the page (data-theme), which what draws with the tokens follows
// (see useScheme in components/colours.ts).
import { Button, Checkbox, createTheme, Slider, type CSSVariablesResolver, type MantineColorsTuple } from '@mantine/core';
import { useSyncExternalStore } from 'react';
import { FONTS } from './fonts';
import { THEMES, variables, type Theme } from './palette';
import { recall, remember } from './shell/remember';

const tuple = (colours: string[]) => colours as unknown as MantineColorsTuple;

export function mantineTheme({ dark, greys, shades }: Theme) {
  return createTheme({
    fontFamily: FONTS.sans,
    fontFamilyMonospace: FONTS.mono,
    headings: { fontFamily: FONTS.display },
    fontSizes: { xs: '11.5px', sm: '12.5px', md: '13px', lg: '15px', xl: '18px' },
    colors: {
      accent: tuple(shades.accent),
      err: tuple(shades.err),
      // (from the text of the dark scheme to its background)
      dark: tuple([dark.fg, dark.fg2, dark.muted, dark.faint, dark.line2, dark.line, dark.raise, dark.panel2, dark.panel, dark.bg]),
      gray: tuple(greys),
    },
    primaryColor: 'accent',
    primaryShade: { light: 7, dark: 4 },
    black: dark['on-accent'],
    defaultRadius: 'sm',
    cursorType: 'pointer',
    // (the text on filled controls: that of the palette in the scheme in
    // use, on-accent and on-err; Mantine's own is worked out once, from the
    // accent by day, and so is white by night, on fills light by night;
    // and a slider in the accent as a mark, its fill, Mantine's filled
    // colour, being bright by day in some themes)
    components: {
      Button: Button.extend({
        vars: (_theme, props) => ({ root: props.variant === 'filled' ? { '--button-color': props.color === 'err' ? 'var(--on-err)' : 'var(--on-accent)' } : {} }),
      }),
      Checkbox: Checkbox.extend({ vars: () => ({ root: { '--checkbox-icon-color': 'var(--on-accent)' } }) }),
      Slider: Slider.extend({ vars: () => ({ root: { '--slider-color': 'var(--accent)' } }) }),
    },
  });
}

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
  '--mantine-color-anchor': 'var(--accent-text)',
  '--mantine-color-accent-text': 'var(--accent-text)',
  '--mantine-color-accent-outline': 'var(--accent-text)',
  '--mantine-color-error': 'var(--err)',
  // (a filled control as hovered: further from its text, see hovered in
  // palette.ts, rather than Mantine's, a shade towards that of the other
  // scheme, which can be nearer its text)
  '--mantine-color-accent-filled-hover': 'var(--accent-hover)',
  '--mantine-color-err-filled-hover': 'var(--err-hover)',
  '--mantine-color-success': 'var(--ok)',
  // (the light variant of the accent, e.g. the row chosen in a list, or a
  // callout: the selection, see through over what is under it, rather than
  // Mantine's own, a darkened accent the same over any surface)
  '--mantine-color-accent-light': 'var(--sel)',
  '--mantine-color-accent-light-hover': 'var(--sel-hover)',
  '--mantine-color-accent-light-color': 'var(--accent-text)',
};

export function resolver({ dark, light }: Theme): CSSVariablesResolver {
  return () => ({
    variables: { '--sans': FONTS.sans, '--display': FONTS.display, '--mono': FONTS.mono },
    dark: { ...variables(dark), ...components },
    light: { ...variables(light), ...components },
  });
}

// The theme chosen, remembered in this browser (the first if none, or one
// no longer known), and those that follow it
const themeOf = (id: string) => THEMES.find((theme) => theme.id === id) ?? THEMES[0];

let chosen = themeOf(recall('theme', THEMES[0].id));
const followers = new Set<() => void>();
document.documentElement.dataset.theme = chosen.id;

export function setTheme(id: string): void {
  chosen = themeOf(id);
  remember('theme', chosen.id);
  document.documentElement.dataset.theme = chosen.id;
  followers.forEach((onChange) => onChange());
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    (onChange) => {
      followers.add(onChange);
      return () => followers.delete(onChange);
    },
    () => chosen,
  );
}
