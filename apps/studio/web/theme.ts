// The look of the components of Mantine (see components/ui), made that of
// the page in the theme chosen (palette.ts): its type (fonts.ts, also the
// CSS variables of the page: the fonts, --sans, --display, --mono, and the
// roles of text, --text-<role>; Mantine's sizes those of the roles, its
// compact controls, xs, of the text of controls), and its colours in
// the scheme in use: the tokens of the page as CSS variables, its accent
// as their primary colour, its error as their red, its greys as those
// Mantine draws its dark scheme (dark) and its light one (gray) with (no
// grey of another palette shows), and the colours of its components those
// tokens. The theme chosen, and the colour scheme, are preferences of the
// user (see shell/preferences.ts); the theme marks the root of the page
// (data-theme), and Mantine marks it with the scheme in use
// (data-mantine-color-scheme), which what draws with the tokens follows
// (see useScheme in components/colours.ts).
import { Button, Checkbox, createTheme, Slider, type CSSVariablesResolver, type MantineColorScheme, type MantineColorSchemeManager, type MantineColorsTuple } from '@mantine/core';
import { FONTS, TYPE, typeVariables } from './fonts';
import { THEMES, variables, type Theme } from './palette';
import { followPreferences, preference, setPreference, usePreference } from './shell/preferences';

const tuple = (colours: string[]) => colours as unknown as MantineColorsTuple;
const px = (size: number) => `${size}px`;

export function mantineTheme({ dark, greys, shades }: Theme) {
  return createTheme({
    fontFamily: FONTS.sans,
    fontFamilyMonospace: FONTS.mono,
    headings: { fontFamily: FONTS.display },
    fontSizes: { xs: px(TYPE.label.size), sm: px(TYPE.body.size), md: px(TYPE.body.size), lg: px(TYPE.subtitle.size), xl: px(TYPE.title.size) },
    lineHeights: { xs: `${TYPE.label.line}`, sm: `${TYPE.body.line}`, md: `${TYPE.body.line}`, lg: `${TYPE.subtitle.line}`, xl: `${TYPE.title.line}` },
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
    variables: typeVariables(),
    dark: { ...variables(dark), ...components },
    light: { ...variables(light), ...components },
  });
}

// The theme chosen (the first if none, or one no longer known)
const isText = (value: unknown): value is string => typeof value === 'string';
const themeOf = (id: string) => THEMES.find((theme) => theme.id === id) ?? THEMES[0];
const chosenTheme = () => themeOf(preference('theme', isText, THEMES[0].id));

export function setTheme(id: string): void {
  setPreference('theme', id);
}

export function useTheme(): Theme {
  return themeOf(usePreference('theme', isText, THEMES[0].id));
}

// (the root marked with the theme, as it changes)
const markRoot = () => (document.documentElement.dataset.theme = chosenTheme().id);
markRoot();
followPreferences(markRoot);

// The colour scheme chosen (that of the system, 'auto', if none), for
// Mantine, which sets it and follows it as it changes in another page
const SCHEMES: readonly MantineColorScheme[] = ['auto', 'light', 'dark'];
const isScheme = (value: unknown): value is MantineColorScheme => SCHEMES.includes(value as MantineColorScheme);
const chosenScheme = () => preference('scheme', isScheme, 'auto');
let unfollowScheme = () => {};

export const schemeManager: MantineColorSchemeManager = {
  get: (fallback) => preference('scheme', isScheme, fallback),
  // (not again as it is told the scheme of another page)
  set: (scheme) => scheme !== chosenScheme() && setPreference('scheme', scheme),
  subscribe: (onUpdate) => {
    let last = chosenScheme();
    unfollowScheme = followPreferences(() => {
      if (chosenScheme() !== last) onUpdate((last = chosenScheme()));
    });
  },
  unsubscribe: () => unfollowScheme(),
  clear: () => setPreference('scheme', null),
};
