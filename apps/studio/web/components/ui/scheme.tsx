// The look of the page, chosen in the bar from a menu that drops from a
// button of the icon of the scheme: its colour scheme (that of the system,
// light or dark; Mantine keeps the choice, in the storage of the browser,
// and marks the root of the page with the scheme in use,
// data-mantine-color-scheme, which the tokens of styles.css follow), and
// its theme (see theme.ts), each shown by a swatch of its colours in the
// scheme in use; the choices made are checked. The mark of the tab of the
// browser is in the accent of the theme.
import { ActionIcon, Menu, useComputedColorScheme, useMantineColorScheme, type MantineColorScheme } from '@mantine/core';
import { useEffect, type CSSProperties } from 'react';
import { STEPS, THEMES, type Scheme } from '../../palette';
import { setTheme, useTheme } from '../../theme';
import { CheckIcon, DarkIcon, LightIcon, SystemIcon } from '../icons';

const SCHEMES: { value: MantineColorScheme; label: string; icon: () => React.JSX.Element }[] = [
  { value: 'auto', label: 'System', icon: SystemIcon },
  { value: 'light', label: 'Light', icon: LightIcon },
  { value: 'dark', label: 'Dark', icon: DarkIcon },
];

export function LookMenu() {
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const inUse = useComputedColorScheme('dark');
  const theme = useTheme();
  const current = SCHEMES.find(({ value }) => value === colorScheme) ?? SCHEMES[0];
  useEffect(() => markTab(theme[inUse].accent), [theme, inUse]);
  return (
    <Menu position="bottom-end" offset={6} classNames={{ dropdown: 'ui-menu' }}>
      <Menu.Target>
        <ActionIcon
          variant="subtle"
          color="gray"
          size="md"
          aria-label={`The look: ${current.label.toLowerCase()}, ${theme.label}`}
          title={`Look: ${current.label}, ${theme.label}`}
          className="ui-scheme"
        >
          <current.icon />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown>
        <Menu.Label>Scheme</Menu.Label>
        {SCHEMES.map(({ value, label, icon: SchemeIcon }) => (
          <Menu.Item
            key={value}
            leftSection={<SchemeIcon />}
            rightSection={value === colorScheme && <CheckIcon />}
            data-current={value === colorScheme || undefined}
            onClick={() => setColorScheme(value)}
          >
            {label}
          </Menu.Item>
        ))}
        <Menu.Divider />
        <Menu.Label>Theme</Menu.Label>
        {THEMES.map((each) => (
          <Menu.Item
            key={each.id}
            leftSection={<Swatch scheme={each[inUse]} />}
            rightSection={each.id === theme.id && <CheckIcon />}
            data-current={each.id === theme.id || undefined}
            onClick={() => setTheme(each.id)}
          >
            {each.label}
          </Menu.Item>
        ))}
      </Menu.Dropdown>
    </Menu>
  );
}

// (a theme in a scheme: its panel, its accent, and the ramp of its maps)
function Swatch({ scheme }: { scheme: Scheme }) {
  const at = (fraction: number) => scheme[`seq${Math.round(fraction * (STEPS - 1))}` as keyof Scheme];
  const colours = {
    '--swatch-panel': scheme.panel,
    '--swatch-line': scheme.line2,
    '--swatch-accent': scheme['accent-fill'],
    '--swatch-ramp': `linear-gradient(to right, ${at(0.25)}, ${at(0.6)}, ${at(1)})`,
  } as CSSProperties;
  return (
    <span className="ui-swatch" style={colours} aria-hidden>
      <i />
      <b />
    </span>
  );
}

// (the mark of the tab: the mark of the logo, see icons.tsx, in a colour)
function markTab(colour: string) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="${colour}" stroke-width="1.8"><ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-30 12 12)"/><circle cx="12" cy="12" r="2.2"/></svg>`;
  const link = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
  if (link) link.href = `data:image/svg+xml,${encodeURIComponent(svg)}`;
}
