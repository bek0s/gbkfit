// The colour scheme of the page (of Mantine; see theme.ts), chosen in the
// bar: that of the system, light or dark, from a menu that drops from a
// button of the icon of the choice. Mantine keeps the choice (in the
// storage of the browser) and marks the root of the page with the scheme
// in use (data-mantine-color-scheme), which the tokens of styles.css follow.
import { ActionIcon, Menu, useMantineColorScheme, type MantineColorScheme } from '@mantine/core';
import { DarkIcon, LightIcon, SystemIcon } from '../icons';

const SCHEMES: { value: MantineColorScheme; label: string; icon: () => React.JSX.Element }[] = [
  { value: 'auto', label: 'System', icon: SystemIcon },
  { value: 'light', label: 'Light', icon: LightIcon },
  { value: 'dark', label: 'Dark', icon: DarkIcon },
];

export function SchemeMenu() {
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const current = SCHEMES.find(({ value }) => value === colorScheme) ?? SCHEMES[0];
  return (
    <Menu position="bottom-end" offset={6} classNames={{ dropdown: 'ui-menu' }}>
      <Menu.Target>
        <ActionIcon
          variant="subtle"
          color="gray"
          size="md"
          aria-label={`The colour scheme: ${current.label.toLowerCase()}`}
          title={`Colour scheme: ${current.label}`}
          className="ui-scheme"
        >
          <current.icon />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown>
        {SCHEMES.map(({ value, label, icon: SchemeIcon }) => (
          <Menu.Item
            key={value}
            leftSection={<SchemeIcon />}
            data-current={value === colorScheme || undefined}
            onClick={() => setColorScheme(value)}
          >
            {label}
          </Menu.Item>
        ))}
      </Menu.Dropdown>
    </Menu>
  );
}
