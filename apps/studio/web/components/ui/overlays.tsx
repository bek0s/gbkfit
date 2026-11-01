// Overlays (of Mantine; see theme.ts): a dialog over the page, shown while
// it is mounted (Escape, or a click beside it, closes it; it keeps the
// focus while open, and gives it back as it closes), and a menu that drops
// from a button (its items move with the arrow keys).
import { Menu, Modal } from '@mantine/core';
import type { ReactElement, ReactNode } from 'react';

export function Dialog({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  return (
    <Modal opened onClose={onClose} title={title} size="md" yOffset="12vh" className="ui-dialog">
      {children}
    </Modal>
  );
}

// An item of a menu: an action (current: the one in use), or the label of
// the items after it
export type MenuEntry = { label: string; onClick: () => void; current?: boolean } | { section: string };

export function DropMenu({ target, entries, label }: { target: ReactElement; entries: MenuEntry[]; label: string }) {
  return (
    <Menu position="bottom-start" width="target" offset={4} classNames={{ dropdown: 'ui-menu' }} aria-label={label}>
      <Menu.Target>{target}</Menu.Target>
      <Menu.Dropdown>
        {entries.map((entry, i) =>
          'section' in entry ? (
            <Menu.Label key={`section-${i}`}>{entry.section}</Menu.Label>
          ) : (
            <Menu.Item key={`${entry.label}-${i}`} onClick={entry.onClick} data-current={entry.current || undefined}>
              {entry.label}
            </Menu.Item>
          ),
        )}
      </Menu.Dropdown>
    </Menu>
  );
}
