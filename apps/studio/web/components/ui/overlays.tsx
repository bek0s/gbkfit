// Overlays (of Mantine; see theme.ts): a dialog over the page, shown while
// it is mounted (Escape, or a click beside it, closes it; it keeps the
// focus while open, and gives it back as it closes), and a menu that drops
// from a button (its items move with the arrow keys): as wide as it, from
// its left, or, of a small button (e.g. of an icon), as wide as its items,
// from its right.
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

export function DropMenu({
  target,
  entries,
  label,
  at = 'start',
  returnFocus = true,
}: {
  target: ReactElement;
  entries: MenuEntry[];
  label: string;
  at?: 'start' | 'end';
  // Whether the target has the focus again as it closes (not when an entry
  // gives it to something else, e.g. an input)
  returnFocus?: boolean;
}) {
  return (
    <Menu
      position={at === 'end' ? 'bottom-end' : 'bottom-start'}
      width={at === 'end' ? undefined : 'target'}
      offset={4}
      returnFocus={returnFocus}
      classNames={{ dropdown: 'ui-menu' }}
      aria-label={label}
    >
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
