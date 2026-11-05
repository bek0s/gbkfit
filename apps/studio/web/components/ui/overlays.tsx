// Overlays (of Mantine; see theme.ts): a dialog over the page, shown while
// it is mounted (Escape, or a click beside it, closes it; it keeps the
// focus while open, and gives it back as it closes), and a menu that drops
// from a button (its items move with the arrow keys): as wide as it, from
// its left, or, of a small button (e.g. of an icon), as wide as its items,
// from its right, or, of a tall one, at its side. A menu is never taller
// than the window lets it be (at most MENU_MOST), and scrolls; one of many
// items (more than SEARCHED) has a field at its top that finds them.
import { Combobox, Menu, Modal, useCombobox, type PopoverProps } from '@mantine/core';
import { cloneElement, Fragment, useEffect, useState, type ReactElement, type ReactNode } from 'react';

// The most a menu is tall (px), and the most items a menu has without a
// field to find them
const MENU_MOST = 420;
const SEARCHED = 10;

// (the menu on the side of its button with the more room, as tall as that
// room lets it be, less a margin)
const fitted: PopoverProps['middlewares'] = {
  flip: true,
  shift: true,
  size: {
    apply: ({ availableHeight, elements }) => {
      elements.floating.style.maxHeight = `${Math.max(140, Math.min(MENU_MOST, availableHeight - 12))}px`;
    },
  },
};

// (wide: for more than a form, e.g. the settings)
export function Dialog({ title, onClose, wide, children }: { title: string; onClose: () => void; wide?: boolean; children: ReactNode }) {
  return (
    <Modal opened onClose={onClose} title={title} size={wide ? 640 : 'md'} yOffset="12vh" className="ui-dialog">
      {children}
    </Modal>
  );
}

// An item of a menu: an action (current: the one in use; with a picture
// after its label, at the end of the item, e.g. a swatch), or the label of
// the items after it; an action can say what it is under its label, and
// be disabled (its description then says why)
export type MenuEntry =
  | { label: string; onClick: () => void; current?: boolean; picture?: ReactNode; description?: string; disabled?: boolean }
  | { section: string };

export function DropMenu({
  target,
  entries,
  label,
  at = 'start',
  width,
  returnFocus = true,
}: {
  target: ReactElement;
  entries: MenuEntry[];
  label: string;
  // (from its left, its right, or, of a tall button, at its left from its
  // top)
  at?: 'start' | 'end' | 'side';
  // Its width (px), for items wider than the button, e.g. with what each is
  width?: number;
  // Whether the target has the focus again as it closes (not when an entry
  // gives it to something else, e.g. an input)
  returnFocus?: boolean;
}) {
  const position = at === 'side' ? 'left-start' : at === 'end' ? 'bottom-end' : 'bottom-start';
  if (entries.filter((entry) => !('section' in entry)).length > SEARCHED) {
    return <SearchMenu target={target} entries={entries} label={label} position={position} width={width ?? (at === 'start' ? 'target' : 280)} />;
  }
  return (
    <Menu
      position={position}
      width={width ?? (at === 'start' ? 'target' : undefined)}
      offset={4}
      returnFocus={returnFocus}
      middlewares={fitted}
      classNames={{ dropdown: 'ui-menu ui-menu-scroll' }}
      aria-label={label}
    >
      <Menu.Target>{target}</Menu.Target>
      <Menu.Dropdown>
        {entries.map((entry, i) =>
          'section' in entry ? (
            <Menu.Label key={`section-${i}`}>{entry.section}</Menu.Label>
          ) : (
            <Menu.Item key={`${entry.label}-${i}`} onClick={entry.onClick} disabled={entry.disabled} rightSection={entry.picture} data-current={entry.current || undefined}>
              {entry.label}
              {entry.description && <div className="ui-menu-description">{entry.description}</div>}
            </Menu.Item>
          ),
        )}
      </Menu.Dropdown>
    </Menu>
  );
}

type Action = Exclude<MenuEntry, { section: string }>;

// A menu of many items, with a field at its top that finds them (by their
// labels and what they are): typed in as it opens, the arrow keys move
// over those found, and Enter chooses one
function SearchMenu({
  target,
  entries,
  label,
  position,
  width,
}: {
  target: ReactElement;
  entries: MenuEntry[];
  label: string;
  position: 'left-start' | 'bottom-end' | 'bottom-start';
  width: number | 'target';
}) {
  const [query, setQuery] = useState('');
  const combobox = useCombobox({
    onDropdownClose: () => {
      combobox.resetSelectedOption();
      setQuery('');
    },
    onDropdownOpen: () => combobox.focusSearchInput(),
  });
  // (the first found ready to be chosen, by Enter, as it opens and as the
  // field changes)
  const opened = combobox.dropdownOpened;
  useEffect(() => {
    if (opened) combobox.selectFirstOption();
  }, [opened, query, combobox]);
  // (the items by their sections, those found)
  const actions: Action[] = [];
  const groups: { section: string | null; items: number[] }[] = [];
  for (const entry of entries) {
    if ('section' in entry) groups.push({ section: entry.section, items: [] });
    else {
      if (groups.length === 0) groups.push({ section: null, items: [] });
      groups[groups.length - 1].items.push(actions.push(entry) - 1);
    }
  }
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const found = (action: Action) => words.every((word) => `${action.label} ${action.description ?? ''}`.toLowerCase().includes(word));
  const shown = groups.map((group) => ({ ...group, items: group.items.filter((i) => found(actions[i])) })).filter((group) => group.items.length > 0);
  const options = (items: number[]) =>
    items.map((i) => {
      const action = actions[i];
      return (
        <Combobox.Option key={i} value={String(i)} disabled={action.disabled} data-current={action.current || undefined}>
          <span className="ui-search-option">
            {action.label}
            {action.picture}
          </span>
          {action.description && <div className="ui-menu-description">{action.description}</div>}
        </Combobox.Option>
      );
    });
  return (
    <Combobox
      store={combobox}
      position={position}
      width={width}
      offset={4}
      middlewares={fitted}
      classNames={{ dropdown: 'ui-menu ui-search-menu' }}
      onOptionSubmit={(value) => {
        combobox.closeDropdown();
        actions[Number(value)].onClick();
      }}
    >
      <Combobox.Target withAriaAttributes={false}>{cloneElement(target as ReactElement<{ onClick?: () => void }>, { onClick: () => combobox.toggleDropdown() })}</Combobox.Target>
      <Combobox.Dropdown aria-label={label}>
        <Combobox.Search
          value={query}
          onChange={(event) => setQuery(event.currentTarget.value)}
          placeholder="Find"
          aria-label={`Find in: ${label}`}
        />
        <Combobox.Options className="ui-search-options">
          {shown.length === 0 ? (
            <Combobox.Empty>Nothing found</Combobox.Empty>
          ) : (
            shown.map((group, g) =>
              group.section ? (
                <Combobox.Group key={g} label={group.section}>
                  {options(group.items)}
                </Combobox.Group>
              ) : (
                <Fragment key={g}>{options(group.items)}</Fragment>
              ),
            )
          )}
        </Combobox.Options>
      </Combobox.Dropdown>
    </Combobox>
  );
}
