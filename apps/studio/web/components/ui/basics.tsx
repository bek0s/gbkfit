// The basic controls (of Mantine; see theme.ts): buttons of a few kinds,
// buttons of an icon, a close button, a row of a list to navigate (a
// block, a view, a file), text fields, a callout, and the tabs of the bar.
import { ActionIcon, Alert, Button as MantineButton, CloseButton, NavLink, Tabs, TextInput } from '@mantine/core';
import type { ComponentPropsWithRef, ReactNode } from 'react';

// A button: primary (the main action of a screen), danger (an action that
// cannot be undone, e.g. delete), plain, quiet (of a bar), or a small
// action beside a value (e.g. Import, clear). Other props
// (and its ref) go to the button, so that it can be the target of a menu.
export function Button({
  kind = 'plain',
  onClick,
  disabled,
  title,
  right,
  full,
  submit,
  children,
  ...rest
}: {
  kind?: 'primary' | 'danger' | 'plain' | 'quiet' | 'small';
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  disabled?: boolean;
  title?: string;
  // What it has at its right (e.g. a chevron)
  right?: ReactNode;
  // Whether it is as wide as its place, its text at the left
  full?: boolean;
  // Whether it submits its form
  submit?: boolean;
  children: ReactNode;
} & Omit<ComponentPropsWithRef<'button'>, 'onClick' | 'disabled' | 'title' | 'type' | 'children'>) {
  const variant = { primary: 'filled', danger: 'filled', plain: 'default', quiet: 'subtle', small: 'subtle' }[kind];
  return (
    <MantineButton
      {...rest}
      type={submit ? 'submit' : 'button'}
      variant={variant}
      color={kind === 'danger' ? 'red' : undefined}
      size={kind === 'small' ? 'compact-xs' : 'xs'}
      onClick={onClick}
      disabled={disabled}
      title={title}
      rightSection={right}
      fullWidth={full}
      justify={full ? 'space-between' : undefined}
      className={`ui-button ui-button-${kind}`}
    >
      {children}
    </MantineButton>
  );
}

// A button of an icon (or a sign), named for screen readers
export function IconButton({
  label,
  onClick,
  pressed,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  pressed?: boolean;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <ActionIcon
      variant={pressed ? 'light' : 'subtle'}
      color={pressed ? undefined : 'gray'}
      size="sm"
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      disabled={disabled}
      onClick={onClick}
      className="ui-icon-button"
    >
      {children}
    </ActionIcon>
  );
}

// A button that removes or closes, named for screen readers
export function Close({ label, onClick }: { label: string; onClick: () => void }) {
  return <CloseButton size="sm" aria-label={label} title={label} onClick={onClick} className="ui-close" />;
}

// A row of a list to navigate or choose from: its label (and what it has
// at its left and right), shown as the current one if active; a group's
// row shows whether it is open
export function NavItem({
  label,
  active,
  onClick,
  left,
  right,
  disabled,
  title,
  opened,
  mono,
}: {
  label: ReactNode;
  active?: boolean;
  onClick: () => void;
  left?: ReactNode;
  right?: ReactNode;
  disabled?: boolean;
  title?: string;
  // Whether the group of the row is open (none: not a group)
  opened?: boolean;
  // Whether its label is a name (in the font of names)
  mono?: boolean;
}) {
  return (
    <NavLink
      component="button"
      label={label}
      active={active}
      onClick={onClick}
      leftSection={opened === undefined ? left : <span className="ui-chevron" data-opened={opened}>▶</span>}
      rightSection={right}
      disabled={disabled}
      title={title}
      aria-current={active ? 'true' : undefined}
      aria-expanded={opened}
      variant="light"
      className={mono ? 'ui-nav ui-nav-mono' : 'ui-nav'}
    />
  );
}

// A field of text (search: with the look of a search box), its label
// shown above it, or read by screen readers alone
export function TextField({
  value,
  onChange,
  placeholder,
  label,
  showLabel,
  search,
  autoFocus,
  onKeyDown,
  onBlur,
  mono,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label: string;
  showLabel?: boolean;
  search?: boolean;
  autoFocus?: boolean;
  onKeyDown?: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  onBlur?: () => void;
  mono?: boolean;
}) {
  return (
    <TextInput
      type={search ? 'search' : 'text'}
      value={value}
      onChange={(event) => onChange(event.currentTarget.value)}
      placeholder={placeholder}
      label={showLabel ? label : undefined}
      aria-label={showLabel ? undefined : label}
      autoFocus={autoFocus}
      onKeyDown={onKeyDown}
      onBlur={onBlur}
      size="xs"
      className={mono ? 'ui-text ui-text-mono' : 'ui-text'}
    />
  );
}

// A chip naming a thing (e.g. a part of the selection): its text, with a
// button that removes it if it can go
export function Chip({
  children,
  title,
  onRemove,
  removeLabel,
}: {
  children: ReactNode;
  title?: string;
  onRemove?: () => void;
  removeLabel?: string;
}) {
  return (
    <span className="ui-chip" title={title}>
      <span className="ui-chip-text">{children}</span>
      {onRemove && <CloseButton size="xs" aria-label={removeLabel} title={removeLabel} onClick={onRemove} className="ui-chip-remove" />}
    </span>
  );
}

// A note that asks for attention (e.g. a fix the data take)
export function Callout({ children }: { children: ReactNode }) {
  return (
    <Alert variant="light" color="accent" radius="sm" className="ui-callout">
      {children}
    </Alert>
  );
}

// The tabs of a bar: one is current (none: null); a tab can be off, and
// have a mark at its right (e.g. a dot)
export function BarTabs<T extends string>({
  tabs,
  value,
  onChange,
  label,
}: {
  tabs: { value: T; label: string; disabled?: boolean; mark?: ReactNode }[];
  value: T | null;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <Tabs value={value} onChange={(next) => next && onChange(next as T)} className="ui-tabs" aria-label={label}>
      <Tabs.List>
        {tabs.map((tab) => (
          <Tabs.Tab key={tab.value} value={tab.value} disabled={tab.disabled} rightSection={tab.mark}>
            {tab.label}
          </Tabs.Tab>
        ))}
      </Tabs.List>
    </Tabs>
  );
}
