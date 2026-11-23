// The pieces of the sheets of the screens that edit a model (the pieces
// of a lens; the components and the parts of the mass of a galaxy): a
// block of them under its head (its title, how many, what unfolds or
// folds them all and what adds one; see FoldAll, AddMenu), each a line
// that unfolds into its body (see PieceHead).
import { useState, type ReactNode } from 'react';
import { DropMenu, type MenuEntry } from './overlays';

const stop = (event: { stopPropagation: () => void }) => event.stopPropagation();

// The line of a piece: its chevron, which folds or unfolds it, and only
// that; its colour (that of the piece, --piece), name and type, and,
// folded, what it is in a few words; a badge (its cell kept when empty);
// its eye (or the room for one, so that nothing moves as it comes); and
// its menu, which renames it (onRename; the field of its name in its
// place) and does what else is given. A click on the line selects it
// (unfolded, if it was folded) or, if it is selected, selects it no
// longer (see onSelect). What folds or selects it is given the line.
export function PieceHead({
  kind,
  name,
  type,
  summary,
  badge,
  badgeTitle,
  open,
  selected,
  onFold,
  onSelect,
  onRename,
  eye,
  menu,
  menuTitle,
}: {
  // What it is, for the label of its menu (e.g. 'piece')
  kind: string;
  name: string;
  type: string;
  summary: ReactNode;
  badge?: ReactNode;
  badgeTitle?: string;
  open: boolean;
  selected: boolean;
  onFold: (line: HTMLElement) => void;
  onSelect: (line: HTMLElement) => void;
  onRename: (name: string) => void;
  eye?: ReactNode;
  menu: MenuEntry[];
  menuTitle: string;
}) {
  const [renaming, setRenaming] = useState(false);
  return (
    <div
      className="sh-piece-head"
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={(event) => onSelect(event.currentTarget)}
      onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.click()}
    >
      <button
        type="button"
        className="sh-fold"
        aria-expanded={open}
        aria-label={open ? `Fold ${name}` : `Unfold ${name}`}
        title={open ? 'Fold it' : 'Unfold it'}
        onClick={(event) => {
          event.stopPropagation();
          onFold(event.currentTarget.parentElement!);
        }}
      >
        <svg className="sh-chev" viewBox="0 0 16 16" aria-hidden="true">
          <path d="M6 4l4 4-4 4" />
        </svg>
      </button>
      <span className="sh-dot" />
      <span className="sh-piece-text">
        <span className="sh-piece-name">
          {renaming ? (
            <span onClick={stop} onKeyDown={stop}>
              <NameText
                value={name}
                onCommit={(to) => {
                  setRenaming(false);
                  onRename(to);
                }}
                onCancel={() => setRenaming(false)}
              />
            </span>
          ) : (
            <b>{name}</b>
          )}
          <span className="sh-piece-type">{type}</span>
        </span>
        {!open && <span className="sh-sum">{summary}</span>}
      </span>
      <span className="sh-badge" title={badgeTitle}>
        {badge}
      </span>
      {eye ?? <span className="sh-eye-room" />}
      <span onClick={stop} onKeyDown={stop}>
        <DropMenu
          label={`The ${kind} ${name}`}
          width={240}
          at="end"
          // (its rename gives the focus to the field of its name)
          returnFocus={false}
          entries={[{ label: 'Rename', onClick: () => setRenaming(true) }, ...menu]}
          target={
            <button type="button" className="sh-icon" title={menuTitle} aria-label="Its actions">
              <svg viewBox="0 0 16 16">
                <circle cx="3.5" cy="8" r=".9" />
                <circle cx="8" cy="8" r=".9" />
                <circle cx="12.5" cy="8" r=".9" />
              </svg>
            </button>
          }
        />
      </span>
    </div>
  );
}

// A name, as typed: set as it is entered or left, if it is a name (else
// as it was); Escape: as it was
export function NameText({ value, onCommit, onCancel }: { value: string; onCommit: (name: string) => void; onCancel: () => void }) {
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    const name = (text ?? value).trim();
    if (name !== value && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) onCommit(name);
    else onCancel();
    setText(null);
  };
  return (
    <input
      className="sh-name"
      value={text ?? value}
      placeholder="name"
      autoFocus
      onChange={(event) => setText(event.currentTarget.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') {
          setText(null);
          onCancel();
        }
      }}
      aria-label="Its name"
      spellCheck={false}
    />
  );
}

// What unfolds every piece of a block and what folds them all (of a kind,
// e.g. 'piece'), each disabled when there is nothing to do
export function FoldAll({ what, canUnfold, canFold, onUnfold, onFold }: { what: string; canUnfold: boolean; canFold: boolean; onUnfold: () => void; onFold: () => void }) {
  return (
    <>
      <button type="button" className="sh-icon" title={`Unfold every ${what}`} aria-label={`Unfold every ${what}`} disabled={!canUnfold} onClick={onUnfold}>
        <svg viewBox="0 0 16 16">
          <path d="M4 6l4-4 4 4M4 10l4 4 4-4" />
        </svg>
      </button>
      <button type="button" className="sh-icon" title={`Fold every ${what}`} aria-label={`Fold every ${what}`} disabled={!canFold} onClick={onFold}>
        <svg viewBox="0 0 16 16">
          <path d="M4 2l4 4 4-4M4 14l4-4 4 4" />
        </svg>
      </button>
    </>
  );
}

// What adds a piece to a block: a +, a menu of what can be added
export function AddMenu({ label, entries, width = 300 }: { label: string; entries: MenuEntry[]; width?: number }) {
  return (
    <DropMenu
      label={label}
      width={width}
      at="end"
      entries={entries}
      target={
        <button type="button" className="sh-icon sh-add" title={label} aria-label={label}>
          <svg viewBox="0 0 16 16">
            <path d="M8 3v10M3 8h10" />
          </svg>
        </button>
      }
    />
  );
}
