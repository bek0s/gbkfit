// The head of a screen, the same on every screen: what it is (its kind,
// and a word on it), its name, the blocks it uses and those that use it,
// and its actions at the end. A screen that edits a block says there
// whether its edits are saved, with undo and redo, Revert and Save (see
// SaveControls).
import { useEffect, useRef, type ReactNode } from 'react';
import { Button } from '../components/ui/basics';

export function ScreenHead({ eyebrow, title, links, children }: { eyebrow: ReactNode; title: ReactNode; links?: ReactNode; children?: ReactNode }) {
  return (
    <div className="screen-head">
      <div className="screen-title">
        <div className="eyebrow">{eyebrow}</div>
        {typeof title === 'string' ? <h1>{title}</h1> : title}
      </div>
      {links}
      {children !== undefined && <div className="head-actions">{children}</div>}
    </div>
  );
}

// The saving of the edits of a screen
export interface Saving {
  dirty: boolean;
  saving: boolean;
  saveError: Error | null;
  save: () => unknown;
  revert: () => void;
  // (of an editor with a history)
  undo?: () => void;
  redo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
}

// Whether the edits of a screen are saved; undo and redo, of an editor
// with a history; Revert, and Save (the primary action, unless the screen
// has another). Ctrl+S saves, Ctrl+Z undoes and Ctrl+Shift+Z (or Ctrl+Y)
// redoes, but in a field of text, whose own undo it is. Of an editor that
// saves each change as it is made (auto), only whether it is saved.
export function SaveControls({ saving: s, primary = true, auto = false }: { saving: Saving; primary?: boolean; auto?: boolean }) {
  useSaveKeys(s, auto);
  const status = s.saving ? 'Saving…' : s.dirty ? 'Not saved' : 'Saved';
  return (
    <>
      {s.undo && !auto && (
        <div className="head-history" role="group" aria-label="History">
          <button onClick={s.undo} disabled={!s.canUndo} title="Undo (Ctrl+Z)" aria-label="Undo">
            <svg className="icon" viewBox="0 0 24 24">
              <path d="M9 14 4 9l5-5" />
              <path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" />
            </svg>
          </button>
          <button onClick={s.redo} disabled={!s.canRedo} title="Redo (Ctrl+Shift+Z)" aria-label="Redo">
            <svg className="icon" viewBox="0 0 24 24">
              <path d="m15 14 5-5-5-5" />
              <path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13" />
            </svg>
          </button>
        </div>
      )}
      <span className="head-status" data-dirty={s.dirty || undefined} role="status">
        {status}
      </span>
      {s.saveError && <span className="error">{s.saveError.message}</span>}
      {!auto && (
        <>
          <Button disabled={!s.dirty} onClick={s.revert}>
            Revert
          </Button>
          <Button kind={primary ? 'primary' : undefined} disabled={!s.dirty || s.saving} onClick={() => void s.save()} title="Save (Ctrl+S)">
            Save
          </Button>
        </>
      )}
    </>
  );
}

// The keys of saving: Ctrl+S, and undo and redo (in no field of text)
function useSaveKeys(s: Saving, auto: boolean) {
  const latest = useRef(s);
  latest.current = s;
  useEffect(() => {
    if (auto) return;
    const onKey = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      const { dirty, save, undo, redo } = latest.current;
      const key = event.key.toLowerCase();
      if (key === 's') {
        event.preventDefault();
        if (dirty) void save();
        return;
      }
      const typing =
        event.target instanceof HTMLElement &&
        event.target.closest('input:not([type=radio]):not([type=checkbox]):not([type=range]):not([type=button]), textarea, [contenteditable]:not([contenteditable=false])');
      if (typing || !undo || !redo) return;
      if (key === 'z' && !event.shiftKey) {
        event.preventDefault();
        undo();
      } else if ((key === 'z' && event.shiftKey) || key === 'y') {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [auto]);
}
