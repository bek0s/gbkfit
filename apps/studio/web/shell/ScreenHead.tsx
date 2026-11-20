// The head of a screen, the same height on every screen, its name always
// in the same place: above the name, the way back to the overview of its
// section (an arrow and the overview's name, e.g. "← Datasets", as Apple's
// and GitHub's back links name the page they go to), or what it is (its
// eyebrow, of a block of a section with no overview yet), or nothing (a
// page of the top: Home, an overview); beside the name, what the block is
// in tags (e.g. "Cube"), or a note (e.g. the counts of a project); and its
// actions at the end. (What a block uses and what uses it are no part of
// the head: they belong to the body of its screen.) A screen that edits a block says there whether its
// edits are saved, with undo and redo, Revert and Save (see SaveControls).
import { useEffect, useRef, type ReactNode } from 'react';
import { Button } from '../components/ui/basics';
import { useStudio } from '../studio';
import { sectionNamed, type SectionKey } from './sections';

export function ScreenHead({
  eyebrow,
  title,
  tags = [],
  note,
  up,
  children,
}: {
  // What it is, above its name (of a screen with no way back)
  eyebrow?: ReactNode;
  title: ReactNode;
  // What it is, beside its name (e.g. the kind of a dataset)
  tags?: string[];
  // A word beside the name (e.g. how many blocks a page of them lists)
  note?: ReactNode;
  // The section whose overview the screen goes back to
  up?: SectionKey;
  children?: ReactNode;
}) {
  const { go } = useStudio();
  const overview = up === undefined ? undefined : sectionNamed(up).overview;
  return (
    <div className="screen-head">
      <div className="screen-title">
        {overview && up !== undefined ? (
          <button className="screen-up" onClick={() => go({ kind: 'section', key: up })}>
            <svg className="icon" viewBox="0 0 16 16" aria-hidden>
              <path d="M13 8H3.5M7.5 4 3.5 8l4 4" />
            </svg>
            {overview.title}
          </button>
        ) : (
          eyebrow !== undefined && <div className="eyebrow">{eyebrow}</div>
        )}
        <div className="screen-name">
          {typeof title === 'string' ? <h1>{title}</h1> : title}
          {tags.map((tag) => (
            <span key={tag} className="screen-tag">
              {tag}
            </span>
          ))}
        </div>
      </div>
      {note}
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
