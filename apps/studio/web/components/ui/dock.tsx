// The dock of a screen: the panel at the side of its view (as the
// inspector of the galaxy screen), full height, with its own scroll, of
// sections one under the other. A section has a head (its title, a short
// note, and actions) whose title folds it, as the page remembers by its
// id; a section that needs the user (e.g. a choice that stops an action)
// is marked, and its head says so.
import { UnstyledButton } from '@mantine/core';
import type { ReactNode } from 'react';
import { useRemembered } from '../../shell/remember';

export function Dock({ children, label }: { children: ReactNode; label: string }) {
  return (
    <aside className="dock" aria-label={label}>
      {children}
    </aside>
  );
}

export function DockSection({
  id,
  title,
  note,
  actions,
  needed,
  children,
}: {
  // The id of the section in the page (its element, and its fold)
  id: string;
  title: string;
  note?: ReactNode;
  actions?: ReactNode;
  // Whether it needs the user before the screen can go on
  needed?: boolean;
  children: ReactNode;
}) {
  const [folded, setFolded] = useRemembered(`dock:${id}`, false);
  // (a section that needs the user is never folded away)
  const open = !folded || needed === true;
  return (
    <section id={id} className="dock-section" data-needed={needed || undefined} data-folded={!open || undefined}>
      <header className="dock-head">
        <UnstyledButton className="dock-title" aria-expanded={open} onClick={() => setFolded(open)}>
          <span className="ui-chevron" data-opened={open}>
            ▶
          </span>
          <h2>{title}</h2>
          {needed && <span className="dock-needed">needed</span>}
        </UnstyledButton>
        {note && <span className="dock-note">{note}</span>}
        {open && actions && <span className="dock-actions">{actions}</span>}
      </header>
      {open && <div className="dock-body">{children}</div>}
    </section>
  );
}

// Brings a section of the dock into view, and flashes it
export function showSection(id: string) {
  const section = document.getElementById(id);
  if (!section) return;
  section.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  section.classList.remove('dock-flash');
  // (restarted: a reflow between the two)
  void section.offsetWidth;
  section.classList.add('dock-flash');
}

// What a screen needs before it can go on (the first, and how many more),
// leading to where it is given: a section of the dock (or a field, which
// takes the focus)
export interface Need {
  text: string;
  // The id of the element where it is given
  at: string;
}

export function NeedsNote({ needs }: { needs: Need[] }) {
  if (needs.length === 0) return null;
  const [first, ...others] = needs;
  const go = () => {
    const element = document.getElementById(first.at);
    if (element instanceof HTMLInputElement) element.focus();
    else showSection(first.at);
  };
  return (
    <button className="ds-need" title={needs.map((need) => need.text).join('\n')} onClick={go}>
      <span className="ds-need-text">{first.text}</span>
      {others.length > 0 && <span className="ds-need-more">+{others.length}</span>}
    </button>
  );
}
