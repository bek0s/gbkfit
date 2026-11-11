// The list of a section: its blocks only, in groups of one folder each
// (see sections.ts), with a box that filters them by name and a "+ New"
// on each group that makes a new block (see Making). Beside the screen (pinned), or hidden;
// hidden, ☰ (or Ctrl+B) opens it over the screen, and opening a block
// closes it. Whether it is pinned is remembered for each section.
import { useEffect, useState } from 'react';
import { useProject } from '../api/queries';
import { ListIcon } from '../components/icons';
import { Button, IconButton, NavItem, TextField } from '../components/ui/basics';
import { useStudio } from '../studio';
import { useRemembered } from './remember';
import { sectionNamed, sectionOf, type Group as GroupInfo, type SectionKey } from './sections';

// The blocks a group shows before 'show all'
const SHOWN = 12;

// Whether the list of the section on the stage is pinned, and how to pin
// it or not (remembered by section)
export function useListPinned(): [boolean, (pinned: boolean) => void] {
  const { screen } = useStudio();
  const section = sectionOf(screen);
  const [hidden, setHidden] = useRemembered<SectionKey[]>('lists-hidden', []);
  return [
    !hidden.includes(section),
    (pinned: boolean) => setHidden(pinned ? hidden.filter((key) => key !== section) : [...hidden, section]),
  ];
}

// Whether a section has a list (home and compare have none, yet)
export function hasList(key: SectionKey): boolean {
  return sectionNamed(key).groups.length > 0;
}

export function SectionList({ flyout, onClose }: { flyout: boolean; onClose: () => void }) {
  const { project, screen } = useStudio();
  const section = sectionNamed(sectionOf(screen));
  const [pinned, setPinned] = useListPinned();
  const [find, setFind] = useState('');
  const query = find.trim().toLowerCase();
  if (project === null) return null;
  return (
    <nav className="list" data-flyout={flyout || undefined} aria-label={`The ${section.title.toLowerCase()} of the project`}>
      <div className="list-head">
        <b>{section.title}</b>
        <IconButton label={pinned ? 'Hide the list (Ctrl+B)' : 'Pin the list (Ctrl+B)'} pressed={pinned} onClick={() => {
          setPinned(!pinned);
          onClose();
        }}>
          <ListIcon />
        </IconButton>
      </div>
      <div className="list-find">
        <TextField search value={find} placeholder="Filter" label={`Filter the ${section.title.toLowerCase()}`} onChange={setFind} />
      </div>
      <div className="list-groups">
        {section.groups.map((group) => (
          <Group key={group.folder} project={project} group={group} find={query} titled={section.groups.length > 1} onOpen={onClose} />
        ))}
      </div>
    </nav>
  );
}

// Ctrl+B pins the list or not
export function useListKey(toggle: () => void): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'b') {
        event.preventDefault();
        toggle();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggle]);
}

function Group({
  project,
  group,
  find,
  titled,
  onOpen,
}: {
  project: string;
  group: GroupInfo;
  // The text to find (lower case; empty: all)
  find: string;
  // Whether it shows its title (a section of one group needs none)
  titled: boolean;
  onOpen: () => void;
}) {
  const { screen, go, startAdding } = useStudio();
  const info = useProject(project);
  const [all, setAll] = useState(false);
  const names = (info.data?.pieces[group.folder] ?? []).filter((name) => name.toLowerCase().includes(find));
  const shown = find || all ? names : names.slice(0, SHOWN);
  return (
    <div className="list-group">
      <div className="list-group-head">
        {titled ? <span>{group.title}</span> : <span />}
        <Button kind="small" onClick={() => startAdding(group.folder)}>
          + New
        </Button>
      </div>
      {shown.map((name) => (
        <div key={name} className="list-item">
          <NavItem
            label={name}
            mono
            active={screen.kind === 'block' && screen.folder === group.folder && screen.name === name}
            onClick={() => {
              go({ kind: 'block', folder: group.folder, name });
              onOpen();
            }}
          />
        </div>
      ))}
      {shown.length === 0 && <p className="list-note">{find ? 'None by that name.' : `No ${group.title.toLowerCase()} yet.`}</p>}
      {shown.length < names.length && (
        <div className="list-more">
          <Button kind="small" onClick={() => setAll(true)}>
            show all {names.length}
          </Button>
        </div>
      )}
    </div>
  );
}
