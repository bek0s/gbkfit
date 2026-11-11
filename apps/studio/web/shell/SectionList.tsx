// The list of a section: its blocks only, in groups of one folder each
// (see sections.ts), with a box that filters them by name and a "+ New"
// on each group that makes a new block (see Making). It opens over the
// screen as a section of the rail is clicked or hovered, and closes as
// the mouse leaves it (or Escape, or a block opens); pinned, it stays
// beside the screen, as the list of the section on the stage (Ctrl+B pins
// it or not).
import { useEffect, useState } from 'react';
import { useProject } from '../api/queries';
import { PinIcon } from '../components/icons';
import { Button, IconButton, NavItem, TextField } from '../components/ui/basics';
import { useStudio } from '../studio';
import { sectionNamed, type Group as GroupInfo, type SectionKey } from './sections';

// The blocks a group shows before 'show all'
const SHOWN = 12;

// Whether a section has a list (home and compare have none, yet)
export function hasList(key: SectionKey): boolean {
  return sectionNamed(key).groups.length > 0;
}

export function SectionList({
  section: key,
  pinned,
  onPin,
  onOpen,
}: {
  section: SectionKey;
  pinned: boolean;
  onPin: (pinned: boolean) => void;
  // Called when a block opens (e.g. to close the list over the screen)
  onOpen: () => void;
}) {
  const { project } = useStudio();
  const section = sectionNamed(key);
  const [find, setFind] = useState('');
  const query = find.trim().toLowerCase();
  if (project === null) return null;
  return (
    <nav className="list" aria-label={`The ${section.title.toLowerCase()} of the project`}>
      <div className="list-head">
        <b>{section.title}</b>
        <IconButton label={pinned ? 'Unpin the list: it opens from the rail (Ctrl+B)' : 'Pin the list beside the screen (Ctrl+B)'} pressed={pinned} onClick={() => onPin(!pinned)}>
          <PinIcon />
        </IconButton>
      </div>
      <div className="list-find">
        <TextField search value={find} placeholder="Filter" label={`Filter the ${section.title.toLowerCase()}`} onChange={setFind} />
      </div>
      <div className="list-groups">
        {section.groups.map((group) => (
          <Group key={group.folder} project={project} group={group} find={query} titled={section.groups.length > 1} onOpen={onOpen} />
        ))}
      </div>
    </nav>
  );
}

// A key held down with Ctrl (or Cmd) does something, anywhere on the page
export function useCtrlKey(key: string, act: () => void): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === key) {
        event.preventDefault();
        act();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [key, act]);
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
        <Button
          kind="small"
          onClick={() => {
            startAdding(group.folder);
            onOpen();
          }}
        >
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
