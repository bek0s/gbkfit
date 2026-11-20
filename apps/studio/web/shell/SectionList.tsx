// The list of a section: its blocks only, in groups of one folder each
// (see sections.ts), with a box that filters them by name and a "+ New"
// that makes a new block (see Making): in the head of a list of one
// group, on each group of the others. Each block says what it is in a
// word, where that helps to tell them apart (a dataset its kind, an
// observation its data, a fit how its latest run went). It opens over the
// screen as a section of the rail is clicked, and closes as the mouse
// leaves it (or Escape, or a block opens); pinned, it stays beside the
// screen, as the list of the section on the stage (Ctrl+B pins it or
// not), as wide as its edge is dragged.
import { useEffect, useState, type KeyboardEvent as KeyEvent, type PointerEvent as DragEvent, type ReactNode } from 'react';
import type { components } from '../api/schema';
import { useProject, useSchema } from '../api/queries';
import { PinIcon } from '../components/icons';
import { titleOf, type Defs } from '../components/SchemaForm';
import { Button, IconButton, NavItem, TextField } from '../components/ui/basics';
import { KINDS } from '../screens/dataset/model';
import { useStudio } from '../studio';
import { sectionNamed, type Group as GroupInfo, type SectionKey } from './sections';

type ProjectInfo = components['schemas']['ProjectInfo'];

// The widths of a pinned list (px): its least, its most, and as it starts
export const LIST_WIDTHS = { least: 180, most: 480, start: 248 };

// The blocks a group shows before 'show all'
const SHOWN = 12;

// Whether a section has a list (home has none)
export function hasList(key: SectionKey): boolean {
  return sectionNamed(key).groups.length > 0;
}

export function SectionList({
  section: key,
  pinned,
  onPin,
  onOpen,
  width,
  onWidth,
}: {
  section: SectionKey;
  pinned: boolean;
  onPin: (pinned: boolean) => void;
  // Called when a block opens (e.g. to close the list over the screen)
  onOpen: () => void;
  // The width of a pinned list, and what sets it as its edge is dragged
  width?: number;
  onWidth?: (width: number) => void;
}) {
  const { project, startAdding } = useStudio();
  const section = sectionNamed(key);
  const [find, setFind] = useState('');
  const query = find.trim().toLowerCase();
  if (project === null) return null;
  const single = section.groups.length === 1 ? section.groups[0] : null;
  return (
    <nav className="list" aria-label={`The ${section.title.toLowerCase()} of the project`}>
      <div className="list-head">
        <b>{section.title}</b>
        {single && (
          <Button
            kind="small"
            title={`A new ${single.kind.toLowerCase()}`}
            onClick={() => {
              startAdding(single.folder);
              onOpen();
            }}
          >
            + New
          </Button>
        )}
        <IconButton label={pinned ? 'Unpin the list: it opens from the rail (Ctrl+B)' : 'Pin the list beside the screen (Ctrl+B)'} pressed={pinned} onClick={() => onPin(!pinned)}>
          <PinIcon />
        </IconButton>
      </div>
      <div className="list-find">
        <TextField search value={find} placeholder="Filter" label={`Filter the ${section.title.toLowerCase()}`} onChange={setFind} />
      </div>
      <div className="list-groups">
        {section.groups.map((group) => (
          <Group key={group.folder} project={project} group={group} find={query} titled={single === null} onOpen={onOpen} />
        ))}
      </div>
      {pinned && width !== undefined && onWidth && <ListEdge width={width} onWidth={onWidth} />}
    </nav>
  );
}

// The edge of a pinned list, dragged (or moved by the arrow keys) to make
// it wider or narrower; a double click sets it back
function ListEdge({ width, onWidth }: { width: number; onWidth: (width: number) => void }) {
  const set = (next: number) => onWidth(Math.round(Math.min(LIST_WIDTHS.most, Math.max(LIST_WIDTHS.least, next))));
  const drag = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const [x0, w0] = [event.clientX, width];
    const move = (moved: PointerEvent) => set(w0 + moved.clientX - x0);
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      document.body.removeAttribute('data-resizing');
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    document.body.setAttribute('data-resizing', '');
  };
  const keys = (event: KeyEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowLeft') set(width - 16);
    if (event.key === 'ArrowRight') set(width + 16);
  };
  return (
    <div
      className="list-edge"
      role="separator"
      aria-orientation="vertical"
      aria-label="The width of the list (drag, or the arrow keys)"
      aria-valuenow={width}
      aria-valuemin={LIST_WIDTHS.least}
      aria-valuemax={LIST_WIDTHS.most}
      tabIndex={0}
      onPointerDown={drag}
      onKeyDown={keys}
      onDoubleClick={() => set(LIST_WIDTHS.start)}
    />
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
  const defs = (useSchema().data as { $defs?: Defs } | undefined)?.$defs;
  const [all, setAll] = useState(false);
  const names = (info.data?.pieces[group.folder] ?? []).filter((name) => name.toLowerCase().includes(find));
  const shown = find || all ? names : names.slice(0, SHOWN);
  return (
    <div className="list-group">
      {titled && (
        <div className="list-group-head">
          <span>{group.title}</span>
          <Button
            kind="small"
            title={`A new ${group.kind.toLowerCase()}`}
            onClick={() => {
              startAdding(group.folder);
              onOpen();
            }}
          >
            + New
          </Button>
        </div>
      )}
      {shown.map((name) => (
        <div key={name} className="list-item">
          <NavItem
            label={name}
            mono
            right={info.data && detailOf(group.folder, name, info.data, defs)}
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

// The families of the schema of the blocks of a folder that are told by
// their type
const FAMILIES: Record<string, string> = { psfs: 'psf', lsfs: 'lsf', noise: 'noise', fitters: 'fitter' };

// What a block of the list is, in a word or two (none where its name says
// enough): a dataset its kind, an observation the dataset it sees, a PSF,
// LSF, noise or fitter its type, a fit how its latest run went
function detailOf(folder: string, name: string, info: ProjectInfo, defs: Defs | undefined): ReactNode {
  const type = info.types?.[folder]?.[name];
  if (folder === 'datasets') {
    const kind = Object.values(KINDS).find((each) => each.type === type);
    return detail(kind?.title ?? (type ? titleOf(defs, 'dataset', type) : null));
  }
  if (folder === 'observations') {
    const data = info.links.find((link) => link.source.folder === folder && link.source.name === name && link.target.folder === 'datasets');
    return detail(data ? data.target.name : 'no data', data ? undefined : 'none');
  }
  if (folder in FAMILIES) return detail(type ? titleOf(defs, FAMILIES[folder], type) : null);
  if (folder === 'setups') {
    const status = info.latest?.[name];
    if (status === undefined) return null;
    const said = { complete: '✓ done', running: '● running', failed: '✕ failed' }[status] ?? status;
    return detail(said, status);
  }
  return null;
}

function detail(text: string | null, state?: string): ReactNode {
  return text ? (
    <span className="list-detail" data-state={state}>
      {text}
    </span>
  ) : null;
}
