// The tree: the blocks of the open project, in sections (data, model,
// fit) of groups, one group per folder. A group folds, shows its first
// blocks and the rest on demand, and makes new blocks (+); the box finds
// blocks by name. Folded whole, the tree is a strip of the icons of its
// sections, each opening its groups over the page.
import { useState, type ComponentType } from 'react';
import { useCreateBlock, useProject } from '../api/queries';
import { DataIcon, FitIcon, ModelIcon } from '../components/icons';
import { Button, IconButton, NavItem, TextField } from '../components/ui/basics';
import { useStudio } from '../studio';
import { SECTIONS, type Group as GroupInfo, type Section } from './blocks';
import { DataKindDialog } from './DataKindDialog';
import { ProjectMenu } from './ProjectMenu';
import { useRemembered } from './remember';

const ICONS = { data: <DataIcon />, model: <ModelIcon />, fit: <FitIcon /> };

// The blocks a group shows before 'show all'
const SHOWN = 5;

// How a new block of a folder is made (+): by its name, from the template
// of its folder (see NewBlock), or by a flow of its own
export interface AdderProps {
  project: string;
  folder: string;
  onDone: () => void;
}

const ADDERS: Record<string, ComponentType<AdderProps>> = { datasets: DataKindDialog };

export function Tree() {
  const { project, folded, setFolded } = useStudio();
  const [flyout, setFlyout] = useState<Section['key'] | null>(null);
  const [closed, setClosed] = useRemembered<string[]>('closed-groups', ['noise', 'params', 'fitters']);
  const folds = {
    closed,
    toggle: (folder: string) =>
      setClosed(closed.includes(folder) ? closed.filter((f) => f !== folder) : [...closed, folder]),
  };
  return (
    <nav className="tree" aria-label="The blocks of the project">
      {!folded && <ProjectMenu />}
      {project === null ? (
        !folded && <p className="tree-note">Open a project to see its blocks here.</p>
      ) : folded ? (
        <div className="strip">
          {SECTIONS.map((section) => (
            <IconButton
              key={section.key}
              label={section.title}
              pressed={flyout === section.key}
              onClick={() => setFlyout(flyout === section.key ? null : section.key)}
            >
              {ICONS[section.key]}
            </IconButton>
          ))}
        </div>
      ) : (
        <Blocks project={project} folds={folds} />
      )}
      <div className="tree-foot">
        <Button kind="quiet" title={folded ? 'Unfold the tree' : 'Fold the tree'} onClick={() => setFolded(!folded)}>
          {folded ? '»' : '« Fold'}
        </Button>
      </div>
      {folded && project !== null && flyout !== null && (
        <div className="flyout">
          <SectionView
            project={project}
            section={SECTIONS.find((section) => section.key === flyout)!}
            find=""
            folds={folds}
            onOpen={() => setFlyout(null)}
          />
        </div>
      )}
    </nav>
  );
}

// The folded groups of the tree, by folder, and how to fold one or not
interface Folds {
  closed: string[];
  toggle: (folder: string) => void;
}

// The sections of the blocks of a project, and the box that finds them
function Blocks({ project, folds }: { project: string; folds: Folds }) {
  const info = useProject(project);
  const [find, setFind] = useState('');
  const query = find.trim().toLowerCase();
  const names = (folder: string) => info.data?.pieces[folder] ?? [];
  const found = SECTIONS.filter((section) =>
    section.groups.some((group) => names(group.folder).some((name) => name.toLowerCase().includes(query))),
  );
  return (
    <>
      <div className="tree-find">
        <TextField search value={find} placeholder="Find in the project" label="Find in the project" onChange={setFind} />
      </div>
      <div className="tree-list">
        {info.error && <p className="error">{info.error.message}</p>}
        {(query ? found : SECTIONS).map((section) => (
          <SectionView key={section.key} project={project} section={section} find={query} folds={folds} />
        ))}
        {query && found.length === 0 && <p className="tree-note">Nothing called “{find.trim()}”.</p>}
      </div>
    </>
  );
}

function SectionView({
  project,
  section,
  find,
  folds,
  onOpen,
}: {
  project: string;
  section: Section;
  // The text to find (lower case; empty: all)
  find: string;
  folds: Folds;
  // Called when a block opens (e.g. to close the flyout)
  onOpen?: () => void;
}) {
  const info = useProject(project);
  return (
    <div>
      <div className="tree-section">{section.title}</div>
      {section.groups.map((group) => {
        const all = info.data?.pieces[group.folder] ?? [];
        const names = all.filter((name) => name.toLowerCase().includes(find));
        if (find && names.length === 0) return null;
        return (
          <Group
            key={group.folder}
            project={project}
            group={group}
            names={names}
            total={all.length}
            open={find !== '' || !folds.closed.includes(group.folder)}
            whole={find !== ''}
            onToggle={() => folds.toggle(group.folder)}
            onOpen={onOpen}
          />
        );
      })}
    </div>
  );
}

function Group({
  project,
  group,
  names,
  total,
  open,
  whole,
  onToggle,
  onOpen,
}: {
  project: string;
  group: GroupInfo;
  // The names it shows (those found, when finding), of its total
  names: string[];
  total: number;
  open: boolean;
  // Whether it shows all its names, or its first ones until asked
  whole: boolean;
  onToggle: () => void;
  onOpen?: () => void;
}) {
  const { screen, go, adding, startAdding } = useStudio();
  const [all, setAll] = useState(false);
  const shown = whole || all ? names : names.slice(0, SHOWN);
  const { folder, title } = group;
  const Adder = ADDERS[folder] ?? NewBlock;
  return (
    <>
      <div className="tree-group">
        <NavItem label={title} opened={open} right={<span className="count">{total}</span>} onClick={onToggle} />
        <IconButton
          label={`New in ${title}`}
          onClick={() => {
            if (!open) onToggle();
            startAdding(folder);
          }}
        >
          +
        </IconButton>
      </div>
      {adding === folder && <Adder project={project} folder={folder} onDone={() => startAdding(null)} />}
      {open &&
        shown.map((name) => (
          <div key={name} className="tree-item">
            <NavItem
              label={name}
              mono
              active={screen.kind === 'block' && screen.folder === folder && screen.name === name}
              onClick={() => {
                go({ kind: 'block', folder, name });
                onOpen?.();
              }}
            />
          </div>
        ))}
      {open && shown.length < names.length && (
        <div className="tree-more">
          <Button kind="small" onClick={() => setAll(true)}>
            show all {names.length}
          </Button>
        </div>
      )}
    </>
  );
}

// The name of a new block of a folder, made from its template: Enter makes
// and opens it, Escape cancels
function NewBlock({ project, folder, onDone }: AdderProps) {
  const { go } = useStudio();
  const create = useCreateBlock(project);
  const [name, setName] = useState('');
  const submit = () =>
    create.mutate(
      { folder, name: name.trim() },
      {
        onSuccess: () => {
          onDone();
          go({ kind: 'block', folder, name: name.trim() });
        },
      },
    );
  return (
    <div className="tree-name">
      <TextField
        autoFocus
        mono
        value={name}
        placeholder="name, e.g. ngc1313"
        label="Name of the new block"
        onChange={setName}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && name.trim()) submit();
          if (event.key === 'Escape') onDone();
        }}
        onBlur={() => !name.trim() && onDone()}
      />
      {create.error && <p className="error">{create.error.message}</p>}
    </div>
  );
}
