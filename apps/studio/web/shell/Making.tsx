// Making things and finding them, from anywhere: the New menu of the bar
// (grouped like the rail), the dialog that names a new block (or the flow
// of its own of a dataset; see SectionList's adders), and Go to (Ctrl+K),
// which finds a section, a block of the project or an action by name.
import { useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import { useCreateBlock, useProject } from '../api/queries';
import { Button, TextField } from '../components/ui/basics';
import { Dialog, DropMenu, type MenuEntry } from '../components/ui/overlays';
import { useStudio } from '../studio';
import { DataKindDialog } from './DataKindDialog';
import { useOpenSection } from './Rail';
import { kindOf, SECTIONS, type SectionKey } from './sections';

// How a new block of a folder is made: by its name, from the template of
// its folder (see NamingDialog), or by a flow of its own
export interface AdderProps {
  project: string;
  folder: string;
  onDone: () => void;
}

const ADDERS: Record<string, ComponentType<AdderProps>> = { datasets: DataKindDialog };

// The new blocks the menu makes, by the section they are in (the tools
// make none yet: comparisons and fits come with their screens)
const MADE: { section: string; folders: string[] }[] = [
  { section: 'Data', folders: ['datasets'] },
  { section: 'Galaxies', folders: ['galaxies'] },
  { section: 'Observations', folders: ['observations', 'psfs', 'lsfs', 'noise'] },
];

export function NewMenu() {
  const { project, startAdding } = useStudio();
  const entries: MenuEntry[] = MADE.flatMap(({ section, folders }) => [
    { section },
    ...folders.map((folder) => ({ label: `${kindOf(folder)}…`, onClick: () => startAdding(folder) })),
  ]);
  if (project === null) return null;
  return (
    <DropMenu
      label="Make a new block"
      entries={entries}
      target={
        <Button kind="primary" right="▾" title="Make a new block">
          + New
        </Button>
      }
    />
  );
}

// The dialog of the block being made (see Studio.adding): its own flow for
// a dataset, its name for the others
export function Adding() {
  const { project, adding, startAdding } = useStudio();
  if (project === null || adding === null) return null;
  const Adder = ADDERS[adding] ?? NamingDialog;
  return <Adder project={project} folder={adding} onDone={() => startAdding(null)} />;
}

export function NamingDialog({ project, folder, onDone }: AdderProps) {
  const { go } = useStudio();
  const create = useCreateBlock(project);
  const [name, setName] = useState('');
  const kind = kindOf(folder);
  const submit = () =>
    name.trim() &&
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
    <Dialog title={`New ${kind.toLowerCase()}`} onClose={onDone}>
      <div className="naming">
        <TextField
          autoFocus
          mono
          value={name}
          placeholder="name, e.g. ngc1313"
          label={`The name of the new ${kind.toLowerCase()}`}
          onChange={setName}
          onKeyDown={(event) => {
            if (event.key === 'Enter') submit();
          }}
        />
        <p className="naming-note">
          Made in {folder}/ from the template of its kind; a name can have folders, e.g. ngc1313/hi.
        </p>
        {create.error && <p className="error">{create.error.message}</p>}
        <div className="naming-actions">
          <Button onClick={onDone}>Cancel</Button>
          <Button kind="primary" disabled={!name.trim() || create.isPending} onClick={submit}>
            Make it
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

interface Found {
  key: string;
  kind: string;
  title: string;
  note?: string;
  run: () => void;
}

// Go to (Ctrl+K): the sections, the blocks of the project and the actions
// whose names hold what is typed; arrows choose, Enter goes
export function GoTo() {
  const { project, go, startAdding } = useStudio();
  const openSection = useOpenSection();
  const info = useProject(project);
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  const [chosen, setChosen] = useState(0);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setOpen((was) => !was);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const all = useMemo<Found[]>(() => {
    const sections = SECTIONS.map((section) => ({
      key: `section/${section.key}`,
      kind: 'Section',
      title: section.title,
      run: () => openSection(section.key as SectionKey),
    }));
    const blocks = SECTIONS.flatMap((section) =>
      section.groups.flatMap((group) =>
        (info.data?.pieces[group.folder] ?? []).map((name) => ({
          key: `${group.folder}/${name}`,
          kind: group.kind,
          title: name,
          run: () => go({ kind: 'block', folder: group.folder, name }),
        })),
      ),
    );
    const actions = MADE.flatMap(({ folders }) =>
      folders.map((folder) => ({
        key: `new/${folder}`,
        kind: 'New',
        title: `${kindOf(folder)}…`,
        note: `a new ${kindOf(folder).toLowerCase()}`,
        run: () => startAdding(folder),
      })),
    );
    return [...blocks, ...sections, ...actions];
  }, [info.data, go, openSection, startAdding]);
  const query = text.trim().toLowerCase();
  const found = (query ? all.filter((item) => `${item.kind} ${item.title}`.toLowerCase().includes(query)) : all).slice(0, 12);
  const input = useRef<HTMLInputElement>(null);
  if (project === null) return null;
  const run = (item: Found | undefined) => {
    if (!item) return;
    setOpen(false);
    setText('');
    item.run();
  };
  return (
    <>
      <button type="button" className="goto" title="Go to (Ctrl+K)" onClick={() => setOpen(true)}>
        Go to anything
        <kbd>Ctrl K</kbd>
      </button>
      {open && (
        <div className="goto-scrim" onMouseDown={() => setOpen(false)}>
          <div className="goto-box" role="dialog" aria-label="Go to" onMouseDown={(event) => event.stopPropagation()}>
            <input
              ref={input}
              autoFocus
              className="goto-input"
              value={text}
              placeholder="A galaxy, a dataset, a section, or New…"
              aria-label="Go to"
              onChange={(event) => {
                setText(event.target.value);
                setChosen(0);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Escape') setOpen(false);
                if (event.key === 'ArrowDown') setChosen(Math.min(chosen + 1, found.length - 1));
                if (event.key === 'ArrowUp') setChosen(Math.max(chosen - 1, 0));
                if (event.key === 'Enter') run(found[chosen]);
              }}
            />
            <div className="goto-list" role="listbox">
              {found.map((item, i) => (
                <button
                  key={item.key}
                  type="button"
                  role="option"
                  aria-selected={i === chosen}
                  className="goto-item"
                  onMouseEnter={() => setChosen(i)}
                  onClick={() => run(item)}
                >
                  <small>{item.kind}</small>
                  <b>{item.title}</b>
                  {item.note && <span>{item.note}</span>}
                </button>
              ))}
              {found.length === 0 && <p className="goto-none">Nothing by that name.</p>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
