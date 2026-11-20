// Making things and finding them, from anywhere: the New menu of the bar
// (grouped like the rail), the dialog that names a new block (or the flow
// of its own of a dataset; see SectionList's adders), and Go to (Ctrl+K),
// which finds a section, a block of the project or an action by name.
import { Fragment, useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
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

// The new blocks the menu makes, by the section they are in (as the rail
// shows them)
const MADE: { section: string; folders: string[] }[] = SECTIONS.filter((section) => section.groups.length > 0).map((section) => ({
  section: section.title,
  folders: section.groups.map((group) => group.folder),
}));

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

// What a new block of a folder starts with, where it is not the template
// of its folder (see the library's project templates, made to be edited
// by hand): a galaxy starts empty, its components and its mass added in
// its sheet; a comparison (of the Studio: it has no template) compares
// nothing yet; a fit, nothing yet, its pieces added in its sheet
const STARTS: Record<string, Record<string, unknown>> = { galaxies: {}, comparisons: {}, setups: { galaxies: [], observations: [] } };

export function NamingDialog({ project, folder, onDone }: AdderProps) {
  const { go } = useStudio();
  const create = useCreateBlock(project);
  const [name, setName] = useState('');
  const kind = inSentence(kindOf(folder));
  const submit = () =>
    name.trim() &&
    create.mutate(
      { folder, name: name.trim(), content: STARTS[folder] },
      {
        onSuccess: () => {
          onDone();
          go({ kind: 'block', folder, name: name.trim() });
        },
      },
    );
  return (
    <Dialog title={`New ${kind}`} onClose={onDone}>
      <div className="naming">
        <TextField
          autoFocus
          mono
          value={name}
          placeholder="name, e.g. ngc1313"
          label={`The name of the new ${kind}`}
          onChange={setName}
          onKeyDown={(event) => {
            if (event.key === 'Enter') submit();
          }}
        />
        <p className="naming-note">
          Made in {folder}/{folder in STARTS ? ', empty' : ' from the template of its kind'}; a name can have folders, e.g. ngc1313/hi.
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

// A kind of block within a sentence: lower case, but an acronym (e.g.
// 'PSF' stays, 'Galaxy' is 'galaxy')
function inSentence(kind: string): string {
  return /^[A-Z]{2}/.test(kind) ? kind : kind.charAt(0).toLowerCase() + kind.slice(1);
}

interface Found {
  key: string;
  // The group it is shown in (e.g. 'Galaxies'), and what it is
  group: string;
  kind: string;
  title: string;
  note?: string;
  run: () => void;
}

// The things Go to shows of a group (all, of what is typed), and of all
const PER_GROUP = 6;
const SHOWN = 40;

// Go to (Ctrl+K): the blocks of the project, the sections and the actions
// whose names hold what is typed, under the title of their group; with
// nothing typed, the blocks opened last first. Arrows choose, Enter goes.
export function GoTo() {
  const { project, go, startAdding, visited } = useStudio();
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
      group: 'Sections',
      kind: 'Section',
      title: section.title,
      run: () => openSection(section.key as SectionKey),
    }));
    const blocks = SECTIONS.flatMap((section) =>
      section.groups.flatMap((group) =>
        (info.data?.pieces[group.folder] ?? []).map((name) => ({
          key: `${group.folder}/${name}`,
          group: group.title,
          kind: group.kind,
          title: name,
          run: () => go({ kind: 'block', folder: group.folder, name }),
        })),
      ),
    );
    const actions = MADE.flatMap(({ folders }) =>
      folders.map((folder) => ({
        key: `new/${folder}`,
        group: 'New',
        kind: 'New',
        title: `${kindOf(folder)}…`,
        note: `a new ${inSentence(kindOf(folder))}`,
        run: () => startAdding(folder),
      })),
    );
    return [...blocks, ...sections, ...actions];
  }, [info.data, go, openSection, startAdding]);
  const query = text.trim().toLowerCase();
  const found = useMemo(() => {
    if (query) return all.filter((item) => `${item.kind} ${item.title}`.toLowerCase().includes(query)).slice(0, SHOWN);
    // (nothing typed: the blocks opened last, then the first of each group)
    const recent = visited
      .flatMap((seen) => (seen.kind === 'block' ? all.filter((item) => item.key === `${seen.folder}/${seen.name}`) : []))
      .slice(0, PER_GROUP)
      .map((item) => ({ ...item, key: `recent/${item.key}`, group: 'Opened last', note: item.kind }));
    const counts: Record<string, number> = {};
    const firsts = all.filter((item) => (counts[item.group] = (counts[item.group] ?? 0) + 1) <= PER_GROUP);
    return [...recent, ...firsts].slice(0, SHOWN);
  }, [query, all, visited]);
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
                <Fragment key={item.key}>
                  {item.group !== found[i - 1]?.group && (
                    <div className="goto-group" role="presentation">
                      {item.group}
                    </div>
                  )}
                  <button type="button" role="option" aria-selected={i === chosen} className="goto-item" onMouseEnter={() => setChosen(i)} onClick={() => run(item)}>
                    <b>{item.title}</b>
                    {item.note && <span>{item.note}</span>}
                  </button>
                </Fragment>
              ))}
              {found.length === 0 && <p className="goto-none">Nothing by that name.</p>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
