// What can be done to a block of the project, wherever it is shown (the
// head of its screen, its row in a table): rename it in place (Enter
// keeps the name typed, as does a click away; Escape lets it go; the
// blocks that name it follow), or delete it (those that name it are shown
// first, as they will name nothing). The head of the screen of a block
// has its name, which a click renames, and its menu, of an icon at the
// end of the actions of the screen.
import { Tooltip } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { useDeleteBlock, useReferences, useRenameBlock } from '../api/queries';
import { joinPath } from '../components/FilePicker';
import { MoreIcon } from '../components/icons';
import { Button } from '../components/ui/basics';
import { Dialog, DropMenu, type MenuEntry } from '../components/ui/overlays';
import { useStudio } from '../studio';
import { kindOf } from './sections';

// The actions on a block: the entries of its menu, whether it is being
// renamed (its place then shows a NameInput, or the title of BlockTitle),
// and the dialog that deletes it, while it is open
export function useBlockActions(project: string, folder: string, name: string) {
  const [renaming, setRenaming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const path = joinPath(project, `${folder}/${name}.yaml`);
  const entries: MenuEntry[] = [
    { label: 'Rename', onClick: () => setRenaming(true) },
    { label: 'Delete…', onClick: () => setDeleting(true) },
  ];
  const dialog = deleting ? <DeleteDialog project={project} folder={folder} name={name} path={path} onClose={() => setDeleting(false)} /> : null;
  return { path, entries, renaming, setRenaming, dialog };
}

export function useBlockHead(project: string, folder: string, name: string) {
  const { path, entries, renaming, setRenaming, dialog } = useBlockActions(project, folder, name);
  const what = kindOf(folder).toLowerCase();
  const title = <BlockTitle key={name} project={project} folder={folder} name={name} path={path} renaming={renaming} onRenaming={setRenaming} />;
  const menu = (
    <>
      <DropMenu
        label={`The ${what}`}
        at="end"
        returnFocus={false}
        entries={entries}
        target={
          <Button title={`Rename or delete the ${what}`} aria-label={`The ${what}`} data-more>
            <MoreIcon />
          </Button>
        }
      />
      {dialog}
    </>
  );
  return { title, menu };
}

function BlockTitle({
  project,
  folder,
  name,
  path,
  renaming,
  onRenaming,
}: {
  project: string;
  folder: string;
  name: string;
  path: string;
  renaming: boolean;
  onRenaming: (renaming: boolean) => void;
}) {
  const what = kindOf(folder).toLowerCase();
  if (renaming) return <TitleInput project={project} folder={folder} name={name} path={path} onDone={() => onRenaming(false)} />;
  return (
    <h1 className="block-title">
      <span className="block-title-box" title={`Rename the ${what}`} onClick={() => onRenaming(true)}>
        <span className="block-title-text">{name}</span>
      </span>
    </h1>
  );
}

interface RenameProps {
  project: string;
  folder: string;
  name: string;
  // The path of its file in the workspace
  path: string;
  onDone: () => void;
}

// (the name as it is renamed: an input over the name typed, which sizes
// its box as the name does, so that the box keeps its place and size)
function TitleInput(props: RenameProps) {
  const { typed, error, inputProps } = useRenaming(props);
  return (
    <h1 className="block-title" data-renaming>
      <span className="block-title-box">
        <span className="block-title-text">{typed || ' '}</span>
        <input className="block-title-input" size={1} {...inputProps} />
      </span>
      {error && <span className="block-title-error error">{error}</span>}
    </h1>
  );
}

// The name of a block as it is renamed in a row of a table: over the
// name, in its place (so that nothing moves), the error of the rename in a
// tooltip
export function NameInput(props: RenameProps) {
  const { error, inputProps } = useRenaming(props);
  return (
    <Tooltip label={error} opened={error !== null} position="top-start" multiline maw={360} classNames={{ tooltip: 'problem-tip' }}>
      <input className="name-input-row" data-error={error !== null || undefined} {...inputProps} />
    </Tooltip>
  );
}

// The state of a name being renamed: as typed, the error of the rename
// (if any), and the props of its input
function useRenaming({ project, folder, name, path, onDone }: RenameProps) {
  const { movedBlock } = useStudio();
  const rename = useRenameBlock(project);
  const [typed, setTyped] = useState(name);
  // (let go by Escape, or kept: not kept again by the blur that follows)
  const dropped = useRef(false);
  const input = useRef<HTMLInputElement>(null);
  const what = kindOf(folder).toLowerCase();
  // (a press anywhere else keeps the name, as a blur does: also where the
  // press does not take the focus, e.g. on a map)
  useEffect(() => {
    const onDown = (event: PointerEvent) => event.target !== input.current && input.current?.blur();
    document.addEventListener('pointerdown', onDown, true);
    return () => document.removeEventListener('pointerdown', onDown, true);
  }, []);
  const keep = () => {
    const next = typed.trim();
    if (dropped.current || rename.isPending) return;
    if (!next || next === name) {
      onDone();
      return;
    }
    rename.mutate(
      { path, folder, from: name, to: next },
      {
        onSuccess: () => {
          dropped.current = true;
          onDone();
          movedBlock(folder, name, next);
        },
      },
    );
  };
  const inputProps = {
    ref: input,
    autoFocus: true,
    spellCheck: false,
    'aria-label': `Name of the ${what}`,
    value: typed,
    onFocus: (event: React.FocusEvent<HTMLInputElement>) => event.target.select(),
    onChange: (event: React.ChangeEvent<HTMLInputElement>) => setTyped(event.target.value),
    onBlur: keep,
    onKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') keep();
      if (event.key === 'Escape') {
        dropped.current = true;
        onDone();
      }
    },
  };
  return { typed, error: rename.error?.message ?? null, inputProps };
}

interface DialogProps {
  project: string;
  folder: string;
  name: string;
  // The path of its file in the workspace
  path: string;
  onClose: () => void;
}

function DeleteDialog({ project, folder, name, path, onClose }: DialogProps) {
  const { movedBlock } = useStudio();
  const remove = useDeleteBlock(project);
  const references = useReferences(path);
  const named = references.data ?? [];
  const what = kindOf(folder).toLowerCase();
  return (
    <Dialog title={`Delete the ${what}`} onClose={onClose}>
      <div className="dialog-form">
        <p>
          This deletes the {what} <b className="mono">{name}</b>.
          {folder === 'datasets' && ' Its files of data stay in the project.'}
        </p>
        {named.length > 0 && (
          <div className="callout-list">
            <p>
              {named.length === 1 ? 'This block names it' : 'These blocks name it'}, and will name nothing once it is deleted, until
              changed:
            </p>
            <ul>
              {named.map((r) => (
                <li key={`${r.folder}/${r.name}`}>
                  <span className="mono">{r.name}</span> ({kindOf(r.folder).toLowerCase()})
                </li>
              ))}
            </ul>
          </div>
        )}
        {remove.error && <p className="error">{remove.error.message}</p>}
        <div className="dialog-actions">
          <Button onClick={onClose}>Cancel</Button>
          <Button
            kind="danger"
            disabled={references.isPending || remove.isPending}
            onClick={() =>
              remove.mutate(path, {
                onSuccess: () => {
                  onClose();
                  movedBlock(folder, name, null);
                },
              })
            }
          >
            {named.length > 0 ? 'Delete anyway' : 'Delete'}
          </Button>
        </div>
      </div>
    </Dialog>
  );
}
