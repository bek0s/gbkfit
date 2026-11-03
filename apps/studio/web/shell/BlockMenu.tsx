// The head of the screen of a block of the project: its name, which a
// click renames in place (Enter keeps the name typed, as does a click
// away; Escape lets it go; the blocks that name it follow), and its menu,
// of an icon at the end of the actions of the screen: rename it (as a
// click on its name does) or delete it (those that name it are shown
// first, as they will name nothing).
import { useEffect, useRef, useState } from 'react';
import { useDeleteBlock, useReferences, useRenameBlock } from '../api/queries';
import { joinPath } from '../components/FilePicker';
import { MoreIcon } from '../components/icons';
import { Button } from '../components/ui/basics';
import { Dialog, DropMenu } from '../components/ui/overlays';
import { useStudio } from '../studio';
import { kindOf } from './blocks';

export function useBlockHead(project: string, folder: string, name: string) {
  const [renaming, setRenaming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const path = joinPath(project, `${folder}/${name}.yaml`);
  const what = kindOf(folder).toLowerCase();
  const title = <BlockTitle key={name} project={project} folder={folder} name={name} path={path} renaming={renaming} onRenaming={setRenaming} />;
  const menu = (
    <>
      <DropMenu
        label={`The ${what}`}
        at="end"
        returnFocus={false}
        entries={[
          { label: 'Rename', onClick: () => setRenaming(true) },
          { label: 'Delete…', onClick: () => setDeleting(true) },
        ]}
        target={
          <Button title={`Rename or delete the ${what}`} aria-label={`The ${what}`} data-more>
            <MoreIcon />
          </Button>
        }
      />
      {deleting && <DeleteDialog project={project} folder={folder} name={name} path={path} onClose={() => setDeleting(false)} />}
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

// (the name as it is renamed: an input over the name typed, which sizes
// its box as the name does, so that the box keeps its place and size)
function TitleInput({ project, folder, name, path, onDone }: { project: string; folder: string; name: string; path: string; onDone: () => void }) {
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
      { path, name: next },
      {
        onSuccess: () => {
          dropped.current = true;
          onDone();
          movedBlock(folder, name, next);
        },
      },
    );
  };
  return (
    <h1 className="block-title" data-renaming>
      <span className="block-title-box">
        <span className="block-title-text">{typed || ' '}</span>
        <input
          ref={input}
          className="block-title-input"
          autoFocus
          spellCheck={false}
          aria-label={`Name of the ${what}`}
          value={typed}
          size={1}
          onFocus={(event) => event.target.select()}
          onChange={(event) => setTyped(event.target.value)}
          onBlur={keep}
          onKeyDown={(event) => {
            if (event.key === 'Enter') keep();
            if (event.key === 'Escape') {
              dropped.current = true;
              onDone();
            }
          }}
        />
      </span>
      {rename.error && <span className="block-title-error error">{rename.error.message}</span>}
    </h1>
  );
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
