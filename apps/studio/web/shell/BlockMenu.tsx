// The menu of a block of the project, at the head of its screen: rename it
// (the blocks that name it follow) or delete it (those that name it are
// shown first, as they will name nothing).
import { useState } from 'react';
import { useDeleteBlock, useReferences, useRenameBlock } from '../api/queries';
import { joinPath } from '../components/FilePicker';
import { Button, TextField } from '../components/ui/basics';
import { Dialog, DropMenu } from '../components/ui/overlays';
import { useStudio } from '../studio';
import { kindOf } from './blocks';

export function BlockMenu({ project, folder, name }: { project: string; folder: string; name: string }) {
  const [dialog, setDialog] = useState<'rename' | 'delete' | null>(null);
  const path = joinPath(project, `${folder}/${name}.yaml`);
  return (
    <>
      <DropMenu
        label={`The ${kindOf(folder).toLowerCase()}`}
        entries={[
          { label: 'Rename…', onClick: () => setDialog('rename') },
          { label: 'Delete…', onClick: () => setDialog('delete') },
        ]}
        target={
          <Button title={`Rename or delete the ${kindOf(folder).toLowerCase()}`} right="▾">
            {kindOf(folder)}
          </Button>
        }
      />
      {dialog === 'rename' && <RenameDialog project={project} folder={folder} name={name} path={path} onClose={() => setDialog(null)} />}
      {dialog === 'delete' && <DeleteDialog project={project} folder={folder} name={name} path={path} onClose={() => setDialog(null)} />}
    </>
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

function RenameDialog({ project, folder, name, path, onClose }: DialogProps) {
  const { movedBlock } = useStudio();
  const rename = useRenameBlock(project);
  const references = useReferences(path).data ?? [];
  const [typed, setTyped] = useState(name);
  const next = typed.trim();
  const what = kindOf(folder).toLowerCase();
  return (
    <Dialog title={`Rename the ${what}`} onClose={onClose}>
      <form
        className="dialog-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!next || next === name) return;
          rename.mutate(
            { path, name: next },
            {
              onSuccess: () => {
                onClose();
                movedBlock(folder, name, next);
              },
            },
          );
        }}
      >
        <TextField autoFocus mono showLabel value={typed} label={`Name of the ${what}`} onChange={setTyped} />
        {references.length > 0 && (
          <p className="note">
            {references.length === 1 ? 'One block names' : `${references.length} blocks name`} it; they follow:{' '}
            {references.map((r) => `${r.name} (${kindOf(r.folder).toLowerCase()})`).join(', ')}.
          </p>
        )}
        {rename.error && <p className="error">{rename.error.message}</p>}
        <div className="dialog-actions">
          <Button onClick={onClose}>Cancel</Button>
          <Button kind="primary" submit disabled={!next || next === name || rename.isPending}>
            Rename
          </Button>
        </div>
      </form>
    </Dialog>
  );
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
