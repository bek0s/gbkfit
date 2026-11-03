// A block of the project being edited: its options as saved, as edited,
// and how they are saved (see the server's PUT /pieces) or reverted; and
// the actions of the head of its screen.
import { useEffect, useState } from 'react';
import { usePiece, useProject, useSchema, useUpdateBlock } from '../api/queries';
import { joinPath } from '../components/FilePicker';
import type { Defs, Options } from '../components/SchemaForm';
import { Button } from '../components/ui/basics';
import { useStudio } from '../studio';

export interface EditedBlock {
  path: string;
  // The options as edited (as saved until edited; undefined: not read)
  options: Options | undefined;
  error: Error | null;
  dirty: boolean;
  set: (options: Options) => void;
  save: () => void;
  revert: () => void;
  saving: boolean;
  saveError: Error | null;
  // The schema of the library, and the blocks of the project by folder
  defs: Defs | undefined;
  blocks: Record<string, string[]>;
}

export function useEditedBlock(project: string, folder: string, name: string): EditedBlock {
  const path = joinPath(project, `${folder}/${name}.yaml`);
  const piece = usePiece(path);
  const schema = useSchema();
  const info = useProject(project);
  const update = useUpdateBlock();
  const [draft, setDraft] = useState<Options | null>(null);
  const saved = piece.data as Options | undefined;
  const options = draft ?? saved;
  const dirty = draft !== null && JSON.stringify(draft) !== JSON.stringify(saved);
  useUnsaved(dirty);
  return {
    path,
    options,
    error: piece.error ?? schema.error,
    dirty,
    set: setDraft,
    save: () => draft && update.mutate({ path, content: draft }, { onSuccess: () => setDraft(null) }),
    revert: () => setDraft(null),
    saving: update.isPending,
    saveError: update.error,
    defs: schema.data?.$defs as Defs | undefined,
    blocks: info.data?.pieces ?? {},
  };
}

// Whether a screen has edits not saved: leaving it (another screen, or
// the page) asks first
export function useUnsaved(unsaved: boolean) {
  const { setUnsaved } = useStudio();
  useEffect(() => {
    setUnsaved(unsaved);
    if (!unsaved) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener('beforeunload', warn);
    return () => {
      window.removeEventListener('beforeunload', warn);
      setUnsaved(false);
    };
  }, [unsaved, setUnsaved]);
}

// Revert and save, with whether the edits are saved
export function SaveActions({ edited }: { edited: EditedBlock }) {
  return (
    <>
      {edited.dirty && <span className="why">Not saved</span>}
      {edited.saveError && <span className="error">{edited.saveError.message}</span>}
      <Button disabled={!edited.dirty} onClick={edited.revert}>
        Revert
      </Button>
      <Button kind="primary" disabled={!edited.dirty || edited.saving} onClick={edited.save}>
        Save
      </Button>
    </>
  );
}
