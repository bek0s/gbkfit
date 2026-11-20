// A block of the project being edited: its options as saved, as edited
// (with a history, see useEdits), and how they are saved (see the
// server's PUT /pieces) or reverted, as the head of its screen does (see
// SaveControls).
import { useEffect } from 'react';
import { usePiece, useProject, useSchema, useUpdateBlock } from '../api/queries';
import { joinPath } from '../components/FilePicker';
import { useEdits } from '../components/history';
import type { Defs, Options } from '../components/SchemaForm';
import type { Saving } from '../shell/ScreenHead';
import { useStudio } from '../studio';

export interface EditedBlock extends Saving {
  path: string;
  // The options as edited (as saved until edited; undefined: not read)
  options: Options | undefined;
  error: Error | null;
  set: (options: Options) => void;
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
  const saved = piece.data as Options | undefined;
  const edits = useEdits(saved);
  const draft = edits.value;
  const options = draft ?? saved;
  const dirty = draft !== null && JSON.stringify(draft) !== JSON.stringify(saved);
  useUnsaved(dirty);
  return {
    path,
    options,
    error: piece.error ?? schema.error,
    dirty,
    set: edits.set,
    save: () => draft && update.mutate({ path, content: draft }, { onSuccess: edits.settle }),
    revert: () => edits.set(null),
    undo: edits.undo,
    redo: edits.redo,
    canUndo: edits.canUndo,
    canRedo: edits.canRedo,
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
