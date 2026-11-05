// A galaxy being built: its options (its block, galaxies/<name>) and the
// properties of its parameters (the params block of the same name,
// params/<name>), edited together as one draft with a history (undo and
// redo; the moves of a slider are one step), described by the library as
// they change (its parameters and problems), and saved together.
//
// The names and sizes of the parameters follow the options (see
// params.transition): as each description of a new state of the options
// comes, the properties are carried over to it from the last one (renamed
// where an edit moved what owns them, e.g. a trait removed before
// others; resampled to a new number of rings; those of parameters gone,
// removed). The step of history of the edit is updated in place.
import { useDebouncedValue } from '@mantine/hooks';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, data } from '../../api/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { keys, useCreateBlock, useGalaxyParameters, usePiece, useSchema, useUpdateBlock } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import type { Defs } from '../../components/SchemaForm';
import { getAt, isObject, ringRadii, withOpacitiesInStep, type Options, type Path } from './options';
import { transition, type OwnerMap, type Param, type Properties } from './params';

export interface State {
  options: Options;
  properties: Properties;
}

// A problem of the galaxy, at the path of its option (relative to it)
export interface Problem {
  severity: string;
  path: Path;
  message: string;
}

// How an edit is made: the map of what it moved (for the properties of
// the parameters; see params.transition), and a key that makes the edits
// of one gesture one step of history (e.g. the moves of a slider)
export interface EditHow {
  moved?: OwnerMap;
  gesture?: string;
}

const HISTORY = 200;

// (the same gesture within this long is one step)
const GESTURE_MS = 1200;

export function useGalaxyDraft(project: string, name: string) {
  const galaxyPath = joinPath(project, `galaxies/${name}.yaml`);
  const paramsPath = joinPath(project, `params/${name}.yaml`);
  const galaxyPiece = usePiece(galaxyPath);
  // (a params block that is not there yet is not asked for again)
  const paramsPiece = useQuery({
    queryKey: keys.piece(paramsPath),
    retry: false,
    queryFn: () => data(api.GET('/api/v1/pieces/{path}', { params: { path: { path: paramsPath } } })),
  });
  const schema = useSchema();
  const client = useQueryClient();
  const updateBlock = useUpdateBlock();
  const createBlock = useCreateBlock(project);

  const savedOptions = isObject(galaxyPiece.data) ? galaxyPiece.data : undefined;
  // (a params block that is not there yet: none, made as it is saved)
  const paramsBlock = isObject(paramsPiece.data) ? paramsPiece.data : undefined;
  const savedProperties = useMemo(() => (isObject(paramsBlock?.properties) ? (paramsBlock.properties as Properties) : {}), [paramsBlock]);
  const loaded = savedOptions !== undefined && !paramsPiece.isPending;

  // The history: the states before, the one shown (null: as saved), and
  // those undone
  const [history, setHistory] = useState<{ past: State[]; present: State | null; future: State[] }>({ past: [], present: null, future: [] });
  const saved = useMemo<State | undefined>(() => (savedOptions ? { options: savedOptions, properties: savedProperties } : undefined), [savedOptions, savedProperties]);
  const state = history.present ?? saved;

  // (the last gesture: its key and when)
  const gesture = useRef<{ key: string; at: number } | null>(null);
  // (what the edits since the last description moved, in turn; and the
  // options of a state come back from history (undone, redone, reverted),
  // whose properties are in step with them already: their description is
  // taken in as it is)
  const pending = useRef<OwnerMap[]>([]);
  const restored = useRef<string | null>(null);
  const restore = (next: State) => {
    pending.current = [];
    restored.current = JSON.stringify(next.options);
    return next;
  };

  const edit = useCallback(
    (change: (state: State) => State, how: EditHow = {}) => {
      if (!saved) return;
      const now = Date.now();
      const same = how.gesture !== undefined && gesture.current?.key === how.gesture && now - gesture.current.at < GESTURE_MS;
      gesture.current = how.gesture === undefined ? null : { key: how.gesture, at: now };
      if (how.moved) pending.current = [...pending.current, how.moved];
      setHistory(({ past, present }) => {
        const current = present ?? saved;
        const next = change(current);
        const options = withOpacitiesInStep(next.options);
        const stepped = { options, properties: next.properties };
        if (same) return { past, present: stepped, future: [] };
        return { past: [...past, current].slice(-HISTORY), present: stepped, future: [] };
      });
    },
    [saved],
  );

  const undo = useCallback(() => {
    if (!saved) return;
    gesture.current = null;
    setHistory(({ past, present, future }) =>
      past.length === 0 ? { past, present, future } : { past: past.slice(0, -1), present: restore(past[past.length - 1]), future: [present ?? saved, ...future] },
    );
  }, [saved]);

  const redo = useCallback(() => {
    if (!saved) return;
    gesture.current = null;
    setHistory(({ past, present, future }) =>
      future.length === 0 ? { past, present, future } : { past: [...past, present ?? saved], present: restore(future[0]), future: future.slice(1) },
    );
  }, [saved]);

  // The description of the options, as they settle (the last stays while
  // the next comes)
  const [settled] = useDebouncedValue(state?.options, 200);
  const description = useGalaxyParameters(project, settled);
  const fresh = description.isSuccess && !description.isPlaceholderData;
  const params = (description.data?.parameters ?? []) as unknown as Param[];
  const problems = (description.data?.problems ?? []) as unknown as Problem[];
  // (valid: the library made the galaxy; it has parameters, or no
  // components to have them)
  const valid = fresh && (params.length > 0 || problems.length === 0);

  // The last valid description that has been taken in: the options it is
  // of (their text, and them), and their parameters. As each new one comes,
  // the properties are carried over to it (see the head).
  const [described, setDescribed] = useState<{ key: string; options: Options; params: Param[] } | null>(null);
  useEffect(() => {
    if (!valid || settled === undefined || state === undefined) return;
    // (the description of the options shown, not of those before)
    const key = JSON.stringify(settled);
    if (key !== JSON.stringify(state.options) || key === described?.key) return;
    const moves = pending.current;
    pending.current = [];
    const back = restored.current === key;
    if (back) restored.current = null;
    setDescribed({ key, options: settled, params });
    if (described === null || back) return;
    const moved: OwnerMap | null = moves.length ? (owner) => moves.reduce<Path | null>((path, move) => (path === null ? null : move(path)), owner) : null;
    const radii = (options: Options) => (component: number) => ringRadii(getAt(options, ['components', component, 'rnodes']));
    const properties = transition(state.properties, described.params, params, moved, { before: radii(described.options), after: radii(settled) });
    if (JSON.stringify(properties) !== JSON.stringify(state.properties)) setHistory((h) => ({ ...h, present: { options: state.options, properties } }));
    // (once per description, or state shown)
  }, [description.data, valid, state]);

  // The state the library has described (its options, and its properties
  // in step with them), to render: the one shown once it is, the last
  // before while it is not (e.g. just after an edit of the options)
  const consistentRef = useRef<State | undefined>(undefined);
  if (state && described && described.key === JSON.stringify(state.options)) consistentRef.current = state;
  const consistent = consistentRef.current;

  const dirty = state !== undefined && saved !== undefined && JSON.stringify(state) !== JSON.stringify(saved);
  const galaxyDirty = state !== undefined && saved !== undefined && JSON.stringify(state.options) !== JSON.stringify(saved.options);
  const paramsDirty = state !== undefined && saved !== undefined && JSON.stringify(state.properties) !== JSON.stringify(saved.properties);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<Error | null>(null);
  const save = useCallback(async () => {
    if (!state || saving) return;
    setSaving(true);
    setSaveError(null);
    try {
      if (galaxyDirty) {
        await updateBlock.mutateAsync({ path: galaxyPath, content: state.options });
        client.setQueryData(keys.piece(galaxyPath), state.options);
      }
      if (paramsDirty) {
        const content = { ...(paramsBlock ?? {}), properties: state.properties };
        if (paramsBlock) await updateBlock.mutateAsync({ path: paramsPath, content });
        else await createBlock.mutateAsync({ folder: 'params', name, content });
        client.setQueryData(keys.piece(paramsPath), content);
      }
      // (as saved now, unless edited since; the history stays, to undo
      // past the save)
      setHistory((h) => (h.present === state ? { ...h, present: null } : h));
    } catch (e) {
      setSaveError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setSaving(false);
    }
  }, [state, saving, galaxyDirty, paramsDirty, updateBlock, galaxyPath, paramsPath, paramsBlock, createBlock, name, client]);

  const revert = useCallback(() => {
    gesture.current = null;
    if (saved) restore(saved);
    setHistory({ past: [], present: null, future: [] });
  }, [saved]);

  return {
    loaded,
    error: (galaxyPiece.error ?? schema.error) as Error | null,
    state,
    edit,
    undo,
    redo,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
    dirty,
    save,
    revert,
    saving,
    saveError,
    defs: schema.data?.$defs as Defs | undefined,
    consistent,
    // The description: the parameters (of the last valid one), the problems
    // (of the options shown), and whether it is of them
    params: described?.params ?? [],
    problems: fresh ? problems : [],
    // (a description that failed: e.g. the server could not be reached)
    describeError: description.error as Error | null,
  };
}

export type GalaxyDraft = ReturnType<typeof useGalaxyDraft>;
