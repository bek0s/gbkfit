// The draft of a lens of the project: its options (the piece of the lens)
// and the properties of its parameters (the params block of the same name,
// params/<name>, by the names of the lens: e.g. sie_theta_e), edited as one
// and saved together, as a galaxy and its parameters are (see the galaxy
// screen's draft).
import { useQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';
import { api, data } from '../../api/client';
import { keys, useCreateBlock, usePiece, useSchema, useUpdateBlock } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import type { Defs, Options } from '../../components/SchemaForm';
import { useUnsaved } from '../editing';

export interface LensState {
  options: Options;
  properties: Options;
}

export function useLensDraft(project: string, name: string) {
  const lensPath = joinPath(project, `lenses/${name}.yaml`);
  const paramsPath = joinPath(project, `params/${name}.yaml`);
  const lensPiece = usePiece(lensPath);
  // (a params block that is not there yet is not asked for again)
  const paramsPiece = useQuery({
    queryKey: keys.piece(paramsPath),
    retry: false,
    queryFn: () => data(api.GET('/api/v1/pieces/{path}', { params: { path: { path: paramsPath } } })),
  });
  const schema = useSchema();
  const updateBlock = useUpdateBlock();
  const createBlock = useCreateBlock(project);
  const paramsBlock = isObject(paramsPiece.data) ? paramsPiece.data : undefined;
  const saved = useMemo<LensState | undefined>(
    () => (isObject(lensPiece.data) ? { options: lensPiece.data, properties: isObject(paramsBlock?.properties) ? paramsBlock.properties : {} } : undefined),
    [lensPiece.data, paramsBlock],
  );
  const [draft, setDraft] = useState<LensState | null>(null);
  const state = draft ?? saved;
  const dirty = draft !== null && JSON.stringify(draft) !== JSON.stringify(saved);
  useUnsaved(dirty);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<Error | null>(null);
  const save = useCallback(async () => {
    if (!draft) return;
    setSaving(true);
    setSaveError(null);
    try {
      await updateBlock.mutateAsync({ path: lensPath, content: draft.options });
      // (the other options of the params block, e.g. its priors, as they
      // were)
      const content = { ...(paramsBlock ?? {}), properties: draft.properties };
      if (paramsBlock) await updateBlock.mutateAsync({ path: paramsPath, content });
      else await createBlock.mutateAsync({ folder: 'params', name, content });
      setDraft(null);
    } catch (e) {
      setSaveError(e as Error);
    } finally {
      setSaving(false);
    }
  }, [draft, lensPath, paramsPath, paramsBlock, updateBlock, createBlock, name]);
  return {
    state,
    loaded: saved !== undefined && !paramsPiece.isPending,
    error: lensPiece.error ?? schema.error,
    defs: schema.data?.$defs as Defs | undefined,
    dirty,
    edit: (next: (state: LensState) => LensState) => state && setDraft(next(state)),
    revert: () => setDraft(null),
    save,
    saving,
    saveError,
  };
}

export type LensDraft = ReturnType<typeof useLensDraft>;

export function isObject(value: unknown): value is Options {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
