// The hooks of the dataset screen: a draft resolved against the server's
// descriptions of its files, the state of a draft as the user edits it,
// and a job of prep that the screen follows.
import { useDebouncedValue } from '@mantine/hooks';
import { useState } from 'react';
import { useDescriptions, useJob, useSubmitJob } from '../../api/queries';
import {
  arraysOf,
  KINDS,
  restOf,
  type Draft,
  type Edits,
  type Item,
  type Kind,
  type PrepRequest,
  type Resolved,
  type Spectral,
} from './model';

// A draft with the arrays of the files of its items, as the server
// describes them: with the convention of velocities given, and the rest
// of the spectral axis of a cube (as it is typed: once it has stood a
// while, not at each key, which would read the files again at each)
export function useResolved(kind: Kind, draft: Draft): Resolved {
  const items = Object.values(draft.items);
  const paths = items.flatMap((item) => Object.values(item.files).map((file) => file.file));
  const [rest] = useDebouncedValue(kind === 'cube' ? restOf(draft.spectral) : null, TYPED_MS);
  const describe = useDescriptions(paths, draft.spectral.convention, rest, kind === 'cube' ? draft.spectral.type : null);
  const arrays = Object.fromEntries(Object.entries(draft.items).map(([key, item]) => [key, arraysOf(item, describe, KINDS[kind].ndim)]));
  return { kind, draft, arrays };
}

// (how long a value typed stands before it is used, ms)
const TYPED_MS = 600;

export interface DraftState {
  draft: Draft;
  setItem: (key: string, item: Item) => void;
  addItem: (key: string, item: Item) => void;
  removeItem: (key: string) => void;
  setSpectral: (spectral: Spectral) => void;
  setEdits: (edits: Edits) => void;
  reset: (draft: Draft) => void;
}

export function useDraftState(initial: Draft): DraftState {
  const [draft, setDraft] = useState(initial);
  return {
    draft,
    setItem: (key, item) =>
      setDraft((now) => {
        // (errors of a file: none made from the noise)
        const edits = item.files.error !== undefined && now.edits.noise ? { ...now.edits, noise: false } : now.edits;
        return { ...now, items: { ...now.items, [key]: item }, edits };
      }),
    addItem: (key, item) => setDraft((now) => ({ ...now, items: { ...now.items, [key]: item } })),
    removeItem: (key) =>
      setDraft((now) => {
        const { [key]: _gone, ...items } = now.items;
        return { ...now, items };
      }),
    setSpectral: (spectral) => setDraft((now) => ({ ...now, spectral })),
    setEdits: (edits) => setDraft((now) => ({ ...now, edits })),
    reset: setDraft,
  };
}

// A job of prep that the screen starts and follows: the request it ran
// (to tell whether it is still that of the edits), and the job
export function usePrepJob() {
  const submit = useSubmitJob();
  const [started, setStarted] = useState<{ id: string; key: string } | null>(null);
  const job = useJob(started?.id ?? null).data;
  const start = (request: PrepRequest, preview: boolean) => {
    const key = requestKey(request);
    submit.mutate({ task: request.task, options: request.options, preview }, { onSuccess: (submitted) => setStarted({ id: submitted.id, key }) });
  };
  const running = submit.isPending || job?.state === 'queued' || job?.state === 'running';
  return {
    start,
    job,
    key: started?.key ?? null,
    running,
    error: submit.error?.message ?? (job?.state === 'failed' ? lastLines(job.output, 6) : null),
    clear: () => setStarted(null),
  };
}

export function requestKey(request: PrepRequest): string {
  return JSON.stringify(request);
}

function lastLines(text: string, count: number): string {
  return text.trimEnd().split('\n').slice(-count).join('\n');
}
