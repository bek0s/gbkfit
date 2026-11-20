// The hooks of the dataset screen: a draft resolved against the server's
// descriptions of its files, the state of a draft as the user edits it,
// and a job of prep that the screen follows.
import { useDebouncedValue } from '@mantine/hooks';
import { useMemo, useState } from 'react';
import { useDescriptions, useJob, useSubmitJob, type Reading } from '../../api/queries';
import {
  arraysOf,
  footprintOf,
  KINDS,
  NO_EDITS,
  NO_SPECTRAL,
  restOf,
  type Draft,
  type Edits,
  type FileArray,
  type Item,
  type Kind,
  type PrepRequest,
  type Resolved,
  type Spectral,
} from './model';

// A draft with the arrays of the files of its items, as the server
// describes them, and how the data of its first item are read (by those
// descriptions and by the views of its cube): the convention of
// velocities given, of a cube the type and the unit of its axis and its
// rest, and the unit of its values given (each as it is typed: once it has
// stood a while, not at each key, which would read the files again at
// each). The data of each item are described with the unit given of them.
export function useResolved(kind: Kind, draft: Draft): Resolved {
  const items = Object.values(draft.items);
  const paths = [...items.flatMap((item) => Object.values(item.files).map((file) => file.file)), ...(draft.footprint ? [draft.footprint.file] : [])];
  const cube = kind === 'cube';
  const [rest] = useDebouncedValue(cube ? restOf(draft.spectral) : null, TYPED_MS);
  const [units] = useDebouncedValue(JSON.stringify(items.map((item) => [item.files.data?.file ?? null, item.unit.trim() || null])), TYPED_MS);
  const { convention, type, axisUnit } = draft.spectral;
  const base = useMemo<Reading>(
    () => ({ rest, velocity: convention, type: cube ? type : null, spectralUnit: cube ? axisUnit || null : null, dataUnit: null }),
    [rest, convention, cube, type, axisUnit],
  );
  const unitOf = useMemo(() => new Map(JSON.parse(units) as [string | null, string | null][]), [units]);
  const readingOf = (path: string): Reading => ({ ...base, dataUnit: unitOf.get(path) ?? null });
  const first = items[0]?.files.data?.file;
  const reading = useMemo<Reading>(() => ({ ...base, dataUnit: first ? (unitOf.get(first) ?? null) : null }), [base, first, unitOf]);
  const describe = useDescriptions(paths, readingOf);
  const arrays = Object.fromEntries(Object.entries(draft.items).map(([key, item]) => [key, arraysOf(item, describe, KINDS[kind].ndim)]));
  const footprint = footprintOf(draft.footprint, describe, Object.values(arrays)[0]?.found.data);
  return { kind, draft, arrays, footprint, reading };
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
  setFootprint: (footprint: FileArray | undefined) => void;
  reset: (draft: Draft) => void;
}

export function useDraftState(initial: Draft): DraftState {
  const [draft, setDraft] = useState(initial);
  return {
    draft,
    setItem: (key, item) =>
      setDraft((now) => {
        const next = withItems(now, { ...now.items, [key]: item });
        // (errors of a file: none made from the noise)
        return item.files.error !== undefined && next.edits.noise ? { ...next, edits: { ...next.edits, noise: false } } : next;
      }),
    addItem: (key, item) => setDraft((now) => withItems(now, { ...now.items, [key]: item })),
    removeItem: (key) =>
      setDraft((now) => {
        const { [key]: _gone, ...items } = now.items;
        return withItems(now, items);
      }),
    setSpectral: (spectral) => setDraft((now) => ({ ...now, spectral })),
    setEdits: (edits) => setDraft((now) => ({ ...now, edits })),
    setFootprint: (footprint) => setDraft((now) => ({ ...now, footprint })),
    reset: setDraft,
  };
}

// The data of the items of a draft (the file and the array of each), as
// a key: what its spectral axis, its edits, its jobs of prep and the
// selections of its view are of
export function dataKey(items: Record<string, Item>): string {
  return JSON.stringify(Object.entries(items).map(([key, item]) => [key, item.files.data?.file ?? null, item.files.data?.array ?? null]));
}

// A draft with other items. Its spectral axis, its edits and its footprint
// are of the data of its items (what their headers leave to say, their
// pixels and channels): if the data of an item are another file or array,
// or go, they start again, so that nothing said of one file is taken for
// another (an item added keeps them).
function withItems(now: Draft, items: Record<string, Item>): Draft {
  const changed = Object.entries(now.items).some(([key, item]) => {
    const before = item.files.data;
    const after = items[key]?.files.data;
    return before !== undefined && (after?.file !== before.file || after?.array !== before.array);
  });
  return changed ? { items, spectral: NO_SPECTRAL, edits: NO_EDITS } : { ...now, items };
}

// A job of prep that the screen starts and follows: the request it ran
// (to tell whether it is still that of the edits), and the job; of the
// data it was started for (see dataKey): of other data, none
export function usePrepJob(of: string) {
  const submit = useSubmitJob();
  const [begun, setStarted] = useState<{ id: string; key: string; of: string } | null>(null);
  const started = begun?.of === of ? begun : null;
  const job = useJob(started?.id ?? null).data;
  const start = (request: PrepRequest, preview: boolean) => {
    const key = requestKey(request);
    submit.mutate({ task: request.task, options: request.options, preview }, { onSuccess: (submitted) => setStarted({ id: submitted.id, key, of }) });
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
