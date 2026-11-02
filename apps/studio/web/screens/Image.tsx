// A dataset of a flux map (pixel_brightness): an image of the light, an
// item of its files (its data, their errors and their mask; see items), to
// look at (see MapsViewer), with what GBKFIT reads from its header. A new
// one is named, given its files and their edits (those of moment maps, of
// one map; see MapEdits), and added to the project once nothing stops it:
// of its files as they are, or of the copies that prep edits as it is
// added. One of the project is shown as it is.
import { useEffect, useMemo, useRef, useState } from 'react';
import { useChannel, useCreateBlock, useJob, useProject, useSubmitJob } from '../api/queries';
import { PanelsIcon } from '../components/icons';
import type { Region } from '../components/MapView';
import { MapsViewer } from '../components/MapsViewer';
import { Button, IconButton } from '../components/ui/basics';
import { SidePanes, type PanesControl } from '../components/ui/panes';
import { useStudio } from '../studio';
import type { Prepared } from './Edits';
import {
  errorKindOf,
  errorSibling,
  Facts,
  FilesCard,
  itemBlocker,
  itemOptions,
  itemSources,
  kept,
  layersOf,
  NO_SOURCES,
  prepFile,
  useArrays,
  type Arrays,
  type Sources,
} from './items';
import { hasMapEdits, leftOut, mapCropOf, mapPrepOptions, MapEditsCard, NO_MAP_EDITS, withMapRegion, type MapEdits } from './MapEdits';

// The key of the map in its edits
const KEY = 'image';

export function NewImage() {
  const { project, go } = useStudio();
  const info = useProject(project);
  const create = useCreateBlock(project ?? '.');
  const submit = useSubmitJob();
  const [name, setName] = useState('');
  const [sources, setSources] = useState<Sources>(NO_SOURCES);
  const [edits, setEdits] = useState<MapEdits>(NO_MAP_EDITS);
  // The job of prep that edits the files as the dataset is added
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId).data;
  const added = useRef<string | null>(null);
  const arrays = useArrays(sources, 2);
  // (the errors of an array of the file of the data, named as errors, are
  // given once for that file; removed, they stay removed)
  const guessed = useRef<string | null>(null);
  useEffect(() => {
    const data = sources.files.data;
    if (data === undefined || arrays.found.data === undefined || guessed.current === data.file) return;
    guessed.current = data.file;
    const errors = errorSibling(arrays);
    if (errors !== undefined && sources.files.error === undefined && sources.errorValue === null) {
      setSources({ ...sources, files: { ...sources.files, error: { file: data.file, array: errors } } });
    }
  }, [sources, arrays]);
  const shape = shapeOf(arrays);
  const reason0 = itemBlocker('flux map', arrays, sources, 2);
  const prep = project !== null && reason0 === null && shape !== undefined && hasMapEdits(edits, [KEY]);
  const request = prep ? prepOptions(edits, sources, arrays, shape, project) : null;
  const finish = (content: Record<string, unknown>) =>
    create.mutate({ folder: 'datasets', name: name.trim(), content }, { onSuccess: () => go({ kind: 'block', folder: 'datasets', name: name.trim() }) });
  // (the files edited, the dataset of the copies is added)
  useEffect(() => {
    if (project !== null && job?.state === 'complete' && job.files && added.current !== job.id) {
      added.current = job.id;
      finish(preparedBlock(job.files as Prepared, sources, project));
    }
  });
  if (project === null) return null;
  const taken = info.data?.pieces.datasets.includes(name.trim()) ?? false;
  const reason =
    reason0 ??
    (typeof request === 'string' ? request : null) ??
    (name.trim() === '' ? 'Name the dataset' : taken ? 'The project has a dataset of that name' : null);
  const busy = submit.isPending || job?.state === 'queued' || job?.state === 'running' || create.isPending;
  const add = () => {
    if (request === null) finish(toBlock(arrays, sources, project));
    else if (typeof request !== 'string') {
      submit.mutate({ task: 'prep_pixel_brightness', options: request }, { onSuccess: (submitted) => setJobId(submitted.id) });
    }
  };
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">New dataset · Flux map</div>
          <input
            className="name-input"
            autoFocus
            value={name}
            placeholder="Name of the dataset"
            aria-label="Name of the dataset"
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        <div className="head-actions">
          {busy ? (
            <span className="note">{create.isPending ? 'Adding…' : 'Writing the edited copy…'}</span>
          ) : (
            reason && <span className="why">{reason}</span>
          )}
          <Button onClick={() => go({ kind: 'home' })}>Cancel</Button>
          <Button kind="primary" disabled={reason !== null || busy} onClick={add}>
            Add to the project
          </Button>
        </div>
      </div>
      {(create.error ?? submit.error) && <p className="error screen-error">{(create.error ?? submit.error)?.message}</p>}
      {job?.state === 'failed' && <pre className="error screen-error">{job.output.trimEnd().split('\n').slice(-4).join('\n')}</pre>}
      <ImageBody
        project={project}
        sources={sources}
        arrays={arrays}
        onChange={(changed) => {
          setSources(changed);
          // (other files, other edits)
          if (changed.files.data?.file !== sources.files.data?.file) setEdits(NO_MAP_EDITS);
        }}
        edits={edits}
        onEdits={setEdits}
      />
    </section>
  );
}

// A dataset of a flux map of the project, of its options (as its piece
// gives them)
export function ImageDataset({ options, project }: { options: Record<string, unknown>; project: string }) {
  const sources = itemSources(options, project);
  const arrays = useArrays(sources, 2);
  return <ImageBody project={project} sources={sources} arrays={arrays} />;
}

// The shape of the map (x, y), once read, if it is one
function shapeOf(arrays: Arrays): [number, number] | undefined {
  const data = arrays.found.data;
  return data !== undefined && kept(data).length === 2 ? (kept(data) as [number, number]) : undefined;
}

// The options of a dataset of a flux map: its type, and its item
function toBlock(arrays: Arrays, sources: Sources, project: string): Record<string, unknown> {
  return { type: 'pixel_brightness', ...itemOptions(arrays, sources, project) };
}

// The options of prep_pixel_brightness of the edits (those of one map of
// prep_pixel_moments, of one value each), or why they cannot be
function prepOptions(edits: MapEdits, sources: Sources, arrays: Arrays, shape: [number, number], project: string): Record<string, unknown> | string {
  const files = {
    file_d: prepFile(sources, arrays, 'data')!,
    file_e: sources.errorValue === null ? prepFile(sources, arrays, 'error') : null,
    file_m: prepFile(sources, arrays, 'mask'),
  };
  const options = mapPrepOptions(edits, [{ key: KEY, files, errorKind: errorKindOf(sources, arrays) }], shape, project);
  if (typeof options === 'string') return options;
  const first = (list: unknown) => (Array.isArray(list) ? (list[0] ?? null) : null);
  return {
    ...options,
    file_d: files.file_d,
    file_e: files.file_e,
    file_m: files.file_m,
    clip_min: first(options.clip_min),
    clip_max: first(options.clip_max),
    sclip_sigma: first(options.sclip_sigma),
  };
}

// The options of a dataset of the files that prep edited (paths of the
// workspace): its errors are those prepared (standard deviations), or else
// stay one value
function preparedBlock(prepared: Prepared, sources: Sources, project: string): Record<string, unknown> {
  const prefix = project === '.' ? '' : `${project}/`;
  const relative = (path: string) => (path.startsWith(prefix) ? path.slice(prefix.length) : path);
  const options: Record<string, unknown> = { type: 'pixel_brightness', data: relative(prepared.file_d!) };
  if (prepared.file_e !== undefined) options.error = relative(prepared.file_e);
  else if (sources.errorValue !== null) options.error = Number(sources.errorValue);
  if (prepared.file_m !== undefined) options.mask = relative(prepared.file_m);
  return options;
}

// The map, and at the side its files, the edits of a new one, and what
// GBKFIT reads from its header; the files and edits can change when it
// says how
function ImageBody({
  project,
  sources,
  arrays,
  onChange,
  edits,
  onEdits,
}: {
  project: string;
  sources: Sources;
  arrays: Arrays;
  onChange?: (sources: Sources) => void;
  edits?: MapEdits;
  onEdits?: (edits: MapEdits) => void;
}) {
  const [selection, setSelection] = useState<Region | null>(null);
  const data = arrays.found.data;
  const shape = shapeOf(arrays);
  const path = (shape && sources.files.data?.file) ?? null;
  const values = useChannel(path, data?.index ?? 0, 0).data?.plane.values;
  const edited = [{ key: KEY, title: 'Flux', unit: data?.unit ?? null, values }];
  const out = useMemo(() => (edits ? leftOut(edits, edited) : null), [edits, values]);
  const viewer = ({ folded, toggle }: PanesControl) =>
    shape && data ? (
      <MapsViewer
        key={`${sources.files.data!.file}:${data.index}`}
        maps={[{ key: KEY, title: 'Flux', short: 'F', colours: 'sequential', layers: layersOf(sources, arrays) }]}
        shape={shape}
        selection={selection}
        onSelection={setSelection}
        crop={edits ? mapCropOf(edits, shape) : null}
        out={out}
        actions={
          <IconButton label={folded ? 'Show the panels at the side' : 'Hide the panels at the side'} pressed={!folded} onClick={toggle}>
            <PanelsIcon />
          </IconButton>
        }
      />
    ) : (
      <div className="card viewer-empty">
        <p className="note">
          {data && !shape ? `The data have ${kept(data).length} axes; a flux map has 2.` : 'Import or upload the flux map (at the side) to see it here.'}
        </p>
      </div>
    );
  const column = (
    <div className="col">
      <FilesCard project={project} sources={sources} arrays={arrays} onChange={onChange} />
      {edits && onEdits && shape && (
        <MapEditsCard
          maps={edited}
          shape={shape}
          edits={edits}
          onChange={onEdits}
          onCropToSelection={selection === null ? undefined : () => onEdits(withMapRegion(edits, selection))}
        />
      )}
      <Facts array={data} velocity={null} cube={false} />
    </div>
  );
  return (
    <div className="screen-body dataset">
      <SidePanes main={viewer} side={column} />
    </div>
  );
}
