// A dataset of moment maps (pixel_moments): maps of moments of the spectra
// on one grid (moment 0 to 7) and of the coefficients of a Gauss-Hermite
// series (h3, h4), each an item of its files (its data, their errors and
// their mask; see items), to look at (see MapsViewer), with what GBKFIT
// reads from the header of the first. A new one has the maps of moments 0,
// 1 and 2 at first, which can go, and others can be added; all must be of
// one shape (one of another is not shown). Its maps have one set of edits
// (see MapEdits); it is named and added to the project once nothing stops
// it: of its files as they are, or of the copies that prep edits as it is
// added. One of the project is shown as it is.
import { useEffect, useMemo, useRef, useState } from 'react';
import { useChannel, useCreateBlock, useJob, useProject, useSubmitJob } from '../api/queries';
import { PanelsIcon } from '../components/icons';
import type { Region } from '../components/MapView';
import { MapsViewer, type MapItem } from '../components/MapsViewer';
import { Button, Close, IconButton } from '../components/ui/basics';
import { DropMenu } from '../components/ui/overlays';
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

// The maps a dataset can have, as the options of pixel_moments name them:
// their titles, the short names on the rail, and how their colours run
const MAPS = [
  { key: 'moment0', title: 'Intensity', short: '0', colours: 'sequential' },
  { key: 'moment1', title: 'Velocity', short: '1', colours: 'median' },
  { key: 'moment2', title: 'Dispersion', short: '2', colours: 'sequential' },
  { key: 'moment3', title: 'Moment 3', short: '3', colours: 'zero' },
  { key: 'moment4', title: 'Moment 4', short: '4', colours: 'zero' },
  { key: 'moment5', title: 'Moment 5', short: '5', colours: 'zero' },
  { key: 'moment6', title: 'Moment 6', short: '6', colours: 'zero' },
  { key: 'moment7', title: 'Moment 7', short: '7', colours: 'zero' },
  { key: 'h3', title: 'Gauss-Hermite h3', short: 'h3', colours: 'zero' },
  { key: 'h4', title: 'Gauss-Hermite h4', short: 'h4', colours: 'zero' },
] as const;

type MapKey = (typeof MAPS)[number]['key'];

const FIRST: MapKey[] = ['moment0', 'moment1', 'moment2'];

type Maps = Partial<Record<MapKey, Sources>>;

// The arrays of the files of each map (of every map a dataset can have, so
// that the hooks are the same at each render)
function useMapArrays(maps: Maps): Record<MapKey, Arrays> {
  const arrays = {} as Record<MapKey, Arrays>;
  for (const { key } of MAPS) arrays[key] = useArrays(maps[key] ?? NO_SOURCES, 2);
  return arrays;
}

// The values of the data of each map, once read (of every map a dataset
// can have, so that the hooks are the same at each render)
function useMapValues(maps: Maps, arrays: Record<MapKey, Arrays>): Partial<Record<MapKey, Float32Array>> {
  const values: Partial<Record<MapKey, Float32Array>> = {};
  for (const { key } of MAPS) {
    const data = arrays[key].found.data;
    const path = (maps[key] && data && maps[key].files.data?.file) ?? null;
    const plane = useChannel(path, data?.index ?? 0, 0).data?.plane;
    if (path !== null && plane !== undefined) values[key] = plane.values;
  }
  return values;
}

// The maps a dataset has, in their order
function present(maps: Maps): (typeof MAPS)[number][] {
  return MAPS.filter(({ key }) => maps[key] !== undefined);
}

// The shape of the maps (x, y): that of the first readable map (none yet)
function shapeOf(maps: Maps, arrays: Record<MapKey, Arrays>): [number, number] | undefined {
  for (const { key } of present(maps)) {
    const data = arrays[key].found.data;
    if (data !== undefined && kept(data).length === 2) return kept(data) as [number, number];
  }
  return undefined;
}

// Why a map is not of the shape of the maps (null: it is, or it is no map)
function misfit(arrays: Arrays, shape: [number, number] | undefined): string | null {
  const data = arrays.found.data;
  if (data === undefined || shape === undefined || kept(data).length !== 2) return null;
  const [nx, ny] = kept(data);
  return nx === shape[0] && ny === shape[1]
    ? null
    : `This map is ${nx} × ${ny} pixels; the first is ${shape[0]} × ${shape[1]}. All the maps must be of one shape.`;
}

export function NewMoments() {
  const { project, go } = useStudio();
  const info = useProject(project);
  const create = useCreateBlock(project ?? '.');
  const [name, setName] = useState('');
  const submit = useSubmitJob();
  const [maps, setMaps] = useState<Maps>(() => Object.fromEntries(FIRST.map((key) => [key, NO_SOURCES])));
  const [edits, setEdits] = useState<MapEdits>(NO_MAP_EDITS);
  // The job of prep that edits the files as the dataset is added
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId).data;
  const added = useRef<string | null>(null);
  const arrays = useMapArrays(maps);
  // (the errors of a map in an array of the file of its data, named as its
  // errors, e.g. MOM1_ERR of MOM1, are guessed for each array of its data,
  // and follow it while they are those guessed; removed or chosen, they
  // stay so)
  const guessed = useRef<Partial<Record<MapKey, { at: string; error: string | number | null }>>>({});
  useEffect(() => {
    let changed: Maps | null = null;
    for (const { key } of MAPS) {
      const sources = maps[key];
      const data = sources?.files.data;
      const array = arrays[key].found.data;
      if (!sources || data === undefined || array === undefined) continue;
      const at = `${data.file}:${array.index}`;
      const last = guessed.current[key];
      if (last?.at === at) continue;
      const error = sources.files.error;
      const ours = error === undefined ? last === undefined || last.error === null : error.file === data.file && error.array === last?.error;
      if (!ours || sources.errorValue !== null) {
        guessed.current[key] = { at, error: null };
        continue;
      }
      const next = errorSibling(arrays[key]) ?? null;
      guessed.current[key] = { at, error: next };
      const files = { ...sources.files };
      if (next === null) delete files.error;
      else files.error = { file: data.file, array: next };
      changed = { ...(changed ?? maps), [key]: { ...sources, files } };
    }
    if (changed !== null) setMaps(changed);
  }, [maps, arrays]);
  const shown = present(maps);
  const shape = shapeOf(maps, arrays);
  const reason0 = blocker(maps, arrays);
  const prep = project !== null && reason0 === null && shape !== undefined && hasMapEdits(edits, shown.map(({ key }) => key));
  const request = prep
    ? mapPrepOptions(
        edits,
        shown.map(({ key }) => ({ key, files: prepFiles(maps[key]!, arrays[key]), errorKind: errorKindOf(maps[key]!, arrays[key]) })),
        shape,
        project,
      )
    : null;
  const finish = (content: Record<string, unknown>) =>
    create.mutate(
      { folder: 'datasets', name: name.trim(), content },
      { onSuccess: () => go({ kind: 'block', folder: 'datasets', name: name.trim() }) },
    );
  // (the files edited, the dataset of the copies is added)
  useEffect(() => {
    if (project !== null && job?.state === 'complete' && job.files && added.current !== job.id) {
      added.current = job.id;
      finish(preparedBlock(shown.map(({ key }) => key), job.files, maps, project));
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
    if (request === null) finish(toBlock(maps, arrays, project));
    else if (typeof request !== 'string') {
      submit.mutate({ task: 'prep_pixel_moments', options: request }, { onSuccess: (submitted) => setJobId(submitted.id) });
    }
  };
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">New dataset · Moment maps</div>
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
            <span className="note">{create.isPending ? 'Adding…' : 'Writing the edited copies…'}</span>
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
      <MomentsBody project={project} maps={maps} arrays={arrays} onChange={setMaps} edits={edits} onEdits={setEdits} />
    </section>
  );
}

// A dataset of moment maps of the project, of its options (as its piece
// gives them)
export function Moments({ options, project }: { options: Record<string, unknown>; project: string }) {
  const maps: Maps = {};
  for (const { key } of MAPS) {
    const item = options[key];
    if (item !== undefined && item !== null && typeof item === 'object') maps[key] = itemSources(item as Record<string, unknown>, project);
  }
  const arrays = useMapArrays(maps);
  return <MomentsBody project={project} maps={maps} arrays={arrays} />;
}

// Why the maps cannot be a dataset yet (null: they can): each must be
// readable and of one shape, as all of them must be
function blocker(maps: Maps, arrays: Record<MapKey, Arrays>): string | null {
  const shown = present(maps);
  if (shown.length === 0) return 'Add a map';
  for (const { key, title } of shown) {
    const reason = itemBlocker(`map of ${title.toLowerCase()}`, arrays[key], maps[key]!, 2);
    if (reason !== null) return reason;
  }
  const shape = shapeOf(maps, arrays);
  const other = shown.find(({ key }) => misfit(arrays[key], shape) !== null);
  if (other) return `The map of ${other.title.toLowerCase()} is not of the shape of the first`;
  return null;
}

// The files of a map as prep takes them
function prepFiles(sources: Sources, arrays: Arrays) {
  return {
    file_d: prepFile(sources, arrays, 'data')!,
    file_e: sources.errorValue === null ? prepFile(sources, arrays, 'error') : null,
    file_m: prepFile(sources, arrays, 'mask'),
  };
}

// The options of a dataset of the maps that prep edited (paths of the
// workspace, a list of each role, of the maps in their order): the errors
// of a map are those prepared (standard deviations), or else stay one
// value
function preparedBlock(keys: MapKey[], prepared: Record<string, string | (string | null)[]>, maps: Maps, project: string): Record<string, unknown> {
  const prefix = project === '.' ? '' : `${project}/`;
  const relative = (path: string) => (path.startsWith(prefix) ? path.slice(prefix.length) : path);
  const options: Record<string, unknown> = { type: 'pixel_moments' };
  keys.forEach((key, i) => {
    const of = (role: keyof Prepared) => (prepared[role] as (string | null)[] | undefined)?.[i] ?? null;
    const item: Record<string, unknown> = { data: relative(of('file_d')!) };
    const error = of('file_e');
    if (error !== null) item.error = relative(error);
    else if (maps[key]!.errorValue !== null) item.error = Number(maps[key]!.errorValue);
    const mask = of('file_m');
    if (mask !== null) item.mask = relative(mask);
    options[key] = item;
  });
  return options;
}

// The options of a dataset of moment maps: its type, and the item of each
// of its maps
function toBlock(maps: Maps, arrays: Record<MapKey, Arrays>, project: string): Record<string, unknown> {
  const options: Record<string, unknown> = { type: 'pixel_moments' };
  for (const { key } of present(maps)) options[key] = itemOptions(arrays[key], maps[key]!, project);
  return options;
}

// The maps of a dataset, and at the side the files of each, the edits of
// a new one, and what GBKFIT reads from those of the first; the maps and
// the edits can change when it says how
function MomentsBody({
  project,
  maps,
  arrays,
  onChange,
  edits,
  onEdits,
}: {
  project: string;
  maps: Maps;
  arrays: Record<MapKey, Arrays>;
  onChange?: (maps: Maps) => void;
  edits?: MapEdits;
  onEdits?: (edits: MapEdits) => void;
}) {
  const [selection, setSelection] = useState<Region | null>(null);
  const shown = present(maps);
  const shape = shapeOf(maps, arrays);
  const values = useMapValues(maps, arrays);
  // (the maps to look at: those whose data are maps of their shape)
  const fitting = shown.filter(({ key }) => {
    const data = arrays[key].found.data;
    return data !== undefined && kept(data).length === 2 && misfit(arrays[key], shape) === null;
  });
  const items: MapItem[] = fitting.map(({ key, title, short, colours }) => ({ key, title, short, colours, layers: layersOf(maps[key]!, arrays[key]) }));
  const first = items.length > 0 ? arrays[items[0].key as MapKey].found.data : undefined;
  const edited = fitting.map(({ key, title }) => ({ key, title, unit: arrays[key].found.data?.unit ?? null, values: values[key] }));
  // (of the same maps and edits, the same pixels: of every map a dataset
  // can have, so that its dependencies are as many at each render)
  const fits = fitting.map(({ key }) => key).join();
  const out = useMemo(() => (edits ? leftOut(edits, edited) : null), [edits, fits, ...MAPS.map(({ key }) => values[key])]);
  const absent = MAPS.filter(({ key }) => maps[key] === undefined);
  const viewer = ({ folded, toggle }: PanesControl) =>
    items.length > 0 && shape ? (
      <MapsViewer
        key={items.map((item) => item.key).join()}
        maps={items}
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
        <p className="note">Import or upload the maps (at the side) to see them here.</p>
      </div>
    );
  const column = (
    <div className="col">
      {shown.map(({ key, title, short }) => (
        <FilesCard
          key={key}
          id={`moments-${key}`}
          title={`${short === 'h3' || short === 'h4' ? '' : `Moment ${short} · `}${title}`}
          project={project}
          sources={maps[key]!}
          arrays={arrays[key]}
          problem={misfit(arrays[key], shape)}
          ndim={2}
          onChange={onChange && ((sources) => onChange({ ...maps, [key]: sources }))}
          actions={
            onChange && (
              <Close
                label={`Remove the map of ${title.toLowerCase()}`}
                onClick={() => onChange(Object.fromEntries(Object.entries(maps).filter(([k]) => k !== key)))}
              />
            )
          }
        />
      ))}
      {onChange && absent.length > 0 && (
        <div className="add-map">
          <DropMenu
            label="Add a map"
            entries={absent.map(({ key, title, short }) => ({
              label: `${short === 'h3' || short === 'h4' ? '' : `Moment ${short} · `}${title}`,
              onClick: () => onChange({ ...maps, [key]: NO_SOURCES }),
            }))}
            target={<Button kind="small">+ Add a map</Button>}
          />
        </div>
      )}
      {edits && onEdits && shape && fitting.length > 0 && (
        <MapEditsCard
          maps={edited}
          shape={shape}
          edits={edits}
          onChange={onEdits}
          onCropToSelection={selection === null ? undefined : () => onEdits(withMapRegion(edits, selection))}
        />
      )}
      <Facts array={first} velocity={null} cube={false} />
    </div>
  );
  return (
    <div className="screen-body dataset">
      <SidePanes main={viewer} side={column} />
    </div>
  );
}
