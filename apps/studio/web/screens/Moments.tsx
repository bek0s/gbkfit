// A dataset of moment maps (pixel_moments): maps of moments of the spectra
// on one grid (moment 0 to 7) and of the coefficients of a Gauss-Hermite
// series (h3, h4), each an item of its files (its data, their errors and
// their mask; see items), to look at (see MapsViewer), with what GBKFIT
// reads from the header of the first. A new one has the maps of moments 0,
// 1 and 2 at first, which can go, and others can be added; it is named
// and added to the project once nothing stops it. One of the project is
// shown as it is.
import { useState } from 'react';
import { useCreateBlock, useProject } from '../api/queries';
import { PanelsIcon } from '../components/icons';
import type { Region } from '../components/MapView';
import { MapsViewer, type MapItem } from '../components/MapsViewer';
import { Button, Close, IconButton } from '../components/ui/basics';
import { DropMenu } from '../components/ui/overlays';
import { SidePanes, type PanesControl } from '../components/ui/panes';
import { useStudio } from '../studio';
import { Facts, FilesCard, itemBlocker, itemOptions, itemSources, kept, layersOf, NO_SOURCES, useArrays, type Arrays, type Sources } from './items';

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

// The maps a dataset has, in their order
function present(maps: Maps): (typeof MAPS)[number][] {
  return MAPS.filter(({ key }) => maps[key] !== undefined);
}

export function NewMoments() {
  const { project, go } = useStudio();
  const info = useProject(project);
  const create = useCreateBlock(project ?? '.');
  const [name, setName] = useState('');
  const [maps, setMaps] = useState<Maps>(() => Object.fromEntries(FIRST.map((key) => [key, NO_SOURCES])));
  const arrays = useMapArrays(maps);
  if (project === null) return null;
  const taken = info.data?.pieces.datasets.includes(name.trim()) ?? false;
  const reason =
    blocker(maps, arrays) ?? (name.trim() === '' ? 'Name the dataset' : taken ? 'The project has a dataset of that name' : null);
  const add = () =>
    create.mutate(
      { folder: 'datasets', name: name.trim(), content: toBlock(maps, arrays, project) },
      { onSuccess: () => go({ kind: 'block', folder: 'datasets', name: name.trim() }) },
    );
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
          {create.isPending ? <span className="note">Adding…</span> : reason && <span className="why">{reason}</span>}
          <Button onClick={() => go({ kind: 'home' })}>Cancel</Button>
          <Button kind="primary" disabled={reason !== null || create.isPending} onClick={add}>
            Add to the project
          </Button>
        </div>
      </div>
      {create.error && <p className="error screen-error">{create.error.message}</p>}
      <MomentsBody project={project} maps={maps} arrays={arrays} onChange={setMaps} />
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
  const shapes = new Set(shown.map(({ key }) => kept(arrays[key].found.data!).join()));
  if (shapes.size > 1) return 'The maps are not of one shape';
  return null;
}

// The options of a dataset of moment maps: its type, and the item of each
// of its maps
function toBlock(maps: Maps, arrays: Record<MapKey, Arrays>, project: string): Record<string, unknown> {
  const options: Record<string, unknown> = { type: 'pixel_moments' };
  for (const { key } of present(maps)) options[key] = itemOptions(arrays[key], maps[key]!, project);
  return options;
}

// The maps of a dataset, and at the side the files of each and what GBKFIT
// reads from those of the first; the maps can change when it says how
function MomentsBody({
  project,
  maps,
  arrays,
  onChange,
}: {
  project: string;
  maps: Maps;
  arrays: Record<MapKey, Arrays>;
  onChange?: (maps: Maps) => void;
}) {
  const [selection, setSelection] = useState<Region | null>(null);
  const shown = present(maps);
  // (the maps to look at: those whose data are maps)
  const items: MapItem[] = shown
    .filter(({ key }) => arrays[key].found.data !== undefined && kept(arrays[key].found.data!).length === 2)
    .map(({ key, title, short, colours }) => ({ key, title, short, colours, layers: layersOf(maps[key]!, arrays[key]) }));
  const first = items.length > 0 ? arrays[items[0].key as MapKey].found.data : undefined;
  const shape = first && (kept(first) as [number, number]);
  const absent = MAPS.filter(({ key }) => maps[key] === undefined);
  const viewer = ({ folded, toggle }: PanesControl) =>
    items.length > 0 && shape ? (
      <MapsViewer
        key={items.map((item) => item.key).join()}
        maps={items}
        shape={shape}
        selection={selection}
        onSelection={setSelection}
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
      <Facts array={first} velocity={null} cube={false} />
    </div>
  );
  return (
    <div className="screen-body dataset">
      <SidePanes main={viewer} side={column} />
    </div>
  );
}
