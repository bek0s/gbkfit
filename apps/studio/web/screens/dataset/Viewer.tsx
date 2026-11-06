// The view of a dataset: its cube (see CubeViewer) or its maps (see
// MapsViewer), as large as the screen allows, or what stops them from
// showing (no data yet, data being read, data of the wrong kind). The
// edits of a new dataset show over its data: the crop outlined, and the
// values left out blank.
import { useMemo, useRef, type ReactNode } from 'react';
import { useMaps } from '../../api/queries';
import { CubeViewer, type Layer } from '../../components/CubeViewer';
import type { Region } from '../../components/MapView';
import { MapsViewer, type MapItem } from '../../components/MapsViewer';
import { frameOf } from '../../components/sky';
import { useStudio } from '../../studio';
import {
  cropRegion,
  hasEdits,
  kept,
  keptValues,
  KINDS,
  NO_ARRAYS,
  presentSlots,
  rangeOf,
  ROLES,
  type Arrays,
  type Edits,
  type Item,
  type Resolved,
} from './model';

export interface Selection {
  region: Region | null;
  band: [number, number] | null;
}

export function DatasetViewer({
  resolved,
  edits,
  selection,
  onSelection,
  actions,
}: {
  resolved: Resolved;
  // The edits to show over the data (none: the data as they are)
  edits: Edits | null;
  selection: Selection;
  onSelection: (selection: Selection) => void;
  // What the head of the view has at its right (e.g. original or prepared)
  actions?: ReactNode;
}) {
  const { kind, draft, arrays } = resolved;
  const { project } = useStudio();
  const info = KINDS[kind];
  const slots = presentSlots(kind, draft);
  const shown = slots.filter((slot) => {
    const data = arrays[slot.key]?.found.data;
    return data !== undefined && kept(data).length === info.ndim;
  });
  const shape = shown.length > 0 ? kept(arrays[shown[0].key].found.data!) : null;
  // (maps of another shape than the first are not shown)
  const fitting = shown.filter((slot) => kept(arrays[slot.key].found.data!).join() === shape?.join());
  const out = useLeftOut(resolved, fitting.map((slot) => slot.key), edits);
  if (shape === null) return <Empty resolved={resolved} />;
  if (kind === 'cube') {
    const key = shown[0].key;
    const crop = edits && cropRegion(edits, shape);
    const channels = edits && rangeOf(edits.channels, shape[2]);
    return (
      <CubeViewer
        key={`${draft.items[key].files.data!.file}:${arrays[key].found.data!.index}`}
        layers={layersOf(draft.items[key], arrays[key])}
        project={project}
        frame={frameOf(arrays[key].found.data?.step, arrays[key].found.data?.rpix)}
        shape={shape}
        selection={selection.region}
        band={selection.band}
        onSelection={(region) => onSelection({ ...selection, region })}
        onBand={(band) => onSelection({ ...selection, band })}
        crop={crop}
        cropChannels={channels}
        kept={edits ? keptValues(edits, key) : null}
        actions={actions}
      />
    );
  }
  const maps: MapItem[] = fitting.map((slot) => ({
    key: slot.key,
    title: slot.title,
    colours: slot.colours,
    layers: layersOf(draft.items[slot.key], arrays[slot.key]),
  }));
  return (
    <MapsViewer
      project={project}
      frame={frameOf(arrays[fitting[0].key]?.found.data?.step, arrays[fitting[0].key]?.found.data?.rpix)}
      key={maps.map((map) => `${map.key}:${map.layers[0].path}`).join()}
      maps={maps}
      shape={shape as [number, number]}
      selection={selection.region}
      onSelection={(region) => onSelection({ ...selection, region })}
      crop={edits && cropRegion(edits, shape)}
      out={out}
      actions={actions}
    />
  );
}

// The arrays of an item to look at: its data, and its errors and mask of
// files (once described)
function layersOf(item: Item, arrays: Arrays): Layer[] {
  return ROLES.flatMap(({ role, title }) => {
    const chosen = item.files[role];
    const array = arrays.found[role];
    return chosen && array ? [{ key: role, title, path: chosen.file, hdu: array.index, unit: array.unit }] : [];
  });
}

// The pixels of maps that the edits leave out of every map (1), as far as
// the page sees them before prep: those without a value in a map, or out
// of the range of the values kept of a map (null: no edits, or a map not
// read yet)
function useLeftOut({ kind, draft, arrays }: Resolved, keys: string[], edits: Edits | null): Uint8Array | null {
  const maps = kind === 'cube' ? [] : keys.map((key) => ({ path: draft.items[key].files.data!.file, hdu: (arrays[key] ?? NO_ARRAYS).found.data!.index }));
  const values = useSame(useMaps(maps));
  return useMemo(() => {
    if (edits === null || !hasEdits(edits, keys) || values.length === 0 || values.some((v) => v === undefined)) return null;
    const out = new Uint8Array(values[0]!.length);
    keys.forEach((key, i) => {
      const [low, high] = keptValues(edits, key);
      values[i]!.forEach((at, p) => {
        if (!Number.isFinite(at) || (low !== null && at < low) || (high !== null && at > high)) out[p] = 1;
      });
    });
    return out;
    // (the keys are those of the values)
  }, [edits, values]);
}

// The same array as at the last render while its items are the same (for
// the dependencies of a memo)
function useSame<T>(items: T[]): T[] {
  const last = useRef(items);
  if (items.length !== last.current.length || items.some((item, i) => item !== last.current[i])) last.current = items;
  return last.current;
}

// What the view shows without data to look at
function Empty({ resolved }: { resolved: Resolved }) {
  const { kind, draft, arrays } = resolved;
  const info = KINDS[kind];
  const slots = presentSlots(kind, draft);
  const chosen = slots.find((slot) => draft.items[slot.key].files.data !== undefined);
  let title = kind === 'moments' ? 'No maps yet' : `No ${info.title.toLowerCase()} yet`;
  let text = `Choose ${info.what} in Files, at the right.`;
  if (chosen) {
    const of = arrays[chosen.key] ?? NO_ARRAYS;
    const file = draft.items[chosen.key].files.data!.file.split('/').pop();
    const data = of.found.data;
    if (of.errors.data) {
      title = `${file} cannot be read`;
      text = of.errors.data;
    } else if (data === undefined) {
      title = `Reading ${file}…`;
      text = '';
    } else {
      const count = kept(data).length;
      title = `${file} is not ${kind === 'cube' ? 'a cube' : 'a map'}`;
      text =
        `It has ${count} ${count === 1 ? 'axis' : 'axes'} of more than one pixel, not ${info.ndim}. ` +
        (kind === 'cube' && count === 2 ? 'A map is a dataset of moment maps or of a flux map: add it as one of those.' : 'Choose another file, or another array of it.');
    }
  }
  return (
    <div className="ds-empty">
      <p className="ds-empty-title">{title}</p>
      {text && <p>{text}</p>}
    </div>
  );
}
