// The edits of the files of new moment maps, which prep makes when the
// dataset is added (see gbkfit.tasks.prep.prep_pixel_moments): one set for
// all the maps, as their pixels must stay the same: crop them (typed, or
// to the selection on the map), leave out values (out of a range, or
// outliers: of each map), and of the pixels left the regions out of the
// largest, crop them to the pixels left, and pad them. A pixel left out of
// one map (or without a value in it) is left out of all. The project
// keeps the original files and the prepared copies; the dataset is of the
// copies.
import { useState } from 'react';
import { significant } from '../components/format';
import type { Region } from '../components/MapView';
import { Button } from '../components/ui/basics';
import { Check, NumberField } from '../components/ui/inputs';
import { Panel } from '../components/ui/panel';
import { ValueRange } from '../components/ui/sliders';
import { clampedCrop, cropOf, SIGMA_ITERATIONS, value, type ErrorKind, type PrepFiles } from './Edits';

// The edits of the values of a map, as typed (blank: none)
export interface ValueEdits {
  below: string;
  above: string;
  sigma: string;
}

// The edits as typed (blank: none), those of the values of each map by
// its key; the crops are inclusive, of pixels from 0
export interface MapEdits {
  x: [string, string];
  y: [string, string];
  values: Record<string, ValueEdits>;
  regions: string;
  pixels: string;
  ratio: string;
  minify: boolean;
  pad: string;
}

export const NO_MAP_EDITS: MapEdits = {
  x: ['', ''],
  y: ['', ''],
  values: {},
  regions: '',
  pixels: '',
  ratio: '',
  minify: false,
  pad: '',
};

const NO_VALUE_EDITS: ValueEdits = { below: '', above: '', sigma: '' };

// A map to edit: its key, its title, the unit of its values, and their
// values (once read)
export interface EditedMap {
  key: string;
  title: string;
  unit: string | null;
  values?: Float32Array;
}

const typed = (edits: ValueEdits) => [edits.below, edits.above, edits.sigma];

// Whether edits are typed (or checked), of the maps given
export function hasMapEdits(edits: MapEdits, keys: string[]): boolean {
  const texts = [
    ...[...edits.x, ...edits.y, edits.regions, edits.pixels, edits.ratio, edits.pad],
    ...keys.flatMap((key) => typed(edits.values[key] ?? NO_VALUE_EDITS)),
  ];
  return texts.some((text) => text.trim() !== '') || edits.minify;
}

// The crop of the edits to draw (pixels from 0, inclusive; null: none),
// of maps of a shape (x, y)
export function mapCropOf(edits: MapEdits, [nx, ny]: [number, number]): Region | null {
  const x = cropOf(edits.x, nx, 0);
  const y = cropOf(edits.y, ny, 0);
  return x || y ? [...(x ?? [0, nx - 1]), ...(y ?? [0, ny - 1])] : null;
}

// The edits of a crop to a region (x0, x1, y0, y1, inclusive)
export function withMapRegion(edits: MapEdits, [x0, x1, y0, y1]: Region): MapEdits {
  return { ...edits, x: [String(x0), String(x1)], y: [String(y0), String(y1)] };
}

// The range of the values kept of a map (null: open at that end), as typed
function keptOf(edits: ValueEdits): [number | null, number | null] {
  const number = (text: string) => {
    const at = value(text);
    return at === null || Number.isNaN(at) ? null : at;
  };
  return [number(edits.below), number(edits.above)];
}

// The pixels that the edits leave out of every map (1), as far as they
// are seen before prep: those without a value in a map, or out of the
// range of the values kept of a map (null: no edits, or a map not read)
export function leftOut(edits: MapEdits, maps: EditedMap[]): Uint8Array | null {
  if (!hasMapEdits(edits, maps.map(({ key }) => key)) || maps.some(({ values }) => values === undefined)) return null;
  const out = new Uint8Array(maps[0].values!.length);
  for (const { key, values } of maps) {
    const [low, high] = keptOf(edits.values[key] ?? NO_VALUE_EDITS);
    values!.forEach((at, i) => {
      if (!Number.isFinite(at) || (low !== null && at < low) || (high !== null && at > high)) out[i] = 1;
    });
  }
  return out;
}

// The files of a map as prep takes them, and what its errors hold
export interface PrepMap {
  key: string;
  files: PrepFiles;
  errorKind: ErrorKind;
}

// The options of prep_pixel_moments of the edits, or why they cannot be:
// the files of the maps (a list of each role), the crop as prep takes it
// (pixels from the first to the last + 1), and the edits of the values of
// each map
export function mapPrepOptions(
  edits: MapEdits,
  maps: PrepMap[],
  [nx, ny]: [number, number],
  project: string,
): Record<string, unknown> | string {
  const values = maps.map(({ key }) => edits.values[key] ?? NO_VALUE_EDITS);
  const texts = [...edits.x, ...edits.y, edits.regions, edits.pixels, edits.ratio, edits.pad, ...values.flatMap(typed)];
  if (texts.some((text) => Number.isNaN(value(text)))) return 'An edit is not a number';
  const region = mapCropOf(edits, [nx, ny]);
  if (region && !(0 <= region[0] && region[0] <= region[1] && region[1] < nx && 0 <= region[2] && region[2] <= region[3] && region[3] < ny)) {
    return 'The crop is out of the maps';
  }
  // (prep reads the errors of all the maps alike)
  const kinds = new Set(maps.filter(({ files }) => files.file_e !== null).map(({ errorKind }) => errorKind));
  if (kinds.size > 1) return 'The errors of the maps are of different kinds (σ, σ², 1/σ²); edits need one';
  const integer = (text: string) => {
    const number = value(text);
    return number === null ? null : Math.round(number);
  };
  // (a list of a value of each map, or none if none has one)
  const each = (of: (edits: ValueEdits) => number | null) => {
    const list = values.map(of);
    return list.some((at) => at !== null) ? list : null;
  };
  const masks = maps.map(({ files }) => files.file_m);
  return {
    file_d: maps.map(({ files }) => files.file_d),
    file_e: maps.map(({ files }) => files.file_e),
    error_kind: [...kinds][0] ?? 'sigma',
    file_m: masks.some((mask) => mask !== null) ? masks : null,
    roi_spat: region && [region[0], region[1] + 1, region[2], region[3] + 1],
    clip_min: each((edits) => value(edits.below)),
    clip_max: each((edits) => value(edits.above)),
    ccl_lcount: integer(edits.regions),
    ccl_pcount: integer(edits.pixels),
    ccl_lratio: value(edits.ratio),
    sclip_sigma: each((edits) => value(edits.sigma)),
    sclip_iters: SIGMA_ITERATIONS,
    minify: edits.minify,
    nanpad: integer(edits.pad),
    dtype: 'float32',
    output_dir: project === '.' ? 'data' : `${project}/data`,
  };
}

// The range of the finite values of a map (none: it has none)
function rangeOf(values: Float32Array): [number, number] | null {
  let [low, high] = [Infinity, -Infinity];
  for (const at of values) {
    if (at < low) low = at;
    if (at > high) high = at;
  }
  return low < high ? [low, high] : null;
}

// The edits, as a form of aligned rows (a label, its input, and its
// unit): the crop, and folded the values left out, a group for each map
export function MapEditsCard({
  maps,
  shape,
  edits,
  onChange,
  onCropToSelection,
}: {
  maps: EditedMap[];
  // The shape of the maps (x, y)
  shape: [number, number];
  edits: MapEdits;
  onChange: (edits: MapEdits) => void;
  // How the crop becomes the selection on the map (none: nothing selected)
  onCropToSelection?: () => void;
}) {
  const [nx, ny] = shape;
  const set = (changed: Partial<MapEdits>) => onChange({ ...edits, ...changed });
  const setValues = (key: string, changed: Partial<ValueEdits>) =>
    set({ values: { ...edits.values, [key]: { ...(edits.values[key] ?? NO_VALUE_EDITS), ...changed } } });
  const keys = maps.map(({ key }) => key);
  // (a crop is clamped to the maps as its fields are left, the first
  // before the last)
  const crop = (key: 'x' | 'y', last: number) => (
    <>
      <span className="edit-label">{key}</span>
      <span className="edit-range" onBlur={() => set({ [key]: clampedCrop(edits[key], last + 1, 0) })}>
        <NumberField
          value={edits[key][0]}
          min={0}
          max={last}
          placeholder="0"
          label={`The first of ${key}`}
          onChange={(text) => set({ [key]: [text, edits[key][1]] })}
        />
        <span>–</span>
        <NumberField
          value={edits[key][1]}
          min={0}
          max={last}
          placeholder={String(last)}
          label={`The last of ${key}`}
          onChange={(text) => set({ [key]: [edits[key][0], text] })}
        />
      </span>
      <span className="edit-unit">pixels</span>
    </>
  );
  const number = (key: 'regions' | 'pixels' | 'ratio' | 'pad', label: string, unit: string, title: string) => (
    <>
      <span className="edit-label" title={title}>
        {label}
      </span>
      <NumberField value={edits[key]} label={label} title={title} onChange={(text) => set({ [key]: text })} />
      <span className="edit-unit">{unit}</span>
    </>
  );
  // (the values left out are open at first if any are typed, and then as
  // opened or closed: never closed by an edit)
  const [leaveOut, setLeaveOut] = useState(() =>
    [edits.regions, edits.pixels, edits.ratio, ...keys.flatMap((key) => typed(edits.values[key] ?? NO_VALUE_EDITS))].some(
      (text) => text.trim() !== '',
    ),
  );
  return (
    <Panel
      id="moments-edits"
      title="Edits"
      actions={
        hasMapEdits(edits, keys) && (
          <Button kind="small" title="Clear the edits" onClick={() => onChange(NO_MAP_EDITS)}>
            clear
          </Button>
        )
      }
    >
      <div className="edits">
        <div className="edit-group edit-group-action">
          crop
          <Button
            kind="small"
            disabled={onCropToSelection === undefined}
            title={onCropToSelection === undefined ? 'Select a region of the map to crop to' : 'Crop to the region selected on the map'}
            onClick={() => onCropToSelection?.()}
          >
            Crop to selection
          </Button>
        </div>
        {crop('x', nx - 1)}
        {crop('y', ny - 1)}
        <details className="edit-more" open={leaveOut} onToggle={(event) => setLeaveOut(event.currentTarget.open)}>
          <summary className="edit-group">leave out values</summary>
          <div className="edits">
            {maps.map(({ key, title, unit, values }) => {
              const own = edits.values[key] ?? NO_VALUE_EDITS;
              const range = values && rangeOf(values);
              const step = range ? Number(significant((range[1] - range[0]) / 200, 1)) : 1;
              return (
                <div className="edits edit-map" key={key}>
                  <span className="edit-sub">{title}</span>
                  <span className="edit-label" title="Leave out the values out of this range (blank: no limit), shown blank on the map">
                    keep values
                  </span>
                  <span className="edit-range">
                    <NumberField value={own.below} placeholder="−∞" label={`Keep the values of ${title} from`} onChange={(below) => setValues(key, { below })} />
                    <span>–</span>
                    <NumberField value={own.above} placeholder="∞" label={`Keep the values of ${title} up to`} onChange={(above) => setValues(key, { above })} />
                  </span>
                  <span className="edit-unit">{unit ?? ''}</span>
                  {range && (
                    <div className="edit-wide">
                      <ValueRange
                        min={range[0]}
                        max={range[1]}
                        step={step}
                        value={keptOf(own)}
                        format={(at) => significant(at, 3)}
                        label={`The values of ${title} kept`}
                        onChange={([low, high]) =>
                          setValues(key, { below: low === null ? '' : significant(low, 4), above: high === null ? '' : significant(high, 4) })
                        }
                      />
                    </div>
                  )}
                  <span className="edit-label" title={`Sigma clipping of the values of ${title}, ${SIGMA_ITERATIONS} times, on both sides`}>
                    outliers beyond
                  </span>
                  <NumberField value={own.sigma} label={`Outliers of ${title} beyond`} onChange={(sigma) => setValues(key, { sigma })} />
                  <span className="edit-unit">σ of them all</span>
                </div>
              );
            })}
            <span className="edit-sub" title="Of the pixels left, those connected (connected-component labelling)">
              of the pixels left, keep the regions
            </span>
            {number('regions', 'largest', 'regions', 'The number of regions kept, the largest first')}
            {number('pixels', 'of at least', 'pixels', 'The pixels of a region kept, at least')}
            {number('ratio', 'and at least', '× the largest', 'The size of a region kept, at least, as a fraction of the largest')}
            <div className="edit-check">
              <Check checked={edits.minify} label="crop to the pixels left" title="Crop the files to the pixels left" onChange={(minify) => set({ minify })} />
            </div>
            {number('pad', 'pad by', 'pixels', 'Pad the files by this many empty pixels on each side')}
          </div>
        </details>
      </div>
      {hasMapEdits(edits, keys) && (
        <p className="note edits-note">
          A pixel left out of one map is left out of all. Adding writes edited copies of the maps into the project; the originals stay.
        </p>
      )}
    </Panel>
  );
}
