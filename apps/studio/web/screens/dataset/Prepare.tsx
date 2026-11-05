// The edits that prep makes to the files of a new dataset (see
// gbkfit.tasks.prep), in the dock: crop them (typed, or to the selection
// on the map and the spectrum), make the errors of a cube from its noise,
// leave out values (out of a range, or outliers) and then all but the
// largest regions of the pixels left, crop to the pixels left, and pad.
// The originals stay in the project; the dataset is of the prepared
// copies. Of moment maps, a pixel left out of one map is left out of all.
import { useMaps, useSpectrum } from '../../api/queries';
import { significant } from '../../components/format';
import { Button } from '../../components/ui/basics';
import { Check, NumberField } from '../../components/ui/inputs';
import { ValueRange } from '../../components/ui/sliders';
import {
  clampRange,
  kept,
  keptValues,
  NO_VALUE_EDITS,
  presentSlots,
  rangeOf,
  SIGMA_ITERATIONS,
  type Edits,
  type Range,
  type Resolved,
  type ValueEdits,
} from './model';

export function PrepareForm({
  resolved,
  onEdits,
  onCropToSelection,
}: {
  resolved: Resolved;
  onEdits: (edits: Edits) => void;
  // Crops to the selection on the map and the spectrum (none: nothing
  // selected)
  onCropToSelection?: () => void;
}) {
  const { kind, draft, arrays } = resolved;
  const { edits } = draft;
  const slots = presentSlots(kind, draft);
  const ranges = useMapRanges(resolved);
  const first = slots.map((slot) => arrays[slot.key]?.found.data).find((data) => data !== undefined);
  if (first === undefined) return <p className="note">The edits show once the data are read.</p>;
  const [nx, ny, nz] = kept(first);
  const set = (changed: Partial<Edits>) => onEdits({ ...edits, ...changed });
  const setValues = (key: string, changed: Partial<ValueEdits>) =>
    set({ values: { ...edits.values, [key]: { ...(edits.values[key] ?? NO_VALUE_EDITS), ...changed } } });
  const item = draft.items[slots[0].key];
  const noiseErrors = kind === 'cube' && item.files.error === undefined && item.errorValue === null;
  const channels = kind === 'cube' ? rangeOf(edits.channels, nz) : null;
  const cube = kind === 'cube' ? draft.items[slots[0].key].files.data : undefined;
  return (
    <div className="ds-edits">
      <div className="ds-group">
        <span>Crop</span>
        <Button
          kind="small"
          disabled={onCropToSelection === undefined}
          title={onCropToSelection ? 'Crop to the region (and the channels) selected' : 'Select a region on the map (or channels on the spectrum) first'}
          onClick={() => onCropToSelection?.()}
        >
          Crop to selection
        </Button>
      </div>
      <CropRow label="x" range={edits.x} size={nx} unit="px" onChange={(x) => set({ x })} />
      <CropRow label="y" range={edits.y} size={ny} unit="px" onChange={(y) => set({ y })} />
      {kind === 'cube' && (
        <>
          <CropRow label="Channels" range={edits.channels} size={nz} unit="" onChange={(range) => set({ channels: range })} />
          {channels && cube && <ChannelSpan path={cube.file} hdu={first.index} channels={channels} />}
        </>
      )}
      {noiseErrors && (
        <>
          <div className="ds-group">
            <span>Errors</span>
          </div>
          <div className="ds-edit-wide">
            <Check
              checked={edits.noise}
              label="From the noise of each channel"
              title="The robust noise of the negative values of each channel, measured before any values are left out"
              onChange={(noise) => set({ noise })}
            />
          </div>
        </>
      )}
      <div className="ds-group">
        <span>Leave out</span>
      </div>
      {slots.map((slot) => {
        const own = edits.values[slot.key] ?? NO_VALUE_EDITS;
        const data = arrays[slot.key]?.found.data;
        const unit = data?.unit ?? '';
        const noise = kind === 'cube' && data?.rms ? data.rms : null;
        const range = ranges[slot.key] ?? null;
        return (
          <div key={slot.key} className="ds-edit-block">
            {kind === 'moments' && <div className="ds-edit-sub">{slot.title}</div>}
            <span className="ds-label" title="Values out of this range are left out (blank: no limit); they show blank on the map">
              Values
            </span>
            <span className="ds-range">
              <NumberField value={own.below} placeholder="−∞" label={`The least value kept of ${slot.title}`} onChange={(below) => setValues(slot.key, { below })} />
              <span>to</span>
              <NumberField value={own.above} placeholder="∞" label={`The greatest value kept of ${slot.title}`} onChange={(above) => setValues(slot.key, { above })} />
            </span>
            <span className="ds-unit">{unit}</span>
            {noise !== null ? (
              <div className="ds-edit-wide">
                <ValueRange
                  min={-3}
                  max={15}
                  step={0.1}
                  value={inUnits(keptValues(edits, slot.key), noise)}
                  mark={0}
                  format={(at) => `${significant(at, 2)}σ`}
                  label="The values kept, in units of the noise of the header"
                  onChange={([low, high]) => setValues(slot.key, { below: fromUnits(low, noise), above: fromUnits(high, noise) })}
                />
              </div>
            ) : (
              range && (
                <div className="ds-edit-wide">
                  <ValueRange
                    min={range[0]}
                    max={range[1]}
                    step={Number(significant((range[1] - range[0]) / 200, 1)) || 1}
                    value={keptValues(edits, slot.key)}
                    format={(at) => significant(at, 3)}
                    label={`The values kept of ${slot.title}`}
                    onChange={([low, high]) =>
                      setValues(slot.key, { below: low === null ? '' : significant(low, 4), above: high === null ? '' : significant(high, 4) })
                    }
                  />
                </div>
              )
            )}
            <span className="ds-label" title={`Sigma clipping: the values beyond this many standard deviations of them all, ${SIGMA_ITERATIONS} times`}>
              Outliers
            </span>
            <NumberField value={own.sigma} placeholder="none" label={`Outliers of ${slot.title} beyond`} onChange={(sigma) => setValues(slot.key, { sigma })} />
            <span className="ds-unit">σ</span>
            {kind === 'cube' && own.sigma.trim() !== '' && (
              <p className="ds-warn ds-edit-wide">
                Sigma clipping leaves out the values far from the mean of them all: in a cube of a line, its emission too. It is for images with bad
                pixels.
              </p>
            )}
          </div>
        );
      })}
      <span className="ds-label" title="Of the pixels left, those connected form regions (connected-component labelling)">
        Regions
      </span>
      <span className="ds-range">
        <NumberField value={edits.regions} placeholder="all" label="The number of regions kept, the largest first" onChange={(regions) => set({ regions })} />
        <span>largest</span>
      </span>
      <span className="ds-unit" />
      <span className="ds-label ds-indent">of at least</span>
      <NumberField value={edits.pixels} placeholder="any" label="The least number of pixels of a region kept" onChange={(pixels) => set({ pixels })} />
      <span className="ds-unit">px</span>
      <span className="ds-label ds-indent">and at least</span>
      <NumberField value={edits.ratio} placeholder="any" label="The least size of a region kept, as a fraction of the largest" onChange={(ratio) => set({ ratio })} />
      <span className="ds-unit">× largest</span>
      <div className="ds-group">
        <span>Finish</span>
      </div>
      <div className="ds-edit-wide">
        <Check checked={edits.minify} label="Crop to the pixels left" title="Crop the files to the box of the pixels left" onChange={(minify) => set({ minify })} />
      </div>
      <span className="ds-label">Pad by</span>
      <NumberField value={edits.pad} placeholder="0" label="Pad the files by this many empty pixels on each side" onChange={(pad) => set({ pad })} />
      <span className="ds-unit">px</span>
    </div>
  );
}

// A range of pixels or channels: its first and last (from 0, inclusive),
// made whole numbers within the axis as the fields are left
function CropRow({ label, range, size, unit, onChange }: { label: string; range: Range; size: number; unit: string; onChange: (range: Range) => void }) {
  return (
    <>
      <span className="ds-label">{label}</span>
      <span className="ds-range" onBlur={() => onChange(clampRange(range, size))}>
        <NumberField value={range[0]} min={0} max={size - 1} placeholder="0" label={`The first of ${label}`} onChange={(first) => onChange([first, range[1]])} />
        <span>to</span>
        <NumberField value={range[1]} min={0} max={size - 1} placeholder={String(size - 1)} label={`The last of ${label}`} onChange={(last) => onChange([range[0], last])} />
      </span>
      <span className="ds-unit">{unit}</span>
    </>
  );
}

function inUnits([low, high]: [number | null, number | null], unit: number): [number | null, number | null] {
  return [low === null ? null : low / unit, high === null ? null : high / unit];
}

function fromUnits(at: number | null, unit: number): string {
  return at === null ? '' : significant(at * unit, 3);
}

// The range of the finite values of a map (null: it has none)
function valueRange(values: Float32Array | undefined): [number, number] | null {
  if (!values) return null;
  let [low, high] = [Infinity, -Infinity];
  for (const at of values) {
    if (at < low) low = at;
    if (at > high) high = at;
  }
  return low < high ? [low, high] : null;
}

// What a range of channels spans along the spectral axis (e.g. in km/s),
// from the total spectrum of the cube (that of its view)
function ChannelSpan({ path, hdu, channels }: { path: string; hdu: number; channels: [number, number] }) {
  const spectrum = useSpectrum(path, hdu, null, null).data;
  const [a, b] = channels.map((channel) => spectrum?.spectral?.[channel]);
  if (a === undefined || b === undefined || a === null || b === null) return null;
  return (
    <p className="ds-edit-note">
      {significant(Math.min(a, b), 4)} to {significant(Math.max(a, b), 4)} {spectrum?.unit ?? ''}
    </p>
  );
}

// The range of the values of each map of a dataset of maps (by key), for
// the sliders of the values kept
function useMapRanges({ kind, draft, arrays }: Resolved): Record<string, [number, number] | null> {
  const keys = kind === 'cube' ? [] : presentSlots(kind, draft).filter((slot) => arrays[slot.key]?.found.data).map((slot) => slot.key);
  const values = useMaps(keys.map((key) => ({ path: draft.items[key].files.data!.file, hdu: arrays[key].found.data!.index })));
  return Object.fromEntries(keys.map((key, i) => [key, valueRange(values[i])]));
}
