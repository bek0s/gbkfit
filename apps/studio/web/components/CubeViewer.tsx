// A cube, to look at: its data, or the cube of their errors or of their
// mask (its layers), one map at a time, with its colour bar, as large as
// the space left allows: a channel, or one of the quick moment maps of the
// data (a look, measured here; not part of the dataset, as its corner
// says), chosen on a rail at its left. A selection (a region of the map, a band of channels of the
// spectrum) is made to look at: the map and the spectrum zoom to it (or
// with the wheel) and back, and the total spectrum is of it; it is never
// kept. The map has two tools (see MapTools): to select (a drag selects;
// Esc clears) and to pan (a drag pans the zoomed map); a click picks a
// pixel with either. A crop (an edit of the dataset, set from the selection; see
// Edits) is outlined. Under the map, the channel (a slider), or the mask
// of the moments (grown by prep's tools from seeds above a number of times
// the noise), in a row of one height, so that the map keeps its size; and
// the spectrum of the pixel picked, or else the total of the selection (or
// of the cube), with the channel marked (a click on the spectrum picks a
// channel; a drag selects a band). The values out of the range kept are
// blank on the channel map of the data.
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import type { FrameArray } from '../api/frames';
import { useChannel, useMoments, useSpectrum } from '../api/queries';
import { ColourBar } from './ColourBar';
import { significant } from './format';
import { MapView, type Region } from './MapView';
import { SpectrumPlot, valueRange } from './SpectrumPlot';
import { ZoomSelectionIcon, ZoomWholeIcon } from './icons';
import { MapTools, type Tool } from './MapTools';
import { Chip, IconButton, NavItem } from './ui/basics';
import { Check, NumberField, Segments } from './ui/inputs';
import { StepSlider } from './ui/sliders';

// A cube of a dataset: its data (the first), their errors or their mask;
// its array in its file, and the unit of its values
export interface Layer {
  key: 'data' | 'error' | 'mask';
  title: string;
  path: string;
  hdu: number;
  unit: string | null;
}

type Moment = 'flux' | 'velocity' | 'dispersion' | 'skewness' | 'kurtosis';
type View = 'channel' | Moment;

// The moment maps (by order), what they are, and how their colours run:
// sequential, diverging about their median (velocities), or about 0
const MOMENTS: { view: Moment; title: string; text: string; colours: 'sequential' | 'median' | 'zero' }[] = [
  {
    view: 'flux',
    title: 'Intensity',
    text: 'Moment 0: the values summed over the channels, times their width. Of values per beam (e.g. Jy/beam), a surface brightness integrated over velocity; of values per pixel, a flux',
    colours: 'sequential',
  },
  { view: 'velocity', title: 'Velocity', text: 'Moment 1: the mean velocity of each spectrum', colours: 'median' },
  { view: 'dispersion', title: 'Dispersion', text: 'Moment 2: the dispersion of the velocities of each spectrum', colours: 'sequential' },
  { view: 'skewness', title: 'Skewness', text: 'Moment 3 over the dispersion cubed: the asymmetry of each spectrum', colours: 'zero' },
  { view: 'kurtosis', title: 'Kurtosis', text: 'Moment 4 over the dispersion to the 4th, less 3 (0 for a Gaussian): the tails of each spectrum', colours: 'zero' },
];

// The signal-to-noise ratio of the mask of the moments, at first (see the
// server's views)
const SNR = 5;

// The width of the colour bar at the side of the map, with its gap
const BAR = 66;

// How often (ms) the spectrum of the pixel under the pointer is updated
const HOVER = 40;

// The fewest pixels a side of the window of a zoomed map has
const LEAST = 8;

export function CubeViewer({
  layers,
  shape,
  selection,
  band,
  crop = null,
  cropChannels = null,
  kept = null,
  onSelection,
  onBand,
  actions,
}: {
  layers: Layer[];
  // Its shape in the order of the model (x, y, then its channels)
  shape: number[];
  // The selection (a region of pixels, a band of channels, from 0 and
  // inclusive; null: none), and how it changes
  selection: Region | null;
  band: [number, number] | null;
  onSelection: (selection: Region | null) => void;
  onBand: (band: [number, number] | null) => void;
  // The crops of the dataset, outlined (null: none), and the range of the
  // values kept
  crop?: Region | null;
  cropChannels?: [number, number] | null;
  kept?: [number | null, number | null] | null;
  // What the head of the viewer has at its right (e.g. a toggle)
  actions?: ReactNode;
}) {
  const [nx, ny, nchannels] = shape;
  const [layerKey, setLayerKey] = useState<Layer['key']>('data');
  const [view, setView] = useState<View>('channel');
  const [tool, setTool] = useState<Tool>('select');
  const [channel, setChannel] = useState(Math.floor(nchannels / 2));
  // The pixel picked (none: the total spectrum is shown)
  const [pixel, setPixel] = useState<[number, number] | null>(null);
  const [snr, setSnr] = useState(SNR);
  // The window of the map shown (null: all of it)
  const [zoom, setZoom] = useState<Region | null>(null);
  // The window of the spectrum shown (first and last channel; null: all)
  const [spectrumZoom, setSpectrumZoom] = useState<[number, number] | null>(null);
  // Whether the range of the values of the spectrum is its own (autoscale),
  // or else the ends typed (from those it had as autoscale was turned off)
  const [autoscale, setAutoscale] = useState(true);
  const [limits, setLimits] = useState<[string, string]>(['', '']);
  // The pixel under the pointer, whose spectrum shows while it hovers
  // (updated at most every HOVER ms, the latest kept)
  const [hover, setHover] = useState<[number, number] | null>(null);
  const hoverNext = useRef<[number, number] | null>(null);
  const hoverTimer = useRef<number | null>(null);
  const hoverAt = (at: [number, number] | null) => {
    hoverNext.current = at;
    if (at === null) {
      if (hoverTimer.current !== null) window.clearTimeout(hoverTimer.current);
      hoverTimer.current = null;
      setHover(null);
    } else if (hoverTimer.current === null) {
      hoverTimer.current = window.setTimeout(() => {
        hoverTimer.current = null;
        const next = hoverNext.current;
        setHover((now) => (now !== null && next !== null && now[0] === next[0] && now[1] === next[1] ? now : next));
      }, HOVER);
    }
  };
  // (the spectrum of the pixel hovered, or else of that picked, or else the
  // total of the selection or of the cube)
  const shownPixel = hover ?? pixel;
  // (a layer gone, the data; the moments are of the data alone)
  const layer = layers.find((l) => l.key === layerKey) ?? layers[0];
  const data = layers[0];
  const shown: View = layer.key === 'data' ? view : 'channel';
  const selecting = tool === 'select';
  const plane = useChannel(layer.path, layer.hdu, channel);
  const moments = useMoments(data.path, data.hdu, snr, shown !== 'channel');
  const spectrum = useSpectrum(layer.path, layer.hdu, shownPixel, selection);
  // The text of a channel (its number and its coordinate) and of a band of
  // channels (their coordinates, or numbers if unknown)
  const spectralOf = (c: number) => spectrum.data?.spectral?.[c];
  const spectralUnit = spectrum.data?.unit ?? '';
  const channelText = (c: number) => {
    const at = spectralOf(c);
    return at === undefined ? `channel ${c + 1}` : `${c + 1} · ${significant(at, 4)} ${spectralUnit}`;
  };
  const bandText = ([c0, c1]: [number, number]) => {
    const [a, b] = [spectralOf(c0), spectralOf(c1)];
    return a === undefined || b === undefined
      ? `channels ${c0 + 1}–${c1 + 1}`
      : `${significant(Math.min(a, b), 4)}–${significant(Math.max(a, b), 4)} ${spectralUnit}`;
  };
  // The range of the values of the spectrum shown (autoscale), as the fields
  // show it, and the one typed (null: not a range)
  const autoRange = spectrum.data
    ? (valueRange(spectrum.data.values, spectrumZoom).map((end) => significant(end, 3)) as [string, string])
    : (['', ''] as [string, string]);
  const typedRange = ((): [number, number] | null => {
    const [low, high] = limits.map((text) => (text.trim() === '' ? NaN : Number(text)));
    return Number.isFinite(low) && Number.isFinite(high) && high > low ? [low, high] : null;
  })();
  // (the pixels shown)
  const shownPixels = zoom ?? ([0, nx - 1, 0, ny - 1] as Region);
  const fitted = useFitted(shownPixels[1] - shownPixels[0] + 1, shownPixels[3] - shownPixels[2] + 1);
  const colours = MOMENTS.find((m) => m.view === shown)?.colours ?? 'sequential';
  const map: FrameArray | undefined =
    shown === 'channel' ? plane.data?.plane : moments.data?.find((array) => array.info.name === shown);
  // (the mask, 0 to 1; the other cubes over the range of their values)
  const range: [number, number] | undefined =
    shown !== 'channel' ? map && robustRange(map.values, colours) : layer.key === 'mask' ? [0, 1] : plane.data?.range;
  const error = shown === 'channel' ? plane.error : moments.error;
  // (Esc clears the selection, all its parts, as the select tool is in use)
  useEffect(() => {
    if (!selecting) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onSelection(null);
        onBand(null);
        setPixel(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selecting, onSelection, onBand]);
  // Zoom about a pixel by a factor, keeping it where it is on the screen;
  // out to the whole map, the window is none
  const zoomAt = (x: number, y: number, factor: number) => {
    const [x0, x1, y0, y1] = shownPixels;
    const w = Math.min(nx, Math.max(LEAST, Math.round((x1 - x0 + 1) / factor)));
    const h = Math.min(ny, Math.max(LEAST, Math.round((y1 - y0 + 1) / factor)));
    if (w >= nx && h >= ny) return setZoom(null);
    const left = Math.min(nx - w, Math.max(0, Math.round(x - ((x - x0) * w) / (x1 - x0 + 1))));
    const bottom = Math.min(ny - h, Math.max(0, Math.round(y - ((y - y0) * h) / (y1 - y0 + 1))));
    setZoom([left, left + w - 1, bottom, bottom + h - 1]);
  };
  // Zoom the spectrum about a channel by a factor, keeping it where it is;
  // out to all the channels, the window is none
  const zoomSpectrum = (at: number, factor: number) => {
    const [c0, c1] = spectrumZoom ?? [0, nchannels - 1];
    const n = Math.min(nchannels, Math.max(LEAST, Math.round((c1 - c0 + 1) / factor)));
    if (n >= nchannels) return setSpectrumZoom(null);
    const first = Math.min(nchannels - n, Math.max(0, Math.round(at - ((at - c0) * n) / (c1 - c0 + 1))));
    setSpectrumZoom([first, first + n - 1]);
  };
  // Pan the window by pixels, within the map
  const panBy = (dx: number, dy: number) => {
    if (zoom === null) return;
    const [x0, x1, y0, y1] = zoom;
    const mx = Math.min(nx - 1 - x1, Math.max(-x0, dx));
    const my = Math.min(ny - 1 - y1, Math.max(-y0, dy));
    setZoom([x0 + mx, x1 + mx, y0 + my, y1 + my]);
  };
  return (
    <div className="card viewer">
      <div className="card-head viewer-head">
        {layers.length > 1 ? (
          <Segments items={layers.map((l) => ({ value: l.key, label: l.title }))} value={layer.key} onChange={setLayerKey} label="The cube shown" />
        ) : (
          <h2>{data.title}</h2>
        )}
        {/* (the selection, a chip of each of its parts, each cleared by its
            own button; Esc clears them all): a place (a region, or a pixel:
            one at a time) and a band of channels */}
        <div className="selection-chips" aria-label="The selection">
          {selection !== null && (
            <Chip title="The region selected on the map" removeLabel="Clear the region" onRemove={() => onSelection(null)}>
              x {selection[0]}–{selection[1]} · y {selection[2]}–{selection[3]}
            </Chip>
          )}
          {pixel !== null && (
            <Chip title="The pixel picked on the map" removeLabel="Clear the pixel" onRemove={() => setPixel(null)}>
              Pixel {pixel[0]}, {pixel[1]}
            </Chip>
          )}
          {band !== null && (
            <Chip title="The channels selected on the spectrum" removeLabel="Clear the channels" onRemove={() => onBand(null)}>
              {bandText(band)}
            </Chip>
          )}
        </div>
        {actions}
      </div>
      <div className="viewer-body">
        <nav className="rail" aria-label="The views of the cube">
          <div className="rail-label">view</div>
          <NavItem label="Channel" active={shown === 'channel'} title="One channel of the cube at a time" onClick={() => setView('channel')} />
          <div className="rail-label" title="Quick moment maps of the data: a look, measured here; the dataset stays a cube">
            quick look
          </div>
          {MOMENTS.map(({ view: v, title, text }, order) => (
            <NavItem
              key={v}
              label={title}
              left={<span className="order">≈{order}</span>}
              active={shown === v}
              disabled={layer.key !== 'data'}
              title={layer.key === 'data' ? text : 'The moments are of the data'}
              onClick={() => setView(v)}
            />
          ))}
        </nav>
        <div className="viewer-main">
          <div className="viewer-stage" ref={fitted.stage}>
            {error && <p className="error">{error.message}</p>}
            {shown !== 'channel' && moments.isPending && <p className="note">Measuring the moments of the cube…</p>}
            {map && range && fitted.size && (
              <div className="viewer-map">
                <div className="viewer-canvas" style={{ width: fitted.size[0], height: fitted.size[1] }}>
                  <MapView
                    values={map.values}
                    shape={map.shape}
                    range={range}
                    diverging={colours !== 'sequential'}
                    kept={shown === 'channel' && layer.key === 'data' ? kept : null}
                    window={zoom}
                    drag={selecting ? 'region' : 'pan'}
                    mark={pixel ?? undefined}
                    region={selection}
                    outline={crop}
                    onPick={(x, y) => {
                      // (one place is selected at a time: a pixel, or a region)
                      setPixel([x, y]);
                      onSelection(null);
                    }}
                    onRegion={
                      selecting
                        ? (region) => {
                            setPixel(null);
                            onSelection(region);
                          }
                        : undefined
                    }
                    onHover={hoverAt}
                    onZoom={zoomAt}
                    onPan={panBy}
                    label={shown === 'channel' ? `A channel of the ${layer.title.toLowerCase()}` : `The ${shown} map of the data`}
                  />
                  {shown !== 'channel' && (
                    <div className="map-source derived" title="A quick moment map of the data, measured here from the cube; the dataset stays a cube">
                      ≈ quick look · not part of the dataset
                    </div>
                  )}
                  {selecting && selection === null && pixel === null && <div className="map-hint">Drag on the map to select a region</div>}
                  <MapTools
                    tool={tool}
                    onTool={setTool}
                    selectTitle="Select: a drag selects a region of the map (Esc clears the selection)"
                    onZoomToSelection={selection ? () => setZoom(selection) : undefined}
                    onWhole={zoom ? () => setZoom(null) : undefined}
                  />
                </div>
                <ColourBar range={range} diverging={colours !== 'sequential'} unit={mapUnit(shown, layer.unit)} />
              </div>
            )}
          </div>
          {shown === 'channel' ? (
            <div className="viewer-controls">
              <StepSlider
                min={spectrumZoom?.[0] ?? 0}
                max={spectrumZoom?.[1] ?? nchannels - 1}
                value={channel}
                format={channelText}
                always
                label="Channel"
                onChange={setChannel}
              />
            </div>
          ) : (
            // (the value at the side, not on the handle, which would cover
            // the colour bar at the end)
            <div
              className="viewer-controls viewer-controls-row"
              title="The moments are of the voxels connected to those above this many times the noise, down to half of it, in the larger regions (0: all of them)"
            >
              <span className="control-text">{snr === 0 ? 'moments of all the voxels' : `moments of the signal from ${snr}σ`}</span>
              <StepSlider
                min={0}
                max={6}
                step={0.5}
                value={snr}
                format={(x) => (x === 0 ? 'All' : `${x}σ`)}
                label="The signal-to-noise ratio of the mask of the moments"
                onChange={setSnr}
              />
            </div>
          )}
          <div className="spectrum-head">
            {/* (what the spectrum is of, and the unit of its values: of a pixel
                hovered or picked, or else the sum of the pixels of the
                selection or of the cube; a picked pixel can go) */}
            <span
              className="spectrum-what"
              data-hover={hover !== null || undefined}
              title={
                hover !== null
                  ? 'The pixel under the pointer (a click keeps it)'
                  : pixel !== null
                    ? 'The pixel picked'
                    : 'The sum of the pixels in each channel (the pointer over the map shows a pixel)'
              }
            >
              {layer.key !== 'data' && `${layer.title} of `}
              {shownPixel !== null ? `Pixel ${shownPixel[0]}, ${shownPixel[1]}` : `Σ ${selection === null ? 'cube' : 'selection'}`}
              {layer.unit && <span className="spectrum-unit"> · {layer.unit}</span>}
            </span>
            <span className="spectrum-actions">
              <span className="spectrum-scale">
                <Check
                  checked={autoscale}
                  label="Autoscale"
                  title="The range of the values is that of the spectrum shown; off, it is the one typed"
                  onChange={(on) => {
                    // (turned off: the ends typed start from those shown)
                    if (!on && spectrum.data) {
                      const [low, high] = valueRange(spectrum.data.values, spectrumZoom);
                      setLimits([significant(low, 3), significant(high, 3)]);
                    }
                    setAutoscale(on);
                  }}
                />
                <NumberField value={autoscale ? autoRange[0] : limits[0]} disabled={autoscale} label="The bottom of the values shown" onChange={(low) => setLimits([low, limits[1]])} />
                <span>–</span>
                <NumberField value={autoscale ? autoRange[1] : limits[1]} disabled={autoscale} label="The top of the values shown" onChange={(high) => setLimits([limits[0], high])} />
              </span>
              <IconButton label="Zoom to the selected channels (the wheel zooms about the pointer)" disabled={band === null} onClick={() => band && setSpectrumZoom(band)}>
                <ZoomSelectionIcon />
              </IconButton>
              <IconButton
                label="The whole spectrum, and its values"
                disabled={spectrumZoom === null && autoscale}
                onClick={() => {
                  setSpectrumZoom(null);
                  setAutoscale(true);
                }}
              >
                <ZoomWholeIcon />
              </IconButton>
            </span>
          </div>
          {spectrum.error && <p className="error">{spectrum.error.message}</p>}
          {spectrum.data && (
            <SpectrumPlot
              values={spectrum.data.values}
              spectral={spectrum.data.spectral}
              unit={spectrum.data.unit}
              channel={channel}
              band={band}
              marked={cropChannels}
              onPick={(picked) => {
                setChannel(picked);
                setView('channel');
              }}
              onBand={onBand}
              window={spectrumZoom}
              limits={autoscale ? null : typedRange}
              onZoom={zoomSpectrum}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// The size of a map of nx by ny pixels that fits the space of its stage,
// beside its colour bar (undefined until the stage is measured)
export function useFitted(nx: number, ny: number) {
  const stage = useRef<HTMLDivElement>(null);
  const [space, setSpace] = useState<[number, number] | null>(null);
  useLayoutEffect(() => {
    const element = stage.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setSpace([entry.contentRect.width, entry.contentRect.height]));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  if (space === null) return { stage, size: undefined };
  const scale = Math.max(0, Math.min((space[0] - BAR) / nx, space[1] / ny));
  return { stage, size: [Math.floor(nx * scale), Math.floor(ny * scale)] as [number, number] };
}

// The unit of a map, from that of the values of the cube
function mapUnit(view: View, unit: string | null): string {
  if (view === 'channel') return unit ?? '';
  if (view === 'flux') return unit ? `${unit} km/s` : 'km/s';
  if (view === 'velocity' || view === 'dispersion') return 'km/s';
  return '';
}

// The range of the colours of a map: from the 1st to the 99th percentile
// of its values; for one diverging, about its median or 0, as wide on
// each side
export function robustRange(values: Float32Array, colours: 'sequential' | 'median' | 'zero'): [number, number] {
  const sorted = Array.from(values.filter(Number.isFinite)).sort((a, b) => a - b);
  if (sorted.length === 0) return [0, 1];
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const [low, high] = [at(0.01), at(0.99)];
  if (colours === 'sequential') return [low, high];
  const middle = colours === 'median' ? at(0.5) : 0;
  const half = Math.max(middle - low, high - middle) || 1;
  return [middle - half, middle + half];
}
