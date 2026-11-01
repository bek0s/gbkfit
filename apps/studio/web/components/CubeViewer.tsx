// A cube, to look at: its data, or the cube of their errors or of their
// mask (its layers), one map at a time, with its colour bar, as large as
// the space left allows: a channel, or one of the quick moment maps of the
// data (a look, measured here; not part of the dataset), chosen on a rail
// at its left. The map zooms (the wheel, or to the crop) and back. With a
// crop to edit, the rail has two tools: to inspect (a click picks a pixel,
// a drag pans the zoomed map) and to crop (a drag crops; Esc clears).
// Under the map, the channel (a slider), or the mask of the moments (grown
// by prep's tools from seeds above a number of times the noise), in a row
// of one height, so that the map keeps its size; and the spectrum of a
// pixel, picked on the map, with the channel marked (a click on the
// spectrum picks a channel; a drag crops the channels). The values out of
// the range kept are blank on the channel map of the data.
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import type { FrameArray } from '../api/frames';
import { useChannel, useMoments, useSpectrum } from '../api/queries';
import { ColourBar } from './ColourBar';
import { significant } from './format';
import { MapView, type Region } from './MapView';
import { SpectrumPlot } from './SpectrumPlot';
import { Segments } from './ui/inputs';
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

// The fewest pixels a side of the window of a zoomed map has
const LEAST = 8;

export function CubeViewer({
  layers,
  shape,
  region = null,
  channels = null,
  kept = null,
  onRegion,
  onChannels,
  actions,
}: {
  layers: Layer[];
  // Its shape in the order of the model (x, y, then its channels)
  shape: number[];
  // The crops (pixels and channels from 0, inclusive; null: none), and the
  // range of the values kept; how a crop is dragged or cleared (none: not)
  region?: Region | null;
  channels?: [number, number] | null;
  kept?: [number | null, number | null] | null;
  onRegion?: (region: Region | null) => void;
  onChannels?: (channels: [number, number] | null) => void;
  // What the head of the viewer has at its right (e.g. a toggle)
  actions?: ReactNode;
}) {
  const [nx, ny, nchannels] = shape;
  const [layerKey, setLayerKey] = useState<Layer['key']>('data');
  const [view, setView] = useState<View>('channel');
  const [tool, setTool] = useState<'inspect' | 'crop'>('inspect');
  const [channel, setChannel] = useState(Math.floor(nchannels / 2));
  const [pixel, setPixel] = useState<[number, number]>([Math.floor(nx / 2), Math.floor(ny / 2)]);
  const [snr, setSnr] = useState(SNR);
  // The window of the map shown (null: all of it)
  const [zoom, setZoom] = useState<Region | null>(null);
  // (a layer gone, the data; the moments are of the data alone)
  const layer = layers.find((l) => l.key === layerKey) ?? layers[0];
  const data = layers[0];
  const shown: View = layer.key === 'data' ? view : 'channel';
  const cropping = onRegion !== undefined && tool === 'crop';
  const plane = useChannel(layer.path, layer.hdu, channel);
  const moments = useMoments(data.path, data.hdu, snr, shown !== 'channel');
  const spectrum = useSpectrum(layer.path, layer.hdu, ...pixel);
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
  // (Esc clears the crop, as the crop tool is in use)
  useEffect(() => {
    if (!cropping) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onRegion?.(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [cropping, onRegion]);
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
        {actions}
      </div>
      <div className="viewer-body">
        <nav className="rail" aria-label="The views and tools of the cube">
          {onRegion && (
            <>
              <div className="rail-label">tool</div>
              <button aria-pressed={tool === 'inspect'} title="Inspect: a click picks a pixel; a drag pans the zoomed map" onClick={() => setTool('inspect')}>
                Inspect
              </button>
              <button aria-pressed={tool === 'crop'} title="Crop: a drag on the map crops it (Esc clears the crop)" onClick={() => setTool('crop')}>
                Crop
              </button>
              <button className="rail-clear" disabled={region === null} onClick={() => onRegion(null)}>
                clear crop
              </button>
            </>
          )}
          <div className="rail-label">view</div>
          <button aria-pressed={shown === 'channel'} title="One channel of the cube at a time" onClick={() => setView('channel')}>
            Channel
          </button>
          <div className="rail-label" title="Quick moment maps of the data: a look, measured here; the dataset stays a cube">
            moments
          </div>
          {MOMENTS.map(({ view: v, title, text }, order) => (
            <button
              key={v}
              aria-pressed={shown === v}
              disabled={layer.key !== 'data'}
              title={layer.key === 'data' ? text : 'The moments are of the data'}
              onClick={() => setView(v)}
            >
              <span className="order">{order}</span>
              {title}
            </button>
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
                    drag={cropping ? 'region' : 'pan'}
                    mark={pixel}
                    region={region}
                    onPick={(x, y) => setPixel([x, y])}
                    onRegion={cropping ? onRegion : undefined}
                    onZoom={zoomAt}
                    onPan={panBy}
                    label={shown === 'channel' ? `A channel of the ${layer.title.toLowerCase()}` : `The ${shown} map of the data`}
                  />
                  <div className="map-zoom">
                    <button
                      disabled={region === null}
                      title="Zoom to the crop (the wheel zooms about the pointer)"
                      onClick={() => region && setZoom(region)}
                    >
                      crop
                    </button>
                    <button disabled={zoom === null} title="Show the whole map" onClick={() => setZoom(null)}>
                      all
                    </button>
                  </div>
                </div>
                <ColourBar range={range} diverging={colours !== 'sequential'} unit={mapUnit(shown, layer.unit)} />
              </div>
            )}
          </div>
          {shown === 'channel' ? (
            <div className="viewer-controls">
              <span>
                channel {channel + 1} / {nchannels}
              </span>
              <StepSlider min={0} max={nchannels - 1} value={channel} format={(c) => `channel ${c + 1}`} label="Channel" onChange={setChannel} />
              <span>{plane.data?.spectral == null ? '' : `${significant(plane.data.spectral, 6)} ${plane.data.unit}`}</span>
            </div>
          ) : (
            <div className="viewer-controls">
              <span title="The moments are of the voxels connected to those above this many times the noise, down to half of it, in the larger regions (0: all of them)">
                moments of the signal from
              </span>
              <StepSlider
                min={0}
                max={6}
                step={0.5}
                value={snr}
                format={(x) => (x === 0 ? 'all' : `${x}σ`)}
                label="The signal-to-noise ratio of the mask of the moments"
                onChange={setSnr}
              />
              <span>{snr === 0 ? 'all the voxels' : `${snr}σ`}</span>
            </div>
          )}
          <div className="spectrum-head">
            <span>
              {layer.key === 'data' ? 'spectrum' : layer.title.toLowerCase()} of the pixel ({pixel[0]}, {pixel[1]})
            </span>
            {onChannels && (
              <span>
                {channels === null ? (
                  'drag to crop the channels'
                ) : (
                  <button className="link" onClick={() => onChannels(null)}>
                    clear the channels
                  </button>
                )}
              </span>
            )}
          </div>
          {spectrum.error && <p className="error">{spectrum.error.message}</p>}
          {spectrum.data && (
            <SpectrumPlot
              values={spectrum.data.values}
              spectral={spectrum.data.spectral}
              unit={spectrum.data.unit}
              valueUnit={layer.unit}
              channel={channel}
              band={channels}
              onPick={(picked) => {
                setChannel(picked);
                setView('channel');
              }}
              onBand={onChannels}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// The size of a map of nx by ny pixels that fits the space of its stage,
// beside its colour bar (undefined until the stage is measured)
function useFitted(nx: number, ny: number) {
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
function robustRange(values: Float32Array, colours: 'sequential' | 'median' | 'zero'): [number, number] {
  const sorted = Array.from(values.filter(Number.isFinite)).sort((a, b) => a - b);
  if (sorted.length === 0) return [0, 1];
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const [low, high] = [at(0.01), at(0.99)];
  if (colours === 'sequential') return [low, high];
  const middle = colours === 'median' ? at(0.5) : 0;
  const half = Math.max(middle - low, high - middle) || 1;
  return [middle - half, middle + half];
}
