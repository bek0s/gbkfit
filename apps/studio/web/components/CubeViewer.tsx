// A cube, to look at: its data, or the cube of their errors or of their
// mask (its layers), one map at a time: a channel, or one of the quick
// moment maps of the data (a look, measured here; not part of the
// dataset, as its corner says; unless the moments are shown elsewhere,
// e.g. of a galaxy), chosen in its head: the first three as segments, the
// others in a menu beside them; in a map panel (see MapPanel: as every map
// of the Studio). Its selection (a box of the map, see MapPanel; a band of
// channels, dragged on the spectrum) is made to look at: the total
// spectrum is of it, and the spectrum zooms to the band; it is never kept.
// A crop (an edit of the dataset, set from the selection; see Edits) is
// outlined. Under the map, the channel shown, or the mask of the moments
// (grown by prep's tools from seeds above a number of times the noise), in
// a row of one height, so that the map keeps its size; and the spectrum of
// the pixel under the pointer, or of the one picked (a click), or else the
// total of the selection (or of the cube), with the channel marked (a
// click on the spectrum shows its channel; a drag selects a band). What
// the spectrum is of, and the band, are cleared at its head, or by Esc.
// The values out of the range kept are blank on the channel map of the
// data.
import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { FrameArray } from '../api/frames';
import { useChannel, useMoments, useSpectrum } from '../api/queries';
import { significant } from './format';
import { ClearIcon, ZoomSelectionIcon, ZoomWholeIcon } from './icons';
import { MapPanel, robustRange } from './MapPanel';
import type { Region } from './MapView';
import { SpectrumPlot, valueRange } from './SpectrumPlot';
import type { SkyFrame } from './sky';
import { IconButton } from './ui/basics';
import { Check, NumberField, Segments } from './ui/inputs';
import { DropMenu } from './ui/overlays';
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

type Moment = 'flux' | 'velocity' | 'dispersion' | 'skewness' | 'kurtosis' | 'h3' | 'h4';
type View = 'channel' | Moment;

// The moment maps (by order), what they are, how their colours run
// (sequential, diverging about their median, e.g. velocities, or about 0),
// and how they are measured: as moments, or by a Gauss-Hermite series
// fitted to each spectrum (h3, h4)
type Method = 'moments' | 'gauss_hermite';
const MOMENTS: { view: Moment; title: string; text: string; colours: 'sequential' | 'median' | 'zero'; method?: Method }[] = [
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
  { view: 'h3', title: 'h3', text: 'Of a Gauss-Hermite series fitted to each spectrum: its asymmetry (0 for a Gaussian)', colours: 'zero', method: 'gauss_hermite' },
  { view: 'h4', title: 'h4', text: 'Of a Gauss-Hermite series fitted to each spectrum: peaked (above 0) or flat-topped (below 0)', colours: 'zero', method: 'gauss_hermite' },
];

// (those of the segments, and those of the menu beside them)
const FIRST_MOMENTS = MOMENTS.slice(0, 3);
const MORE_MOMENTS = MOMENTS.slice(3);

// The signal-to-noise ratio of the mask of the moments, at first (see the
// server's views)
const SNR = 5;

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
  frame = null,
  noiseless = false,
  moments: which = 'all',
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
  // The pixels of the map on the sky (none: not known)
  frame?: SkyFrame | null;
  // Whether it is a model without noise (e.g. a render of a galaxy): its
  // moments are of all its voxels at first, and are not of a dataset
  noiseless?: boolean;
  // Which of its quick moment maps can be shown: all, those above the
  // dispersion alone (where the others are shown another way, e.g. the
  // maps of a galaxy), or none
  moments?: 'all' | 'higher' | 'none';
}) {
  const [nx, ny, nchannels] = shape;
  const [layerKey, setLayerKey] = useState<Layer['key']>('data');
  const [view, setView] = useState<View>('channel');
  // (within the cube: one made again can have fewer channels, e.g. a render)
  const [picked, setChannel] = useState(Math.floor(nchannels / 2));
  const channel = Math.min(picked, nchannels - 1);
  // The pixel picked (none: the total spectrum is shown)
  const [pixel, setPixel] = useState<[number, number] | null>(null);
  const [snr, setSnr] = useState(noiseless ? 0 : SNR);
  // The window of the spectrum shown (first and last channel; null: all)
  const [zoomedSpectrum, setSpectrumZoom] = useState<[number, number] | null>(null);
  const spectrumZoom = zoomedSpectrum !== null && zoomedSpectrum[1] < nchannels ? zoomedSpectrum : null;
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
  const withMoments = which !== 'none';
  const shown: View = withMoments && layer.key === 'data' ? view : 'channel';
  const ofData = layer.key === 'data';
  const more = MORE_MOMENTS.find((m) => m.view === shown);
  const plane = useChannel(layer.path, layer.hdu, channel);
  const method = MOMENTS.find((m) => m.view === shown)?.method ?? 'moments';
  const moments = useMoments(data.path, data.hdu, snr, shown !== 'channel', method);
  const spectrum = useSpectrum(layer.path, layer.hdu, shownPixel, selection);
  // (a model without noise starts at the brightest channel of its total
  // spectrum, as its middle one may hold none of its light, e.g. between
  // two lines; once, before a channel is picked)
  const started = useRef(!noiseless);
  useEffect(() => {
    if (started.current || !spectrum.data || shownPixel !== null || selection !== null) return;
    started.current = true;
    const values = spectrum.data.values;
    let peak = 0;
    for (let c = 1; c < values.length; c++) if ((values[c] ?? -Infinity) > (values[peak] ?? -Infinity)) peak = c;
    setChannel(peak);
  }, [spectrum.data, shownPixel, selection]);
  // The text of a channel (its number, from 0 as pixels are, and its
  // coordinate) and of a band of channels (their coordinates, or numbers
  // if unknown)
  const spectralOf = (c: number) => spectrum.data?.spectral?.[c];
  const spectralUnit = spectrum.data?.unit ?? '';
  const channelText = (c: number) => {
    const at = spectralOf(c);
    return at === undefined ? `channel ${c}` : `${c} · ${significant(at, 4)} ${spectralUnit}`;
  };
  const bandText = ([c0, c1]: [number, number]) => {
    const [a, b] = [spectralOf(c0), spectralOf(c1)];
    return a === undefined || b === undefined
      ? `channels ${c0}–${c1}`
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
  const colours = MOMENTS.find((m) => m.view === shown)?.colours ?? 'sequential';
  const map: FrameArray | undefined =
    shown === 'channel' ? plane.data?.plane : moments.data?.find((array) => array.info.name === shown);
  // (the mask, 0 to 1; the other cubes over the range of their values)
  const range: [number, number] =
    (shown !== 'channel' ? map && robustRange(map.values, colours) : layer.key === 'mask' ? [0, 1] : plane.data?.range) ?? [0, 1];
  const error = shown === 'channel' ? plane.error : moments.error;
  // (the arrow keys step through the channels, as a channel is shown, unless
  // they move something else: a field, a slider, a choice)
  useEffect(() => {
    if (shown !== 'channel') return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      const target = event.target as HTMLElement | null;
      if (target && (target.isContentEditable || target.closest('input, textarea, select, [role="slider"], [role="radiogroup"], [role="listbox"], [role="menu"]'))) return;
      event.preventDefault();
      const [first, last] = spectrumZoom ?? [0, nchannels - 1];
      setChannel((c) => Math.min(last, Math.max(first, Math.min(c, nchannels - 1) + (event.key === 'ArrowRight' ? 1 : -1))));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [shown, spectrumZoom, nchannels]);
  // (Esc clears the pixel picked and the band; the map, its selection)
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onBand(null);
        setPixel(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onBand]);
  // Zoom the spectrum about a channel by a factor, keeping it where it is;
  // out to all the channels, the window is none
  const zoomSpectrum = (at: number, factor: number) => {
    const [c0, c1] = spectrumZoom ?? [0, nchannels - 1];
    const n = Math.min(nchannels, Math.max(LEAST, Math.round((c1 - c0 + 1) / factor)));
    if (n >= nchannels) return setSpectrumZoom(null);
    const first = Math.min(nchannels - n, Math.max(0, Math.round(at - ((at - c0) * n) / (c1 - c0 + 1))));
    setSpectrumZoom([first, first + n - 1]);
  };
  return (
    <div className="viewer">
      <div className="viewer-head">
        {withMoments && (
          <div className="viewer-views">
            <Segments<View>
              items={[
                { value: 'channel', label: 'Channel', title: 'One channel of the cube at a time' },
                ...(which === 'all' ? FIRST_MOMENTS : []).map((m) => ({ value: m.view, label: m.title, disabled: !ofData, title: ofData ? `≈ ${m.text}` : 'The moments are of the data' })),
              ]}
              value={more ? null : shown}
              onChange={setView}
              label="The view of the cube"
            />
            <DropMenu
              label="More moments"
              at="end"
              width={320}
              entries={MORE_MOMENTS.map((m) => ({ label: m.title, description: m.text, current: shown === m.view, disabled: !ofData, onClick: () => setView(m.view) }))}
              target={
                <button className="viewer-more" data-on={more ? true : undefined} title="The shape of each spectrum: its skewness and kurtosis, or h3 and h4 of a Gauss-Hermite series">
                  {more ? more.title : 'More'} ▾
                </button>
              }
            />
          </div>
        )}
        {layers.length > 1 ? (
          <Segments items={layers.map((l) => ({ value: l.key, label: l.title }))} value={layer.key} onChange={setLayerKey} label="The cube shown" />
        ) : (
          !withMoments && <h2>{data.title}</h2>
        )}
        <span className="viewer-head-gap" />
        {actions}
      </div>
      <div className="viewer-body">
        <div className="viewer-main">
          <MapPanel
            values={map?.values ?? null}
            shape={[ny, nx]}
            range={range}
            diverging={colours !== 'sequential'}
            unit={mapUnit(shown, layer.unit)}
            stretchKey={`${layer.key}/${shown}`}
            frame={frame}
            selection={selection}
            onSelection={(region) => {
              // (the spectrum then of the selection)
              setPixel(null);
              onSelection(region);
            }}
            kept={shown === 'channel' && layer.key === 'data' ? kept : null}
            mark={pixel ?? undefined}
            onPick={(x, y) => setPixel([x, y])}
            onHover={hoverAt}
            outline={crop}
            label={shown === 'channel' ? `A channel of the ${layer.title.toLowerCase()}` : `The ${shown} map of the data`}
            note={
              error ? (
                <p className="error">{error.message}</p>
              ) : shown !== 'channel' && moments.isPending ? (
                <p className="note">{method === 'gauss_hermite' ? 'Fitting a Gauss-Hermite series to each spectrum…' : 'Measuring the moments of the cube…'}</p>
              ) : null
            }
            tags={
              shown !== 'channel' && (
                <span
                  className="map-source derived"
                  title={noiseless ? 'A quick moment map, measured here from the cube' : 'A quick moment map of the data, measured here from the cube; the dataset stays a cube'}
                >
                  {noiseless ? '≈ quick look' : '≈ quick look · not part of the dataset'}
                </span>
              )
            }
          />
          {shown === 'channel' ? (
            // (the channel shown: set on the spectrum, or stepped by the
            // arrow keys)
            <div className="viewer-controls viewer-controls-row" title="A click on the spectrum shows its channel; ← and → step through them">
              <span className="control-text">Channel {channelText(channel)}</span>
              <span className="viewer-channel-hint">click the spectrum · ← →</span>
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
            <span className="spectrum-what">
              <span
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
              {hover === null && pixel !== null && (
                <IconButton label="Clear the pixel picked (Esc)" onClick={() => setPixel(null)}>
                  <ClearIcon />
                </IconButton>
              )}
              {band !== null && (
                <>
                  <span title="The channels selected on the spectrum">· {bandText(band)}</span>
                  <IconButton label="Clear the channels (Esc)" onClick={() => onBand(null)}>
                    <ClearIcon />
                  </IconButton>
                </>
              )}
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

// The unit of a map, from that of the values of the cube
function mapUnit(view: View, unit: string | null): string {
  if (view === 'channel') return unit ?? '';
  if (view === 'flux') return unit ? `${unit} km/s` : 'km/s';
  if (view === 'velocity' || view === 'dispersion') return 'km/s';
  return '';
}
