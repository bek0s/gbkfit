// A cube, to look at: one map at a time (a channel, or one of its quick
// moment maps), with its colour bar, as large as the screen allows; under
// it, the channel (a slider) or the mask of the moments (grown by prep's
// tools from seeds above a number of times the noise); and the spectrum of a
// pixel, picked on the map, with the channel marked (a click on the
// spectrum picks a channel).
import { useState } from 'react';
import type { FrameArray } from '../api/frames';
import { useChannel, useMoments, useSpectrum } from '../api/queries';
import { ColourBar } from './ColourBar';
import { significant } from './format';
import { MapView } from './MapView';
import { SpectrumPlot } from './SpectrumPlot';

type View = 'channel' | 'flux' | 'velocity' | 'dispersion' | 'skewness' | 'kurtosis';

// The maps, and how their colours run: sequential, diverging about their
// median (velocities), or diverging about 0
const VIEWS: { view: View; title: string; colours: 'sequential' | 'median' | 'zero' }[] = [
  { view: 'channel', title: 'Channel', colours: 'sequential' },
  { view: 'flux', title: 'Flux', colours: 'sequential' },
  { view: 'velocity', title: 'Velocity', colours: 'median' },
  { view: 'dispersion', title: 'Dispersion', colours: 'sequential' },
  { view: 'skewness', title: 'Skewness', colours: 'zero' },
  { view: 'kurtosis', title: 'Kurtosis', colours: 'zero' },
];

// The signal-to-noise ratio of the mask of the moments, at first (see the
// server's views)
const SNR = 5;

export function CubeViewer({
  path,
  hdu,
  shape,
  unit,
}: {
  path: string;
  hdu: number;
  // Its shape in the order of the model (x, y, then its channels), and
  // the unit of its values
  shape: number[];
  unit: string | null;
}) {
  const [nx, ny, nchannels] = shape;
  const [view, setView] = useState<View>('channel');
  const [channel, setChannel] = useState(Math.floor(nchannels / 2));
  const [pixel, setPixel] = useState<[number, number]>([Math.floor(nx / 2), Math.floor(ny / 2)]);
  const [snr, setSnr] = useState(SNR);
  const plane = useChannel(path, hdu, channel);
  const moments = useMoments(path, hdu, snr, view !== 'channel');
  const spectrum = useSpectrum(path, hdu, ...pixel);
  const { colours } = VIEWS.find((v) => v.view === view)!;
  const map: FrameArray | undefined =
    view === 'channel' ? plane.data?.plane : moments.data?.find((array) => array.info.name === view);
  const range: [number, number] | undefined =
    view === 'channel' ? plane.data?.range : map && robustRange(map.values, colours);
  const error = view === 'channel' ? plane.error : moments.error;
  return (
    <div className="card viewer">
      <div className="card-head viewer-head">
        <h2>The cube</h2>
        <div className="seg" role="group" aria-label="The map">
          {VIEWS.map(({ view: v, title }) => (
            <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>
              {title}
            </button>
          ))}
        </div>
      </div>
      <div className="card-body">
        {error && <p className="error">{error.message}</p>}
        {view !== 'channel' && moments.isPending && <p className="note">Measuring the moments of the cube…</p>}
        {map && range && (
          <div className="viewer-map">
            <MapView
              values={map.values}
              shape={map.shape}
              range={range}
              diverging={colours !== 'sequential'}
              mark={pixel}
              onPick={(x, y) => setPixel([x, y])}
              label={`The ${view} map of the cube`}
            />
            <ColourBar range={range} diverging={colours !== 'sequential'} unit={mapUnit(view, unit)} />
          </div>
        )}
        <div className="viewer-controls">
          {view === 'channel' ? (
            <>
              <span>
                channel {channel + 1} / {nchannels}
              </span>
              <input
                type="range"
                min={0}
                max={nchannels - 1}
                value={channel}
                aria-label="Channel"
                onChange={(event) => setChannel(Number(event.target.value))}
              />
              <span>
                {plane.data?.spectral == null ? '' : `${significant(plane.data.spectral, 6)} ${plane.data.unit}`}
              </span>
            </>
          ) : (
            <>
              <span title="The moments are of the voxels connected to those above this many times the noise, down to half of it, in the larger regions (0: all of them)">
                mask from
              </span>
              <input
                type="range"
                min={0}
                max={6}
                step={0.5}
                value={snr}
                aria-label="The signal-to-noise ratio of the mask"
                onChange={(event) => setSnr(Number(event.target.value))}
              />
              <span>{snr === 0 ? 'no mask' : `${snr}σ`}</span>
            </>
          )}
        </div>
        <div className="spectrum-head">
          spectrum of the pixel ({pixel[0]}, {pixel[1]})
        </div>
        {spectrum.error && <p className="error">{spectrum.error.message}</p>}
        {spectrum.data && (
          <SpectrumPlot
            values={spectrum.data.values}
            spectral={spectrum.data.spectral}
            unit={spectrum.data.unit}
            channel={channel}
            onPick={(picked) => {
              setChannel(picked);
              setView('channel');
            }}
          />
        )}
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
