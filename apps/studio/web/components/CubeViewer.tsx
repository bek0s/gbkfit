// A cube, to look at: a channel, chosen with a slider, or its quick moment
// maps; and the spectrum of a pixel, picked on a map, with the channel
// marked (a click on the spectrum picks a channel).
import { useState } from 'react';
import { useChannel, useMoments, useSpectrum } from '../api/queries';
import { significant } from './format';
import { MapView } from './MapView';
import { SpectrumPlot } from './SpectrumPlot';

// The titles of the moment maps, and whether they diverge (about their
// median: velocities)
const MOMENTS: { name: string; title: string; diverging: boolean }[] = [
  { name: 'moment0', title: 'flux', diverging: false },
  { name: 'moment1', title: 'velocity (km/s)', diverging: true },
  { name: 'moment2', title: 'dispersion (km/s)', diverging: false },
];

export function CubeViewer({ path, hdu, shape }: { path: string; hdu: number; shape: number[] }) {
  // (its shape in the order of the model: x, y, then its channels last)
  const [nx, ny, nchannels] = [shape[0], shape[1], shape[shape.length - 1]];
  const [view, setView] = useState<'channels' | 'moments'>('channels');
  const [channel, setChannel] = useState(Math.floor(nchannels / 2));
  const [pixel, setPixel] = useState<[number, number]>([Math.floor(nx / 2), Math.floor(ny / 2)]);
  const plane = useChannel(path, hdu, channel);
  const spectrum = useSpectrum(path, hdu, ...pixel);
  const moments = useMoments(path, hdu, view === 'moments');
  const spectral = plane.data?.spectral;
  return (
    <div className="card viewer">
      <div className="card-head viewer-head">
        <h2>The cube</h2>
        <div className="seg" role="group" aria-label="View">
          <button aria-pressed={view === 'channels'} onClick={() => setView('channels')}>
            Channels
          </button>
          <button aria-pressed={view === 'moments'} onClick={() => setView('moments')}>
            Moments
          </button>
        </div>
      </div>
      <div className="card-body">
        {view === 'channels' ? (
          <>
            {plane.error && <p className="error">{plane.error.message}</p>}
            {plane.data && (
              <MapView
                values={plane.data.plane.values}
                shape={plane.data.plane.shape}
                range={plane.data.range}
                mark={pixel}
                onPick={(x, y) => setPixel([x, y])}
                label={`Channel ${channel + 1} of the cube`}
              />
            )}
            <div className="channel-controls">
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
              <span>{spectral === null || spectral === undefined ? '' : `${significant(spectral, 6)} ${plane.data?.unit}`}</span>
            </div>
          </>
        ) : (
          <>
            {moments.isPending && <p className="note">Measuring the moments of the cube…</p>}
            {moments.error && <p className="error">{moments.error.message}</p>}
            <div className="moments">
              {MOMENTS.map(({ name, title, diverging }) => {
                const map = moments.data?.find((array) => array.info.name === name);
                if (map === undefined) return null;
                const range = robustRange(map.values, diverging);
                return (
                  <figure key={name}>
                    <MapView
                      values={map.values}
                      shape={map.shape}
                      range={range}
                      diverging={diverging}
                      mark={pixel}
                      onPick={(x, y) => setPixel([x, y])}
                      label={`The ${title} map of the cube`}
                    />
                    <figcaption>
                      <b>{title}</b>
                      <span>
                        {significant(range[0], 3)} to {significant(range[1], 3)}
                      </span>
                    </figcaption>
                  </figure>
                );
              })}
            </div>
          </>
        )}
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
              setView('channels');
            }}
          />
        )}
      </div>
    </div>
  );
}

// The range of the colours of a map: from the 1st to the 99th percentile
// of its values, or about their median, as wide on each side, if it
// diverges
function robustRange(values: Float32Array, diverging: boolean): [number, number] {
  const sorted = Array.from(values.filter(Number.isFinite)).sort((a, b) => a - b);
  if (sorted.length === 0) return [0, 1];
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const [low, high] = [at(0.01), at(0.99)];
  if (!diverging) return [low, high];
  const middle = at(0.5);
  const half = Math.max(middle - low, high - middle);
  return [middle - half, middle + half];
}
