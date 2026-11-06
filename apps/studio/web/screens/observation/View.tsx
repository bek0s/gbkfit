// The view of an observation: its stages at a glance (each leads to its
// section of the dock), the sky as it sees it (the map of its data, or its
// grid, with its PSF and the beam of the data drawn to scale at a corner),
// its spectral resolution (the LSF against the channels), and its checks.
import type { ArrayInfo } from '../../api/client';
import { useChannel, useMoments } from '../../api/queries';
import { robustRange, useFitted } from '../../components/CubeViewer';
import { useStretches } from '../../components/colours';
import { MapView, useZoom } from '../../components/MapView';
import { RegionsBar, useSkyRegions } from '../../components/RegionsBar';
import { frameOf } from '../../components/sky';
import { useStudio } from '../../studio';
import { showSection } from '../../components/ui/dock';
import type { Options } from '../../components/SchemaForm';
import { glanceOf, STAGES, type Check, type Ellipse } from './model';

export function Stages({ options, titleOf, withData }: { options: Options; titleOf: (family: string, type: unknown) => string; withData: boolean }) {
  return (
    <ol className="ob-stages" aria-label="The stages of the observation">
      {STAGES.filter((stage) => stage.key !== 'likelihood' || withData).map((stage) => (
        <li key={stage.key}>
          <button onClick={() => showSection(stage.section)}>
            <span className="ob-stage-title">{stage.title}</span>
            <span className="ob-stage-glance">{glanceOf(stage.key, options, titleOf)}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}

// The data as the observation sees them: an array of a file and how to
// make a map of it (a cube: its quick moment 0; a map: itself)
export interface SkyData {
  path: string;
  array: ArrayInfo;
  cube: boolean;
}

export function Sky({ data, grid, psf, beam }: { data: SkyData | null; grid: { size: [number, number]; step: [number, number] } | null; psf: Ellipse | null; beam: Ellipse | null }) {
  return (
    <figure className="ob-figure ob-sky">
      <figcaption>
        <span>On the sky</span>
        <Legend psf={psf} beam={beam} />
      </figcaption>
      {data ? <DataSky data={data} psf={psf} beam={beam} /> : grid ? <GridSky grid={grid} psf={psf} /> : <p className="ob-empty">Give the data, or the grid of the observable, to see the sky.</p>}
    </figure>
  );
}

function Legend({ psf, beam }: { psf: Ellipse | null; beam: Ellipse | null }) {
  return (
    <span className="ob-legend">
      {psf && (
        <span>
          <i className="ob-key-psf" /> PSF {size(psf)}
        </span>
      )}
      {beam && (
        <span>
          <i className="ob-key-beam" /> beam of the data {size(beam)}
        </span>
      )}
    </span>
  );
}

function size(ellipse: Ellipse): string {
  const r = (v: number) => Number(v.toPrecision(3));
  return ellipse.major === ellipse.minor ? `${r(ellipse.major)}″` : `${r(ellipse.major)}″ × ${r(ellipse.minor)}″ at ${r(ellipse.angle)}°`;
}

// The map of the data, with the ellipses at its lower left corner
function DataSky({ data, psf, beam }: { data: SkyData; psf: Ellipse | null; beam: Ellipse | null }) {
  const moments = useMoments(data.path, data.array.index, 3, data.cube);
  const plane = useChannel(data.cube ? null : data.path, data.array.index, 0);
  const map = data.cube ? moments.data?.find((array) => array.info.name === 'flux') : plane.data?.plane;
  const [ny, nx] = map?.shape ?? [1, 1];
  // (as every map: zoomed, panned, and a region drawn on it, with the
  // statistics of the map in it; Esc clears it)
  const { zoom, setZoom, zoomAt, panBy, shown } = useZoom(nx, ny);
  const { project } = useStudio();
  const regions = useSkyRegions(project, frameOf(data.array.step, data.array.rpix), [nx, ny]);
  const region = regions.active?.pixels ?? null;
  const [stretch, setStretch] = useStretches('observation/data');
  const across = shown[1] - shown[0] + 1;
  const fitted = useFitted(across, shown[3] - shown[2] + 1, 0);
  const step = data.array.step ? Math.abs(data.array.step[0]) : null;
  const error = (data.cube ? moments.error : plane.error)?.message;
  return (
    <div className="ob-stage" ref={fitted.stage}>
      {error && <p className="error">{error}</p>}
      {!map && !error && <p className="ob-empty">{data.cube ? 'Measuring the intensity of the cube…' : 'Reading the map…'}</p>}
      {map && fitted.size && (
        <div className="ob-canvas" style={{ width: fitted.size[0], height: fitted.size[1] }}>
          <MapView
            values={map.values}
            shape={map.shape}
            range={robustRange(map.values, 'sequential')}
            window={zoom}
            region={region}
            regions={regions.placed.filter((r) => r.pixels && r.id !== regions.active?.id).map((r) => r.pixels!)}
            onRegion={regions.add}
            onZoom={zoomAt}
            onPan={panBy}
            onWhole={() => setZoom(null)}
            stretch={stretch}
            onStretch={setStretch}
            label="The data"
          />
          {step && <Ellipses width={fitted.size[0]} height={fitted.size[1]} scale={fitted.size[0] / (across * step)} psf={psf} beam={beam} />}
          <div className="map-regions-over">
            <RegionsBar regions={regions} values={map.values} nx={nx} unit="" />
          </div>
        </div>
      )}
    </div>
  );
}

// The grid of an observable without data: a frame of its size on the sky
function GridSky({ grid, psf }: { grid: { size: [number, number]; step: [number, number] }; psf: Ellipse | null }) {
  const [nx, ny] = grid.size;
  const fitted = useFitted(nx, ny, 0);
  const [sx, sy] = grid.step.map(Math.abs);
  return (
    <div className="ob-stage" ref={fitted.stage}>
      {fitted.size && (
        <div
          className="ob-canvas ob-grid"
          // (the lines of the pixels, if they are wide enough to see)
          style={{ width: fitted.size[0], height: fitted.size[1], ['--cell' as string]: fitted.size[0] / nx >= 6 ? `${fitted.size[0] / nx}px` : '0px' }}
        >
          <span className="ob-grid-size">
            {nx} × {ny} px of {sx === sy ? `${sx}″` : `${sx}″ × ${sy}″`} · {Number((nx * sx).toPrecision(4))}″ × {Number((ny * sy).toPrecision(4))}″
          </span>
          <Ellipses width={fitted.size[0]} height={fitted.size[1]} scale={fitted.size[0] / (nx * sx)} psf={psf} beam={null} />
        </div>
      )}
    </div>
  );
}

// The ellipses of the PSF and of the beam at the lower left corner of a
// map (north up, east to the left), to the scale of the map (screen
// pixels per arcsec)
function Ellipses({ width, height, scale, psf, beam }: { width: number; height: number; scale: number; psf: Ellipse | null; beam: Ellipse | null }) {
  const largest = Math.max(psf?.major ?? 0, beam?.major ?? 0) * scale;
  if (largest === 0) return null;
  const margin = 14;
  const [cx, cy] = [margin + largest / 2, height - margin - largest / 2];
  // (a position angle east of north: from up, towards the left)
  const shape = (ellipse: Ellipse, className: string) => (
    <ellipse
      className={className}
      cx={cx}
      cy={cy}
      rx={(ellipse.major * scale) / 2}
      ry={(ellipse.minor * scale) / 2}
      transform={`rotate(${-90 - ellipse.angle} ${cx} ${cy})`}
    />
  );
  return (
    <svg className="ob-ellipses" width={width} height={height} aria-hidden>
      <rect className="ob-ellipse-box" x={cx - largest / 2 - 6} y={cy - largest / 2 - 6} width={largest + 12} height={largest + 12} rx={6} />
      {beam && shape(beam, 'ob-beam')}
      {psf && shape(psf, 'ob-psf')}
    </svg>
  );
}

// The LSF against the channels: its profile at half maximum over the
// width of a channel, to scale
export function Resolution({ fwhm, channel, named }: { fwhm: number | null; channel: number | null; named: boolean }) {
  return (
    <figure className="ob-figure ob-resolution">
      <figcaption>
        <span>Spectral resolution</span>
      </figcaption>
      {fwhm === null ? (
        <p className="ob-empty">{named ? 'Measuring the LSF…' : 'No LSF: a perfect spectrograph.'}</p>
      ) : (
        <ResolutionPlot fwhm={fwhm} channel={channel} />
      )}
    </figure>
  );
}

function ResolutionPlot({ fwhm, channel }: { fwhm: number; channel: number | null }) {
  const [w, h] = [300, 92];
  const span = 4 * Math.max(fwhm, channel ?? 0);
  const x = (v: number) => w / 2 + (v / span) * w;
  const sigma = fwhm / 2.3548;
  const points = Array.from({ length: 121 }, (_, i) => {
    const v = -span / 2 + (i / 120) * span;
    return `${x(v).toFixed(1)},${(h - 18 - (h - 30) * Math.exp(-0.5 * (v / sigma) ** 2)).toFixed(1)}`;
  });
  const edges = channel ? Array.from({ length: Math.floor(span / channel) + 2 }, (_, i) => (i - Math.floor(span / channel / 2) - 0.5) * channel) : [];
  return (
    <>
      <svg className="ob-plot" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-label="The LSF against the channels">
        {edges.map((edge) => (
          <line key={edge} className="ob-channel" x1={x(edge)} x2={x(edge)} y1={6} y2={h - 18} />
        ))}
        <polyline className="ob-lsf" points={points.join(' ')} />
        <line className="ob-fwhm" x1={x(-fwhm / 2)} x2={x(fwhm / 2)} y1={h - 18 - (h - 30) / 2} y2={h - 18 - (h - 30) / 2} />
        <line className="ob-axis" x1={0} x2={w} y1={h - 18} y2={h - 18} />
      </svg>
      <p className="ob-caption">
        FWHM {Number(fwhm.toPrecision(3))} km/s
        {channel ? ` · channels of ${Number(channel.toPrecision(3))} km/s (${Number((fwhm / channel).toPrecision(2))} channels)` : ''}
      </p>
    </>
  );
}

export function Checks({ checks }: { checks: Check[] }) {
  return (
    <figure className="ob-figure ob-checks">
      <figcaption>
        <span>Checks</span>
      </figcaption>
      <ul>
        {checks.map((check) => (
          <li key={check.text} data-ok={check.ok === null ? 'note' : String(check.ok)}>
            <span className="ob-check-mark">{check.ok === null ? '·' : check.ok ? '✓' : '!'}</span>
            {check.text}
          </li>
        ))}
      </ul>
    </figure>
  );
}
