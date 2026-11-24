// The view of an observation: its stages at a glance (each leads to its
// section of the dock), the sky as it sees it (its grid, with its PSF
// drawn to scale at a corner), its spectral resolution (the LSF against
// the channels), a patch of the noise of its mock data, and its checks.
import { useEffect, useRef } from 'react';
import { useFitted } from '../../components/MapPanel';
import { showSection } from '../../components/ui/dock';
import type { Options } from '../../components/SchemaForm';
import { glanceOf, STAGES, type Check, type Ellipse } from './model';

export function Stages({ options, titleOf, noise }: { options: Options; titleOf: (family: string, type: unknown) => string; noise: string }) {
  return (
    <ol className="ob-stages" aria-label="The stages of the observation">
      {STAGES.map((stage) => (
        <li key={stage.key}>
          <button onClick={() => showSection(stage.section)}>
            <span className="ob-stage-title">{stage.title}</span>
            <span className="ob-stage-glance">{glanceOf(stage.key, options, titleOf, noise)}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}

// A patch of the noise of the mock data of an observation (none: what it
// takes): white noise of a fixed seed, correlated on the sky as its PSF if
// the noise is (smoothed by a Gaussian of the width of the PSF in pixels),
// in grey, about its mean
const PATCH = 64;

export function NoisePatch({ noise, text, psf, pixel }: { noise: Options | undefined; text: string; psf: Ellipse | null; pixel: number | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const correlated = noise?.spatial_correlation === 'instrument' && psf !== null && pixel !== null && pixel > 0;
  const sigma = correlated ? psf.major / 2.3548 / pixel : 0;
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context || !noise) return;
    const values = smoothed(whiteNoise(PATCH * PATCH), PATCH, sigma);
    let low = Infinity;
    let high = -Infinity;
    for (const v of values) [low, high] = [Math.min(low, v), Math.max(high, v)];
    const image = context.createImageData(PATCH, PATCH);
    values.forEach((v, i) => {
      const grey = Math.round(((v - low) / (high - low || 1)) * 255);
      image.data.set([grey, grey, grey, 255], i * 4);
    });
    context.putImageData(image, 0, 0);
  }, [noise, sigma]);
  return (
    <figure className="ob-figure ob-noise">
      <figcaption>Noise of its mock data</figcaption>
      {noise ? (
        <div className="ob-noise-body">
          <canvas ref={canvas} width={PATCH} height={PATCH} aria-label="A patch of its noise" />
          <span>
            {text}
            <small>
              {PATCH} × {PATCH} pixels{correlated ? `, correlated over ${Number(sigma.toPrecision(2))} px (σ)` : ', independent'}
            </small>
          </span>
        </div>
      ) : (
        <p className="ob-caption">{text === 'None' ? 'None given (see Noise).' : text}</p>
      )}
    </figure>
  );
}

// White Gaussian noise of unit dispersion, of a fixed seed (a patch that
// stays the same as the options change)
function whiteNoise(count: number): Float32Array {
  let seed = 12345;
  const uniform = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) + 0.5) / 4294967296;
  const values = new Float32Array(count);
  for (let i = 0; i < count; i++) values[i] = Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  return values;
}

// A square of values smoothed by a Gaussian of a dispersion in pixels (none
// below a third of a pixel), along x and then y, wrapping around
function smoothed(values: Float32Array, n: number, sigma: number): Float32Array {
  if (sigma < 0.33) return values;
  const half = Math.ceil(3 * sigma);
  const kernel = Array.from({ length: 2 * half + 1 }, (_, i) => Math.exp(-0.5 * ((i - half) / sigma) ** 2));
  const pass = (input: Float32Array, step: (i: number, k: number) => number) => {
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      let sum = 0;
      for (let k = -half; k <= half; k++) sum += kernel[k + half] * input[step(i, k)];
      output[i] = sum;
    }
    return output;
  };
  const along = pass(values, (i, k) => Math.floor(i / n) * n + (((i % n) + k + n) % n));
  return pass(along, (i, k) => ((Math.floor(i / n) + k + n) % n) * n + (i % n));
}

export function Sky({ grid, psf }: { grid: { size: [number, number]; step: [number, number] } | null; psf: Ellipse | null }) {
  return (
    <figure className="ob-figure ob-sky">
      <figcaption>
        <span>On the sky</span>
        {psf && (
          <span className="ob-legend">
            <span>
              <i className="ob-key-psf" /> PSF {size(psf)}
            </span>
          </span>
        )}
      </figcaption>
      {grid ? <GridSky grid={grid} psf={psf} /> : <p className="ob-empty">Give the grid of the observation to see the sky.</p>}
    </figure>
  );
}

function size(ellipse: Ellipse): string {
  const r = (v: number) => Number(v.toPrecision(3));
  return ellipse.major === ellipse.minor ? `${r(ellipse.major)}″` : `${r(ellipse.major)}″ × ${r(ellipse.minor)}″ at ${r(ellipse.angle)}°`;
}

// The grid of an observation: a frame of its size on the sky
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
          <PsfEllipse width={fitted.size[0]} height={fitted.size[1]} scale={fitted.size[0] / (nx * sx)} psf={psf} />
        </div>
      )}
    </div>
  );
}

// The ellipse of the PSF at the lower left corner of a map (north up, east
// to the left), to the scale of the map (screen pixels per arcsec)
function PsfEllipse({ width, height, scale, psf }: { width: number; height: number; scale: number; psf: Ellipse | null }) {
  const largest = (psf?.major ?? 0) * scale;
  if (psf === null || largest === 0) return null;
  const margin = 14;
  const [cx, cy] = [margin + largest / 2, height - margin - largest / 2];
  // (a position angle east of north: from up, towards the left)
  return (
    <svg className="ob-ellipses" width={width} height={height} aria-hidden>
      <rect className="ob-ellipse-box" x={cx - largest / 2 - 6} y={cy - largest / 2 - 6} width={largest + 12} height={largest + 12} rx={6} />
      <ellipse className="ob-psf" cx={cx} cy={cy} rx={(psf.major * scale) / 2} ry={(psf.minor * scale) / 2} transform={`rotate(${-90 - psf.angle} ${cx} ${cy})`} />
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
