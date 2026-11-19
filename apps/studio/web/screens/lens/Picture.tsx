// What a lens makes of a source behind it (see the server's LensViewInfo):
// the image plane, with its critical curves and the images of a round test
// source (each pixel shot back to the source plane through the deflection
// of the lens: beta = theta - alpha), and the source plane, with its
// caustics and the source, which a drag moves (not saved).
import { useEffect, useRef } from 'react';
import type { components } from '../../api/schema';

type LensView = components['schemas']['LensViewInfo'];

// A round test source: its centre and its dispersion (arcsec)
export interface Source {
  x: number;
  y: number;
  size: number;
}

// The pixels across a picture
const PIXELS = 288;

export function LensPicture({ view, error, source, onSource }: { view: LensView | undefined; error: string | null; source: Source; onSource?: (source: Source) => void }) {
  if (error) return <div className="ob-view"><p className="ob-empty">{error.replace(/^[A-Za-z]+Error: /, '')}</p></div>;
  if (!view) return <div className="ob-view"><div className="gx-map-wait" /></div>;
  return (
    <div className="ob-view ln-view">
      <div className="ln-planes">
        <figure className="ob-figure">
          <figcaption>
            What it makes of a source <span className="ln-key"><i className="ln-key-critical" /> critical curves</span>
          </figcaption>
          <div className="ln-plane">
            <ImagePlane view={view} source={source} />
            <Curves lines={view.critical} half={view.half} className="ln-critical" />
          </div>
          <p className="ob-caption">image plane · {fixed(2 * view.half)}″ × {fixed(2 * view.half)}″</p>
        </figure>
        <figure className="ob-figure">
          <figcaption>
            The source <span className="ln-key"><i className="ln-key-caustic" /> caustics</span>
          </figcaption>
          <div
            className="ln-plane ln-drag"
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              moved(event, view.half / 2, source, onSource);
            }}
            onPointerMove={(event) => event.buttons === 1 && moved(event, view.half / 2, source, onSource)}
          >
            <SourcePlane source={source} half={view.half / 2} />
            <Curves lines={view.caustics} half={view.half / 2} className="ln-caustic" />
          </div>
          <p className="ob-caption">source plane · {fixed(view.half)}″ × {fixed(view.half)}″</p>
        </figure>
      </div>
      {view.left_out.length > 0 && <p className="ob-text">Not drawn: {view.left_out.join(', ')} (the mass of a galaxy, which its setups lay out).</p>}
    </div>
  );
}

// The images of the source: each pixel of the image plane shot back to the
// source plane by the deflection of the lens (bilinear between its points),
// lit by the source there
function ImagePlane({ view, source }: { view: LensView; source: Source }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const n = view.size;
    const image = context.createImageData(PIXELS, PIXELS);
    for (let j = 0; j < PIXELS; j++) {
      for (let i = 0; i < PIXELS; i++) {
        // (the pixel on the sky, y up; at the grid of the deflection)
        const x = -view.half + ((i + 0.5) / PIXELS) * 2 * view.half;
        const y = view.half - ((j + 0.5) / PIXELS) * 2 * view.half;
        const gx = ((x + view.half) / (2 * view.half)) * (n - 1);
        const gy = ((y + view.half) / (2 * view.half)) * (n - 1);
        const ax = bilinear(view.alpha_x, n, gx, gy);
        const ay = bilinear(view.alpha_y, n, gx, gy);
        const r2 = (x - ax - source.x) ** 2 + (y - ay - source.y) ** 2;
        const light = Math.exp(-r2 / (2 * source.size ** 2));
        const at = (j * PIXELS + i) * 4;
        image.data[at] = 255 * Math.min(1, light * 1.1);
        image.data[at + 1] = 200 * light;
        image.data[at + 2] = 90 * light;
        image.data[at + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [view, source]);
  return <canvas ref={canvas} width={PIXELS} height={PIXELS} aria-label="The images of the source" />;
}

// The source plane: the source, as a disc of its size
function SourcePlane({ source, half }: { source: Source; half: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const image = context.createImageData(PIXELS, PIXELS);
    for (let j = 0; j < PIXELS; j++) {
      for (let i = 0; i < PIXELS; i++) {
        const x = -half + ((i + 0.5) / PIXELS) * 2 * half;
        const y = half - ((j + 0.5) / PIXELS) * 2 * half;
        const light = Math.exp(-((x - source.x) ** 2 + (y - source.y) ** 2) / (2 * source.size ** 2));
        const at = (j * PIXELS + i) * 4;
        image.data.set([255 * Math.min(1, light * 1.1), 200 * light, 90 * light, 255], at);
      }
    }
    context.putImageData(image, 0, 0);
  }, [source, half]);
  return <canvas ref={canvas} width={PIXELS} height={PIXELS} aria-label="The source" />;
}

// Curves of a plane over its picture (arcsec, y up), of a half-width
function Curves({ lines, half, className }: { lines: number[][][]; half: number; className: string }) {
  return (
    <svg className="ln-curves" viewBox={`${-half} ${-half} ${2 * half} ${2 * half}`} preserveAspectRatio="none" aria-hidden="true">
      {lines.map((line, k) => (
        <polyline key={k} className={className} points={line.map(([x, y]) => `${x},${-y}`).join(' ')} vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  );
}

// The source moved to the pointer in its plane (of a half-width, arcsec)
function moved(event: React.PointerEvent<HTMLElement>, half: number, source: Source, onSource?: (source: Source) => void) {
  if (!onSource) return;
  const box = event.currentTarget.getBoundingClientRect();
  const x = -half + ((event.clientX - box.left) / box.width) * 2 * half;
  const y = half - ((event.clientY - box.top) / box.height) * 2 * half;
  onSource({ ...source, x: Number(x.toFixed(3)), y: Number(y.toFixed(3)) });
}

// A map (n by n, row-major, the first row at the lowest y) between its
// points
function bilinear(values: number[], n: number, gx: number, gy: number): number {
  const x0 = Math.max(0, Math.min(n - 2, Math.floor(gx)));
  const y0 = Math.max(0, Math.min(n - 2, Math.floor(gy)));
  const fx = Math.min(1, Math.max(0, gx - x0));
  const fy = Math.min(1, Math.max(0, gy - y0));
  const at = (x: number, y: number) => values[y * n + x];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}

function fixed(value: number): string {
  return Number(value.toPrecision(3)).toString();
}
