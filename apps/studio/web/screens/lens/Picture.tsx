// What a lens makes of a source behind it (see the server's LensViewInfo):
// the image plane, as the images of a round test source (each pixel shot
// back to the source plane through the deflection of the lens: beta =
// theta - alpha), or as the convergence or the magnification of the lens
// (of the Jacobian of its deflection, by finite differences), with its
// critical curves and the handles of the piece selected (dragged, they set
// its parameters); and the source plane, with its caustics and the source,
// which a drag moves (not saved). Each plane has a scale bar and the
// compass of the sky (north up, east left; +x is west).
import { useEffect, useMemo, useRef, useState } from 'react';
import type { components } from '../../api/schema';
import type { Dots, Handles } from './LensScreen';
import { fmt, Track } from './SliderRow';

type LensView = components['schemas']['LensViewInfo'];

// A round test source: its centre and its dispersion (arcsec)
export interface Source {
  x: number;
  y: number;
  size: number;
}

type Mode = 'images' | 'kappa' | 'mu';

const MODES: [Mode, string][] = [
  ['images', 'Images'],
  ['kappa', 'Convergence κ'],
  ['mu', 'Magnification μ'],
];

// The pixels across a picture
const PIXELS = 300;

// The largest test source (arcsec)
const LARGEST_SOURCE = 0.4;

export function LensPicture({
  view,
  error,
  empty,
  source,
  onSource,
  handles,
  dots,
  onPick,
  onClearPick,
  onHandle,
}: {
  view: LensView | undefined;
  error: string | null;
  // Why there is nothing to draw, if so
  empty: string | null;
  source: Source;
  onSource: (source: Source) => void;
  handles: Handles;
  // The centres of the other pieces, and what a click on one does; what
  // Esc does (the piece no longer selected)
  dots: Dots;
  onPick: (piece: number) => void;
  onClearPick: () => void;
  onHandle: (name: string, value: number) => void;
}) {
  const [mode, setMode] = useState<Mode>('images');
  useEffect(() => {
    const key = (event: KeyboardEvent) => event.key === 'Escape' && !(event.target instanceof HTMLInputElement) && onClearPick();
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onClearPick]);
  const [curves, setCurves] = useState(true);
  const [showHandles, setShowHandles] = useState(true);
  if (empty) return <div className="ob-view"><p className="ob-empty">{empty}</p></div>;
  if (error) return <div className="ob-view"><p className="ob-empty">{error.replace(/^[A-Za-z]+Error: /, '')}</p></div>;
  if (!view) return <div className="ob-view"><div className="gx-map-wait" /></div>;
  return (
    <div className="ob-view lp-view">
      <div className="lp-tools">
        <span className="lp-seg" role="tablist" aria-label="What the image plane shows">
          {MODES.map(([key, title]) => (
            <button key={key} type="button" role="tab" aria-selected={mode === key} onClick={() => setMode(key)}>
              {title}
            </button>
          ))}
        </span>
        <label className="lp-check">
          <input type="checkbox" checked={curves} onChange={(event) => setCurves(event.target.checked)} />
          curves
        </label>
        <label className="lp-check">
          <input type="checkbox" checked={showHandles} onChange={(event) => setShowHandles(event.target.checked)} />
          handles
        </label>
        <span className="lp-tools-end">field {fmt(2 * view.half)}″</span>
      </div>
      <div className="lp-planes">
        <figure className="lp-plane-box lp-image-box">
          <figcaption>
            Image plane
            <span className="lp-key" data-off={!curves || undefined}><i className="lp-key-critical" /> critical curves</span>
          </figcaption>
          <ImagePlane view={view} source={source} mode={mode} curves={curves} handles={showHandles ? handles : null} dots={showHandles ? dots : []} onPick={onPick} onHandle={onHandle} />
        </figure>
        <figure className="lp-plane-box lp-source-box">
          <figcaption>
            Source plane
            <span className="lp-key" data-off={!curves || undefined}><i className="lp-key-caustic" /> caustics</span>
          </figcaption>
          <SourcePlane view={view} source={source} curves={curves} onSource={onSource} />
          <div className="lp-test">
            <span>Test source</span>
            <span className="lp-test-xy">
              x {fmt(source.x)}″ · y {fmt(source.y)}″
            </span>
            <span>Size σ</span>
            <span className="lp-test-size">
              <Track at={source.size / LARGEST_SOURCE} onAt={(t) => onSource({ ...source, size: Math.max(0.02, t * LARGEST_SOURCE) })} />
              <span>{fmt(source.size)}″</span>
            </span>
            <span />
            <span className="lp-note">Drag it in its plane: only for the picture, not saved.</span>
          </div>
        </figure>
      </div>
      {view.left_out.length > 0 && <p className="lp-note">Not drawn: {view.left_out.join(', ')} (the mass of a galaxy, which its setups lay out).</p>}
    </div>
  );
}

// The fields of the deflection on its grid, for the maps of the lens: its
// convergence (half the divergence) and the determinant of the Jacobian of
// the lens mapping (1 / magnification), by finite differences
function lensFields(view: LensView) {
  const n = view.size;
  const step = (2 * view.half) / (n - 1);
  const ax = view.alpha_x;
  const ay = view.alpha_y;
  const kappa = new Float64Array(n * n);
  const det = new Float64Array(n * n);
  const d = (values: number[], i: number, j: number, along: 'x' | 'y') => {
    if (along === 'x') {
      const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
      return (values[j * n + b] - values[j * n + a]) / ((b - a) * step);
    }
    const a = Math.max(0, j - 1), b = Math.min(n - 1, j + 1);
    return (values[b * n + i] - values[a * n + i]) / ((b - a) * step);
  };
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const xx = d(ax, i, j, 'x'), yy = d(ay, i, j, 'y'), xy = d(ax, i, j, 'y'), yx = d(ay, i, j, 'x');
      kappa[j * n + i] = 0.5 * (xx + yy);
      det[j * n + i] = (1 - xx) * (1 - yy) - xy * yx;
    }
  }
  return { kappa, det };
}

function ImagePlane({
  view,
  source,
  mode,
  curves,
  handles,
  dots,
  onPick,
  onHandle,
}: {
  view: LensView;
  source: Source;
  mode: Mode;
  curves: boolean;
  handles: Handles;
  dots: Dots;
  onPick: (piece: number) => void;
  onHandle: (name: string, value: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const fields = useMemo(() => lensFields(view), [view]);
  const [pointer, setPointer] = useState<[number, number] | null>(null);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const n = view.size;
    const h = view.half;
    const image = context.createImageData(PIXELS, PIXELS);
    for (let j = 0; j < PIXELS; j++) {
      for (let i = 0; i < PIXELS; i++) {
        const x = -h + ((i + 0.5) / PIXELS) * 2 * h;
        const y = h - ((j + 0.5) / PIXELS) * 2 * h;
        const gx = ((x + h) / (2 * h)) * (n - 1);
        const gy = ((y + h) / (2 * h)) * (n - 1);
        let rgb: [number, number, number];
        if (mode === 'images') {
          const ax = bilinear(view.alpha_x, n, gx, gy);
          const ay = bilinear(view.alpha_y, n, gx, gy);
          const light = Math.exp(-((x - ax - source.x) ** 2 + (y - ay - source.y) ** 2) / (2 * source.size ** 2));
          rgb = [255 * Math.min(1, light * 1.1), 200 * light, 90 * light];
        } else if (mode === 'kappa') {
          const t = Math.sqrt(Math.max(0, Math.min(1, bilinear(fields.kappa, n, gx, gy) / 1.5)));
          rgb = [20 + 235 * t, 14 + 150 * t * t, 40 + 30 * t];
        } else {
          const det = bilinear(fields.det, n, gx, gy);
          const m = Math.min(1, Math.log10(1 + 1 / Math.abs(det)) / 1.6);
          rgb = det > 0 ? [30 + 225 * m, 22 + 150 * m, 20 + 50 * m] : [20 + 60 * m, 30 + 140 * m, 50 + 205 * m];
        }
        image.data.set([rgb[0], rgb[1], rgb[2], 255], (j * PIXELS + i) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }, [view, source, mode, fields]);
  const h = view.half;
  const world = (event: React.PointerEvent<Element>, box: DOMRect): [number, number] => [
    -h + ((event.clientX - box.left) / box.width) * 2 * h,
    h - ((event.clientY - box.top) / box.height) * 2 * h,
  ];
  return (
    <div
      className="lp-plane"
      onPointerMove={(event) => setPointer(world(event, event.currentTarget.getBoundingClientRect()))}
      onPointerLeave={() => setPointer(null)}
    >
      <canvas ref={canvas} width={PIXELS} height={PIXELS} aria-label="What the lens makes of the source" />
      <svg className="lp-overlay" viewBox={`${-h} ${-h} ${2 * h} ${2 * h}`} preserveAspectRatio="none">
        {curves && view.critical.map((line, k) => <polyline key={k} className="lp-critical" points={points(line)} />)}
        <ScaleBar half={h} />
        {dots.map((dot) => (
          <circle key={dot.piece} className="lp-h-pick" cx={dot.x} cy={-dot.y} r={h * 0.018} style={{ fill: dot.colour }} onClick={() => onPick(dot.piece)}>
            <title>Select the piece</title>
          </circle>
        ))}
        {handles && <HandleShapes handles={handles} half={h} onHandle={onHandle} world={world} />}
      </svg>
      <Compass />
      {pointer && (
        <span className="lp-readout">
          x {fmt(pointer[0])}″ · y {fmt(pointer[1])}″
        </span>
      )}
    </div>
  );
}

// The handles of a piece (see handlesOf): a drag of one sets the
// parameters it shows
function HandleShapes({ handles, half, onHandle, world }: { handles: NonNullable<Handles>; half: number; onHandle: (name: string, value: number) => void; world: (event: React.PointerEvent<Element>, box: DOMRect) => [number, number] }) {
  const { keys, values } = handles;
  const v = (key: keyof typeof keys, fallback = 0) => (keys[key] ? values[keys[key]!] : fallback);
  const angle = (dx: number, dy: number) => ((((Math.atan2(-dx, dy) * 180) / Math.PI) % 180) + 180) % 180;
  const r = half * 0.024;
  const drag = (onMove: (x: number, y: number) => void) => ({
    onPointerDown: (event: React.PointerEvent<SVGCircleElement>) => {
      // (no text selected by the drag, which would cancel it; see Track)
      event.preventDefault();
      event.stopPropagation();
      (event.currentTarget.ownerSVGElement ?? event.currentTarget).setPointerCapture(event.pointerId);
      const svg = event.currentTarget.ownerSVGElement!;
      const box = svg.getBoundingClientRect();
      const move = (e: PointerEvent) => onMove(...world(e as unknown as React.PointerEvent<Element>, box));
      const up = () => {
        svg.removeEventListener('pointermove', move);
        svg.removeEventListener('pointerup', up);
      };
      svg.addEventListener('pointermove', move);
      svg.addEventListener('pointerup', up);
    },
  });
  const posa = v('posa');
  const ux = -Math.sin((posa * Math.PI) / 180);
  const uy = Math.cos((posa * Math.PI) / 180);
  const label = { fontSize: half * 0.036, strokeWidth: half * 0.008 };
  if (handles.kind === 'shear') {
    const length = v('gamma') * half * 8;
    return (
      <g>
        <line className="lp-h-line" x1={-length * ux} y1={length * uy} x2={length * ux} y2={-length * uy} />
        <circle className="lp-h-dot" cx={0} cy={0} r={r * 0.5} />
        <circle className="lp-h-knob" cx={length * ux} cy={-length * uy} r={r} {...drag((x, y) => {
          onHandle(keys.posa!, angle(x, y));
          onHandle(keys.gamma!, Number(Math.min(0.5, Math.hypot(x, y) / (half * 8)).toFixed(4)));
        })} />
        <text className="lp-h-label" style={label} x={length * ux * 1.15} y={-length * uy * 1.15 - half * 0.05} textAnchor="middle">
          γ {fmt(v('gamma'))} · {fmt(posa)}°
        </text>
      </g>
    );
  }
  const cx = v('xpos');
  const cy = v('ypos');
  const q = keys.q ? Math.max(0.001, v('q')) : 1;
  const size = keys.theta_e ? v('theta_e') : half / 4;
  const a = size / Math.sqrt(q);
  const b = size * Math.sqrt(q);
  return (
    <g>
      {keys.posa && (
        <>
          <ellipse className="lp-h-line lp-h-dash" cx={cx} cy={-cy} rx={a} ry={b} transform={`rotate(${-(90 + posa)} ${cx} ${-cy})`} />
          <line className="lp-h-line" x1={cx} y1={-cy} x2={cx + a * ux} y2={-(cy + a * uy)} />
          <circle className="lp-h-knob" cx={cx + a * ux} cy={-(cy + a * uy)} r={r} {...drag((x, y) => {
            onHandle(keys.posa!, Number(angle(x - cx, y - cy).toFixed(2)));
            if (keys.theta_e) onHandle(keys.theta_e, Number(Math.max(0.01, Math.hypot(x - cx, y - cy) * Math.sqrt(q)).toFixed(4)));
          })} />
          {keys.q && (
            <circle className="lp-h-knob lp-h-small" cx={cx + b * uy} cy={-(cy - b * ux)} r={r * 0.8} {...drag((x, y) => {
              const along = Math.abs((x - cx) * uy - (y - cy) * ux);
              onHandle(keys.q!, Number(Math.min(1, Math.max(0.05, (along / size) ** 2)).toFixed(4)));
            })} />
          )}
          <text className="lp-h-label" style={label} x={cx + a * ux * 1.1} y={-(cy + a * uy * 1.1) - half * 0.05} textAnchor="middle">
            {keys.theta_e ? `θ_E ${fmt(size)}″ · ` : ''}
            {fmt(posa)}°{keys.q ? ` · q ${fmt(q)}` : ''}
          </text>
        </>
      )}
      <circle className="lp-h-knob" cx={cx} cy={-cy} r={r} {...drag((x, y) => {
        onHandle(keys.xpos!, Number(x.toFixed(3)));
        onHandle(keys.ypos!, Number(y.toFixed(3)));
      })} />
    </g>
  );
}

function SourcePlane({ view, source, curves, onSource }: { view: LensView; source: Source; curves: boolean; onSource: (source: Source) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const half = view.half / 2;
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const image = context.createImageData(PIXELS, PIXELS);
    for (let j = 0; j < PIXELS; j++) {
      for (let i = 0; i < PIXELS; i++) {
        const x = -half + ((i + 0.5) / PIXELS) * 2 * half;
        const y = half - ((j + 0.5) / PIXELS) * 2 * half;
        const light = Math.exp(-((x - source.x) ** 2 + (y - source.y) ** 2) / (2 * source.size ** 2));
        image.data.set([255 * Math.min(1, light * 1.1), 200 * light, 90 * light, 255], (j * PIXELS + i) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }, [source, half]);
  const move = (event: React.PointerEvent<HTMLElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    onSource({ ...source, x: Number((-half + ((event.clientX - box.left) / box.width) * 2 * half).toFixed(3)), y: Number((half - ((event.clientY - box.top) / box.height) * 2 * half).toFixed(3)) });
  };
  return (
    <div
      className="lp-plane lp-drag"
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        move(event);
      }}
      onPointerMove={(event) => event.buttons & 1 && move(event)}
    >
      <canvas ref={canvas} width={PIXELS} height={PIXELS} aria-label="The source" />
      <svg className="lp-overlay" viewBox={`${-half} ${-half} ${2 * half} ${2 * half}`} preserveAspectRatio="none">
        {curves && view.caustics.map((line, k) => <polyline key={k} className="lp-caustic" points={points(line)} />)}
        <ScaleBar half={half} />
        <circle className="lp-h-knob" cx={source.x} cy={-source.y} r={half * 0.03} />
      </svg>
      <Compass />
    </div>
  );
}

// A bar of 1 arcsec, in the lower left of a plane of a half-width
function ScaleBar({ half }: { half: number }) {
  const x = -half * 0.9;
  const y = half * 0.88;
  return (
    <g>
      <line className="lp-scale" x1={x} y1={y} x2={x + 1} y2={y} />
      <text className="lp-h-label" style={{ fontSize: half * 0.04, strokeWidth: half * 0.008 }} x={x} y={y - half * 0.04}>
        1″
      </text>
    </g>
  );
}

// North up, east left
function Compass() {
  return (
    <svg className="lp-compass" viewBox="-17 -17 34 34" aria-label="North up, east left">
      <path d="M0 9V-10M9 0H-10" />
      <path className="lp-compass-tip" d="M0 -12l-2.5 4h5z" />
      <text x={-3.2} y={-13}>N</text>
      <text x={-17} y={3}>E</text>
    </svg>
  );
}

const points = (line: number[][]) => line.map(([x, y]) => `${x},${-y}`).join(' ');

// A map (n by n, row-major, the first row at the lowest y) between its
// points
function bilinear(values: ArrayLike<number>, n: number, gx: number, gy: number): number {
  const x0 = Math.max(0, Math.min(n - 2, Math.floor(gx)));
  const y0 = Math.max(0, Math.min(n - 2, Math.floor(gy)));
  const fx = Math.min(1, Math.max(0, gx - x0));
  const fy = Math.min(1, Math.max(0, gy - y0));
  const at = (x: number, y: number) => values[y * n + x];
  return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}
