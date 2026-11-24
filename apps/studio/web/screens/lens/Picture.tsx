// What a lens makes of a source behind it (see the server's LensViewInfo):
// the image plane, as the images of a round test source (each pixel shot
// back to the source plane through the deflection of the lens: beta =
// theta - alpha), or as the convergence or the magnification of the lens
// (of the Jacobian of its deflection, by finite differences), with its
// critical curves and the handles of the piece selected (dragged, they set
// its parameters); and the source plane, with its caustics and the source,
// which a drag moves (not saved). Each plane has a scale bar and the
// compass of the sky (north up, east left; +x is west). The field is that
// of the view (see LensFrame), which its bar zooms by round steps or fits
// to the lens again.
import { useEffect, useMemo, useRef, useState } from 'react';
import type { LensFrame, LensView } from '../../api/queries';
import { fmt } from '../../components/format';
import { Track } from '../../components/ui/rows';
import { snap } from '../../components/values';
import type { Dots, Handles } from './LensScreen';

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

// The largest test source, in half-widths of the field
const LARGEST_SOURCE = 0.13;

// The round fields (their width, arcsec, times a power of ten) a zoom
// steps through: those the server fits the field to (see its lenses)
const ROUND_FIELDS = [1, 1.5, 2, 3, 4, 6, 8];

// The round field next to a field, wider or narrower
export function zoomed(field: number, wider: boolean): number {
  const power = 10 ** Math.floor(Math.log10(field));
  const rounds = [power / 10, power, power * 10].flatMap((p) => ROUND_FIELDS.map((r) => Number((r * p).toPrecision(2))));
  const near = field * 1e-9;
  return wider ? rounds.find((r) => r > field + near)! : [...rounds].reverse().find((r) => r < field - near)!;
}

// What the picture says of what it shows, if not the lens as it is (e.g.
// the last picture, of a lens with a problem); bad, if it is a problem
export type PictureNote = { text: string; bad: boolean };

// A view of no lens (no piece shown): the source as it is, over a field
// (by default, 4 arcsec about the origin)
export function bareView(frame: LensFrame | null): LensView {
  const { half, centre } = frame ?? { half: 2, centre: [0, 0] as [number, number] };
  return { half, centre, source_redshift: null, size: 2, alpha_x: [0, 0, 0, 0], alpha_y: [0, 0, 0, 0], critical: [], caustics: [], left_out: [] };
}

export function LensPicture({
  view,
  note,
  onZoom,
  onFit,
  source,
  onSource,
  handles,
  dots,
  onPick,
  onClearPick,
  onHandle,
}: {
  // What it draws (none: none yet), and what it says of it
  view: LensView | undefined;
  note: PictureNote | null;
  // What the zoom does and what fits the field to the lens again
  onZoom: (wider: boolean) => void;
  onFit: () => void;
  source: Source;
  onSource: (source: Source) => void;
  handles: Handles;
  // The centres of the other pieces, and what a click on one does; what
  // Esc does (the piece no longer selected)
  dots: Dots;
  onPick: (piece: string) => void;
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
  // (nothing drawn yet: why, or a wait)
  if (!view) return <div className="ob-view lp-view">{note ? <p className="ob-empty">{note.text}</p> : <div className="gx-map-wait" />}</div>;
  const largest = LARGEST_SOURCE * view.half;
  return (
    <div className="ob-view lp-view">
      <div className="lp-tools">
        <span className="lp-grp lp-grp-even" role="tablist" aria-label="What the image plane shows">
          {MODES.map(([key, title]) => (
            <button key={key} type="button" role="tab" aria-selected={mode === key} onClick={() => setMode(key)}>
              {title}
            </button>
          ))}
        </span>
        <span className="lp-grp lp-grp-even">
          <button type="button" aria-pressed={curves} title="The critical curves and the caustics" onClick={() => setCurves(!curves)}>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path className="lp-ic-critical" d="M2 11c2-6 10-6 12 0" />
              <path className="lp-ic-caustic" d="M5 13c1.5-2 4.5-2 6 0" />
            </svg>
            Curves
          </button>
          <button type="button" aria-pressed={showHandles} title="The handles of the selected piece" onClick={() => setShowHandles(!showHandles)}>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <circle cx="8" cy="8" r="5.5" />
              <circle className="lp-ic-fill" cx="8" cy="8" r="1.4" />
              <circle className="lp-ic-fill" cx="13.5" cy="8" r="1.4" />
            </svg>
            Handles
          </button>
        </span>
        <span className="lp-field">
          Field
          <span className="lp-grp">
            <button type="button" className="lp-grp-icon" title="Zoom out" aria-label="Zoom out" onClick={() => onZoom(true)}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8" /></svg>
            </button>
            <span className="lp-grp-val">{roundText(2 * view.half)}″</span>
            <button type="button" className="lp-grp-icon" title="Zoom in" aria-label="Zoom in" onClick={() => onZoom(false)}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8M8 4v8" /></svg>
            </button>
            <button type="button" className="lp-grp-icon" title="Fit the field to the lens" aria-label="Fit the field to the lens" onClick={onFit}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" /></svg>
            </button>
          </span>
        </span>
      </div>
      <div className="lp-planes">
        <figure className="lp-plane-box lp-image-box">
          <figcaption>
            Image plane
            {note ? (
              <span className="lp-key" data-bad={note.bad || undefined} title={note.text}>
                {note.text}
              </span>
            ) : (
              <span className="lp-key" data-off={!curves || view.critical.length === 0 || undefined}><i className="lp-key-critical" /> critical curves</span>
            )}
          </figcaption>
          <ImagePlane view={view} source={source} mode={mode} curves={curves} handles={showHandles ? handles : null} dots={showHandles ? dots : []} onPick={onPick} onHandle={onHandle} />
        </figure>
        <figure className="lp-plane-box lp-source-box">
          <figcaption>
            Source plane
            <span className="lp-key" data-off={!curves || view.caustics.length === 0 || undefined}><i className="lp-key-caustic" /> caustics</span>
          </figcaption>
          <SourcePlane view={view} source={source} curves={curves} onSource={onSource} />
          <div className="lp-src">
            <span><span className="lp-src-name">Test source </span>σ</span>
            <Track at={Math.min(1, source.size / largest)} onAt={(t) => onSource({ ...source, size: Math.max(largest / 50, snap(t * largest, [0, largest])) })} />
            <span className="lp-src-v">{fmt(source.size)}″</span>
          </div>
        </figure>
      </div>
    </div>
  );
}

// A round length (e.g. of a field), as few digits as it has
export function roundText(length: number): string {
  return String(Number(length.toPrecision(3)));
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
  onPick: (piece: string) => void;
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
    const [cx, cy] = view.centre;
    const image = context.createImageData(PIXELS, PIXELS);
    for (let j = 0; j < PIXELS; j++) {
      for (let i = 0; i < PIXELS; i++) {
        const x = cx - h + ((i + 0.5) / PIXELS) * 2 * h;
        const y = cy + h - ((j + 0.5) / PIXELS) * 2 * h;
        const gx = ((x - cx + h) / (2 * h)) * (n - 1);
        const gy = ((y - cy + h) / (2 * h)) * (n - 1);
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
  const [cx, cy] = view.centre;
  const world = (event: React.PointerEvent<Element>, box: DOMRect): [number, number] => [
    cx - h + ((event.clientX - box.left) / box.width) * 2 * h,
    cy + h - ((event.clientY - box.top) / box.height) * 2 * h,
  ];
  return (
    <div
      className="lp-plane"
      onPointerMove={(event) => setPointer(world(event, event.currentTarget.getBoundingClientRect()))}
      onPointerLeave={() => setPointer(null)}
    >
      <canvas ref={canvas} width={PIXELS} height={PIXELS} aria-label="What the lens makes of the source" />
      <svg className="lp-overlay" viewBox={`${cx - h} ${-cy - h} ${2 * h} ${2 * h}`} preserveAspectRatio="none">
        {curves && view.critical.map((line, k) => <polyline key={k} className="lp-critical" points={points(line)} />)}
        {dots.map((dot) => (
          <circle key={dot.piece} className="lp-h-pick" cx={dot.x} cy={-dot.y} r={h * 0.018} style={{ fill: dot.colour }} onClick={() => onPick(dot.piece)}>
            <title>Select the piece</title>
          </circle>
        ))}
        {handles && <HandleShapes handles={handles} half={h} onHandle={onHandle} world={world} />}
      </svg>
      <ScaleBar length={barLength(view.half)} half={h} />
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
// parameters it shows, to the values under the pointer (which the screen
// settles and rounds as their rows do)
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
          onHandle(keys.gamma!, Math.hypot(x, y) / (half * 8));
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
            onHandle(keys.posa!, angle(x - cx, y - cy));
            if (keys.theta_e) onHandle(keys.theta_e, Math.hypot(x - cx, y - cy) * Math.sqrt(q));
          })} />
          {keys.q && (
            <circle className="lp-h-knob" cx={cx + b * uy} cy={-(cy - b * ux)} r={r * 0.8} {...drag((x, y) => {
              const along = Math.abs((x - cx) * uy - (y - cy) * ux);
              onHandle(keys.q!, (along / size) ** 2);
            })} />
          )}
          <text className="lp-h-label" style={label} x={cx + a * ux * 1.1} y={-(cy + a * uy * 1.1) - half * 0.05} textAnchor="middle">
            {keys.theta_e ? `θ_E ${fmt(size)}″ · ` : ''}
            {fmt(posa)}°{keys.q ? ` · q ${fmt(q)}` : ''}
          </text>
        </>
      )}
      <circle className="lp-h-knob" cx={cx} cy={-cy} r={r} {...drag((x, y) => {
        onHandle(keys.xpos!, x);
        onHandle(keys.ypos!, y);
      })} />
    </g>
  );
}

function SourcePlane({ view, source, curves, onSource }: { view: LensView; source: Source; curves: boolean; onSource: (source: Source) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const half = view.half / 2;
  const [cx, cy] = view.centre;
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const image = context.createImageData(PIXELS, PIXELS);
    for (let j = 0; j < PIXELS; j++) {
      for (let i = 0; i < PIXELS; i++) {
        const x = cx - half + ((i + 0.5) / PIXELS) * 2 * half;
        const y = cy + half - ((j + 0.5) / PIXELS) * 2 * half;
        const light = Math.exp(-((x - source.x) ** 2 + (y - source.y) ** 2) / (2 * source.size ** 2));
        image.data.set([255 * Math.min(1, light * 1.1), 200 * light, 90 * light, 255], (j * PIXELS + i) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }, [source, half, cx, cy]);
  const move = (event: React.PointerEvent<HTMLElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const span: [number, number] = [0, 2 * half];
    onSource({ ...source, x: snap(cx - half + ((event.clientX - box.left) / box.width) * 2 * half, span), y: snap(cy + half - ((event.clientY - box.top) / box.height) * 2 * half, span) });
  };
  return (
    <div
      className="lp-plane lp-drag"
      title="Drag to move the test source (only for the picture, not saved)"
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        move(event);
      }}
      onPointerMove={(event) => event.buttons & 1 && move(event)}
    >
      <canvas ref={canvas} width={PIXELS} height={PIXELS} aria-label="The source" />
      <svg className="lp-overlay" viewBox={`${cx - half} ${-cy - half} ${2 * half} ${2 * half}`} preserveAspectRatio="none">
        {curves && view.caustics.map((line, k) => <polyline key={k} className="lp-caustic" points={points(line)} />)}
        <circle className="lp-h-knob" cx={source.x} cy={-source.y} r={half * 0.03} />
      </svg>
      <ScaleBar length={barLength(view.half)} half={half} />
      <Compass />
      <span className="lp-readout lp-readout-corner">
        x {fmt(source.x)}″ · y {fmt(source.y)}″
      </span>
    </div>
  );
}

// The length of the scale bars (arcsec): a round one (1, 2 or 5 times a
// power of ten) of at most a fifth of the image plane, of a half-width;
// the same in the source plane, so that the planes compare
export function barLength(half: number): number {
  const most = (2 * half) / 5;
  const power = 10 ** Math.floor(Math.log10(most));
  return [5, 2, 1].map((r) => r * power).find((r) => r <= most * (1 + 1e-9))!;
}

// A bar of a length, in the lower left of a plane of a half-width (its
// label of the same size in every plane)
export function ScaleBar({ length, half }: { length: number; half: number }) {
  return (
    <span className="lp-scale" style={{ width: `${(length / (2 * half)) * 100}%` }}>
      {roundText(length)}″
    </span>
  );
}

// North up, east left: two arms from a corner, as sky charts mark them
export function Compass() {
  return (
    <svg className="lp-compass" viewBox="0 0 40 40" aria-label="North up, east left">
      <path d="M32 32V14M32 32H14" />
      <text x={32} y={10} textAnchor="middle">N</text>
      <text x={7} y={32} textAnchor="middle" dominantBaseline="central">E</text>
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
