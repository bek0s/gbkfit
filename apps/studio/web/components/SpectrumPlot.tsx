// A spectrum: its values against the coordinates of its channels along
// its spectral axis (or their numbers, if unknown), with a line at the
// channel shown; a click picks the nearest channel.
import { significant } from './format';

const WIDTH = 600;
const HEIGHT = 160;
const PAD = { left: 8, right: 8, top: 10, bottom: 10 };

export function SpectrumPlot({
  values,
  spectral,
  unit,
  channel,
  onPick,
}: {
  values: (number | null)[];
  spectral: number[] | null;
  unit: string | null;
  channel: number;
  onPick: (channel: number) => void;
}) {
  const xs = spectral ?? values.map((_, i) => i);
  const finite = values.filter((v): v is number => v !== null);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = finite.length ? [Math.min(0, ...finite), Math.max(...finite)] : [0, 1];
  const X = (x: number) => PAD.left + ((x - x0) / (x1 - x0 || 1)) * (WIDTH - PAD.left - PAD.right);
  const Y = (y: number) => HEIGHT - PAD.bottom - ((y - y0) / (y1 - y0 || 1)) * (HEIGHT - PAD.top - PAD.bottom);
  // (a channel without a value breaks the line)
  const lines: string[] = [];
  let line: string[] = [];
  values.forEach((value, i) => {
    if (value === null) {
      if (line.length) lines.push(line.join(' '));
      line = [];
    } else line.push(`${X(xs[i]).toFixed(1)},${Y(value).toFixed(1)}`);
  });
  if (line.length) lines.push(line.join(' '));
  const axis = spectral && unit ? unit : 'channel';
  return (
    <figure className="spectrum">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="The spectrum of the pixel"
        onClick={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const along = ((event.clientX - box.left) / box.width) * WIDTH - PAD.left;
          const x = x0 + (along / (WIDTH - PAD.left - PAD.right)) * (x1 - x0);
          const nearest = xs.reduce((best, value, i) => (Math.abs(value - x) < Math.abs(xs[best] - x) ? i : best), 0);
          onPick(nearest);
        }}
      >
        <line className="zero" x1={PAD.left} x2={WIDTH - PAD.right} y1={Y(0)} y2={Y(0)} />
        <line className="now" x1={X(xs[channel])} x2={X(xs[channel])} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
        {lines.map((points, i) => (
          <polyline key={i} points={points} />
        ))}
      </svg>
      <figcaption>
        <span>
          {significant(x0, 6)} {axis}
        </span>
        <span>
          {significant(y0, 4)} to {significant(y1, 4)}
        </span>
        <span>
          {significant(x1, 6)} {axis}
        </span>
      </figcaption>
    </figure>
  );
}
