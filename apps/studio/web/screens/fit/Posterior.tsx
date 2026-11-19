// The posterior of a run of a sampler (its product fit/posterior.ecsv: a
// table of samples of every parameter): a corner plot of its free
// parameters, the histogram of each and the density of each pair, over
// the middle 99.4% of their samples; the median and the 68% interval of
// each under its histogram.
import { useElementSize } from '@mantine/hooks';
import { useMemo } from 'react';
import { useFileText } from '../../api/queries';
import { rampColour, rampStops, useColourMap, useScheme } from '../../components/colours';

export function PosteriorCard({ path, free }: { path: string; free: string[] }) {
  const text = useFileText(path);
  const table = useMemo(() => (text.data ? parseTable(text.data) : null), [text.data]);
  const names = free.filter((name) => table?.columns[name] !== undefined);
  return (
    <div className="fx-card">
      <div className="fx-card-head">
        <h2>Posterior</h2>
        <span className="fx-faint fx-small">{table ? `${table.rows} samples` : 'reading…'}</span>
      </div>
      {text.error && <p className="error fx-pad">{text.error.message}</p>}
      {table && names.length > 0 && <Corner names={names} units={names.map((name) => table.units[name] ?? '')} samples={names.map((name) => table.columns[name])} />}
    </div>
  );
}

// An ECSV table of numbers: its columns by name, their units, and the
// number of its rows
function parseTable(text: string) {
  const units: Record<string, string> = {};
  const rows: number[][] = [];
  let names: string[] | null = null;
  for (const line of text.split('\n')) {
    const header = /^# - \{name: ([^,]+), unit: ([^,]+),/.exec(line);
    if (header) units[header[1]] = header[2];
    if (line.startsWith('#') || !line.trim()) continue;
    const parts = line.trim().split(/\s+/);
    if (names === null) names = parts;
    else rows.push(parts.map(Number));
  }
  const columns: Record<string, Float64Array> = {};
  (names ?? []).forEach((name, k) => (columns[name] = Float64Array.from(rows, (row) => row[k])));
  return { columns, units, rows: rows.length };
}

const BINS = 28;

function Corner({ names, units, samples }: { names: string[]; units: string[]; samples: Float64Array[] }) {
  const { ref, width } = useElementSize();
  useScheme();
  useColourMap();
  const n = names.length;
  const [gap, left, bottom] = [8, 36, 52];
  const cell = Math.max(0, Math.min(220, Math.floor((width - left - gap * (n - 1)) / n)));
  const sorted = samples.map((s) => Float64Array.from(s).sort());
  const at = (k: number, q: number) => sorted[k][Math.min(sorted[k].length - 1, Math.floor(sorted[k].length * q))];
  const ranges = sorted.map((_, k) => [at(k, 0.003), at(k, 0.997)] as [number, number]);
  const bin = (value: number, k: number) => Math.floor(((value - ranges[k][0]) / (ranges[k][1] - ranges[k][0] || 1)) * BINS);
  const stops = rampStops(false);
  const cells = [];
  for (let row = 0; row < n; row++) {
    for (let col = 0; col <= row; col++) {
      const [x, y] = [left + col * (cell + gap), row * (cell + gap)];
      if (row === col) {
        const counts = new Array(BINS).fill(0);
        samples[col].forEach((v) => {
          const b = bin(v, col);
          if (b >= 0 && b < BINS) counts[b]++;
        });
        const top = Math.max(...counts, 1);
        cells.push(
          <g key={`${row}-${col}`}>
            <rect x={x} y={y} width={cell} height={cell} className="fx-cell" />
            {counts.map((c, b) => (
              <rect key={b} x={x + (b * cell) / BINS + 0.5} y={y + cell - (c / top) * (cell - 6)} width={Math.max(0, cell / BINS - 1)} height={(c / top) * (cell - 6)} className="fx-bar" />
            ))}
          </g>,
        );
      } else {
        const counts = new Array(BINS * BINS).fill(0);
        for (let s = 0; s < samples[0].length; s++) {
          const [bx, by] = [bin(samples[col][s], col), bin(samples[row][s], row)];
          if (bx >= 0 && bx < BINS && by >= 0 && by < BINS) counts[by * BINS + bx]++;
        }
        const top = Math.max(...counts, 1);
        cells.push(
          <g key={`${row}-${col}`}>
            <rect x={x} y={y} width={cell} height={cell} className="fx-cell" />
            {counts.map((c, k) =>
              c === 0 ? null : (
                <rect key={k} x={x + ((k % BINS) * cell) / BINS} y={y + cell - (Math.floor(k / BINS) + 1) * (cell / BINS)} width={cell / BINS + 0.4} height={cell / BINS + 0.4} fill={`rgb(${rampColour(stops, 0.15 + 0.85 * Math.sqrt(c / top)).join(',')})`} />
              ),
            )}
          </g>,
        );
      }
    }
  }
  const size = n * cell + (n - 1) * gap;
  const short = (v: number) => String(Number(v.toPrecision(4)));
  return (
    <div className="fx-corner" ref={ref}>
      {cell > 0 && (
        <svg width={left + size} height={size + bottom} role="img" aria-label="The posterior: each free parameter, and each pair">
          {cells}
          {names.map((name, k) => {
            const [low, mid, high] = [at(k, 0.16), at(k, 0.5), at(k, 0.84)];
            return (
              <g key={name}>
                <text x={left + k * (cell + gap) + cell / 2} y={size + 16} textAnchor="middle">
                  {name}
                  {units[k] ? ` (${units[k]})` : ''}
                </text>
                <text x={left + k * (cell + gap) + cell / 2} y={size + 32} textAnchor="middle" className="fx-corner-value">
                  {short(mid)} −{short(mid - low)} +{short(high - mid)}
                </text>
              </g>
            );
          })}
          {names.map((name, k) =>
            k === 0 ? null : (
              <text key={name} x={12} y={k * (cell + gap) + cell / 2} textAnchor="middle" transform={`rotate(-90 12 ${k * (cell + gap) + cell / 2})`}>
                {name}
              </text>
            ),
          )}
        </svg>
      )}
    </div>
  );
}
