// A run of a fit as it runs: the reports of its progress (its product
// logs/progress.jsonl, read again every few seconds; see the library's
// FitProgress): what it is doing, the evaluations and the time so far,
// the best chi-square (or cost) so far against the evaluations, and the
// free values at it.
import { useMemo, useState } from 'react';
import { useFileText } from '../../api/queries';
import { significant } from '../../components/format';

interface Report {
  stage: string;
  nevals: number;
  elapsed: number;
  merit: string;
  best_chi2: number;
  best: Record<string, number> | null;
  final: boolean;
}

export function LiveView({ path, version }: { path: string | null; version: string }) {
  const text = useFileText(path, 2000, version);
  const reports = useMemo(
    () =>
      (text.data ?? '')
        .split('\n')
        .filter((line) => line.trim())
        .flatMap((line) => {
          try {
            return [JSON.parse(line) as Report];
          } catch {
            return [];
          }
        }),
    [text.data],
  );
  const last = reports.at(-1);
  if (!path || !last) return <p className="ob-text">The fit reports its progress as it evaluates its first models…</p>;
  const merit = last.merit === 'chi2' ? 'χ²' : 'cost';
  return (
    <div className="fx-view-grid">
      <div className="fx-card fx-merits">
        <div>
          <span className="fx-overline">Doing</span>
          <b>{last.stage}</b>
        </div>
        <div>
          <span className="fx-overline">Models</span>
          <b>{last.nevals.toLocaleString()}</b>
        </div>
        <div>
          <span className="fx-overline">Time</span>
          <b>{duration(last.elapsed)}</b>
        </div>
        <div>
          <span className="fx-overline">Best {merit}</span>
          <b>{Number.isFinite(last.best_chi2) ? last.best_chi2.toPrecision(7) : '—'}</b>
        </div>
      </div>
      <div className="fx-live-two">
        <div className="fx-card">
          <div className="fx-card-head">
            <h2>Best {merit}</h2>
            <span className="fx-faint fx-small">against the models evaluated</span>
          </div>
          <Trace reports={reports} merit={merit} />
        </div>
        <div className="fx-card">
          <div className="fx-card-head">
            <h2>Best so far</h2>
            <span className="fx-faint fx-small">the free values at the best {merit}</span>
          </div>
          {last.best ? (
            <table className="fx-best">
              <tbody>
                {Object.entries(last.best).map(([name, value]) => (
                  <tr key={name}>
                    <td className="fx-name">{name}</td>
                    <td className="fx-num">{significant(value, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="ob-text fx-pad">None yet: no model has a finite {merit}.</p>
          )}
        </div>
      </div>
    </div>
  );
}

// The best chi-square against the models evaluated: a line, its ends
// labelled, and the report under the pointer
function Trace({ reports, merit }: { reports: Report[]; merit: string }) {
  const [hover, setHover] = useState<Report | null>(null);
  const points = reports.filter((r) => Number.isFinite(r.best_chi2));
  if (points.length < 2) return <p className="ob-text fx-pad">The line shows from the second report.</p>;
  const [w, h, left, bottom] = [520, 200, 64, 24];
  const xs = points.map((r) => r.nevals);
  const ys = points.map((r) => r.best_chi2);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs) || 1];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x: number) => left + ((x - x0) / (x1 - x0 || 1)) * (w - left - 8);
  const sy = (y: number) => 8 + (1 - (y - y0) / (y1 - y0 || 1)) * (h - bottom - 8);
  const line = points.map((r) => `${sx(r.nevals).toFixed(1)},${sy(r.best_chi2).toFixed(1)}`).join(' ');
  // (as many digits as tell its ends apart)
  let digits = 5;
  while (digits < 12 && y0 !== y1 && y0.toPrecision(digits) === y1.toPrecision(digits)) digits++;
  const pick = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const x = x0 + (((event.clientX - box.left) / box.width) * w - left) / ((w - left - 8) / (x1 - x0 || 1));
    setHover(points.reduce((a, b) => (Math.abs(b.nevals - x) < Math.abs(a.nevals - x) ? b : a)));
  };
  return (
    <div className="fx-trace">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`The best ${merit} against the models evaluated`} onMouseMove={pick} onMouseLeave={() => setHover(null)}>
        <line x1={left} y1={h - bottom} x2={w - 8} y2={h - bottom} className="fx-trace-axis" />
        <line x1={left} y1={8} x2={left} y2={h - bottom} className="fx-trace-axis" />
        <text x={left - 6} y={sy(y1) + 4} textAnchor="end">
          {y1.toPrecision(digits)}
        </text>
        <text x={left - 6} y={sy(y0) + 4} textAnchor="end">
          {y0.toPrecision(digits)}
        </text>
        <text x={left} y={h - 6}>
          {x0}
        </text>
        <text x={w - 8} y={h - 6} textAnchor="end">
          {x1} models
        </text>
        <polyline points={line} className="fx-trace-line" />
        {hover && <circle cx={sx(hover.nevals)} cy={sy(hover.best_chi2)} r={4} className="fx-trace-dot" />}
      </svg>
      <span className="fx-trace-read">{hover ? `${hover.nevals} models · ${duration(hover.elapsed)} · best ${merit} ${hover.best_chi2.toPrecision(7)}` : ' '}</span>
    </div>
  );
}

function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}
