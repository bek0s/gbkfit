// The summary of the result of a fit (fit/result.json; see
// gbkfit.fitting.FitResult): how it went, the statistics of a solution,
// and its free parameters: start and range, best value and error, and the
// 16/50/84th percentiles of the posterior (of a sampler) or of the Laplace
// approximation about the best fit.
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api, data } from '../api/client';
import { format } from '../components/ImageView';

interface Solution {
  free: Record<string, number>;
  chi2: number;
  reduced_chi2: number | null;
  dof: number;
  aic: number | null;
  bic: number | null;
  covariance: { std: Record<string, number> } | null;
  laplace: Record<string, number[]> | null;
}

interface FitResult {
  fitter: { type: string };
  params: { properties: Record<string, unknown> };
  nevals: number;
  time: number;
  success: boolean;
  message: string;
  evidence: { log_z: number | null } | null;
  posterior: Record<string, number[]> | null;
  solutions: Solution[];
}

export function FitSummary({ file }: { file: string }) {
  const result = useQuery({
    queryKey: ['file', file],
    queryFn: async () =>
      JSON.parse(
        await data(api.GET('/api/v1/files/{path}', { params: { path: { path: file } }, parseAs: 'text' })),
      ) as FitResult,
  });
  const [index, setIndex] = useState(0);
  if (result.error) return <p className="error">{result.error.message}</p>;
  if (!result.data) return null;
  const fit = result.data;
  const solution = fit.solutions[Math.min(index, fit.solutions.length - 1)];
  const percentiles = fit.posterior ?? solution.laplace;
  const stats: [string, string][] = [
    ['fitter', fit.fitter.type],
    ['evaluations', String(fit.nevals)],
    ['time', `${fit.time.toFixed(1)} s`],
    ['χ²', format(solution.chi2)],
    ['χ² / dof', solution.reduced_chi2 === null ? '–' : solution.reduced_chi2.toFixed(4)],
    ['AIC', solution.aic === null ? '–' : format(solution.aic)],
    ['BIC', solution.bic === null ? '–' : format(solution.bic)],
  ];
  if (fit.evidence?.log_z != null) stats.push(['log Z', format(fit.evidence.log_z)]);
  return (
    <section className="summary">
      <div className="summary-head">
        <b>Fit</b>
        <span className={fit.success ? 'badge ok' : 'badge bad'}>{fit.success ? 'success' : 'failed'}</span>
        <span className="kind">{fit.message}</span>
        {fit.solutions.length > 1 && (
          <select value={index} onChange={(event) => setIndex(Number(event.target.value))}>
            {fit.solutions.map((_, i) => (
              <option key={i} value={i}>
                solution {i + 1} of {fit.solutions.length}
              </option>
            ))}
          </select>
        )}
      </div>
      <dl className="stats">
        {stats.map(([name, value]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <table className="params-table">
        <thead>
          <tr>
            <th>parameter</th>
            <th>start</th>
            <th>range</th>
            <th>best ± error</th>
            <th colSpan={3}>{fit.posterior ? 'posterior' : 'Laplace'} 16 / 50 / 84 %</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(solution.free).map(([name, best]) => {
            const start = startOf(fit.params.properties, name);
            const std = solution.covariance?.std[name];
            return (
              <tr key={name}>
                <td className="name">{name}</td>
                <td>{start.value === undefined ? '–' : format(start.value)}</td>
                <td>{start.min === undefined ? '–' : `${format(start.min)} – ${format(start.max!)}`}</td>
                <td>{withError(best, std)}</td>
                {(percentiles?.[name] ?? [NaN, NaN, NaN]).map((p, i) => (
                  <td key={i}>{Number.isFinite(p) ? rounded(p, std) : '–'}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

// The start and range of a free element (by its property: the element's
// own, or that of its parameter)
function startOf(
  properties: Record<string, unknown>,
  name: string,
): { value?: number; min?: number; max?: number } {
  const property = properties[name] ?? properties[name.replace(/\[.*$/, '')];
  if (typeof property !== 'object' || property === null) return {};
  const { value, min, max } = property as Record<string, unknown>;
  const number = (x: unknown) => (typeof x === 'number' ? x : undefined);
  return { value: number(value), min: number(min), max: number(max) };
}

// A value and its error, to the decimals of the second significant digit
// of the error
function withError(value: number, error: number | undefined): string {
  if (error === undefined || !(error > 0)) return format(value);
  return `${rounded(value, error)} ± ${rounded(error, error)}`;
}

// A value to the decimals of the second significant digit of an error
function rounded(value: number, error: number | undefined): string {
  if (error === undefined || !(error > 0)) return format(value);
  return value.toFixed(Math.max(0, 1 - Math.floor(Math.log10(error))));
}
