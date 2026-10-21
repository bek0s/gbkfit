// The figures of a run, which the plot task makes in a run of its own in
// <run>/plots (see gbkfit.tasks.plot): images shown, PDFs as links.
import { useQuery } from '@tanstack/react-query';
import { api, ApiError, data, tokenUrl } from '../api/client';

export function Figures({ run }: { run: string }) {
  const path = `${run}/plots`;
  const plots = useQuery({
    queryKey: ['run', path],
    queryFn: () => data(api.GET('/api/v1/runs/{path}', { params: { path: { path } } })),
    // (most runs have no figures)
    retry: (_, error) => !(error instanceof ApiError && error.status === 404),
  });
  const figures = (plots.data?.products ?? []).filter((p) => p.kind === 'figure');
  if (figures.length === 0) return null;
  return (
    <section className="figures">
      <div className="map-head">Figures</div>
      <div className="figure-grid">
        {figures.map((figure) => {
          const url = tokenUrl(`/api/v1/raw/${path}/${String(figure.path)}`);
          const title = String(figure.title ?? figure.path);
          return figure.format === 'pdf' ? (
            <a key={url} className="link" href={url} target="_blank" rel="noreferrer">
              {title} (PDF)
            </a>
          ) : (
            <figure key={url} className="figure">
              <img src={url} alt={title} />
              <figcaption>{title}</figcaption>
            </figure>
          );
        })}
      </div>
    </section>
  );
}
