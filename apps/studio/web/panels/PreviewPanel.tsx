// The live preview: for each data item (and each image of it, e.g. the
// moment maps of a cube) of the setup of the live session, its data, its
// model and its residual ((model - data) / error) side by side, at the
// values of the free parameters (see Parameters), as they change, or at
// the best values of a fit it follows (see useFitWatch); and the
// chi-square.
import { useProgress } from '../api/events';
import type { Frame, FrameArray } from '../api/frames';
import { Comparison, type View } from '../components/Comparison';
import { format } from '../components/ImageView';
import { useStudio } from '../studio';

// The columns of a data item, by the kinds of their arrays (those of its
// images end in _image)
const COLUMNS = ['data', 'model', 'residual'];

export function PreviewPanel() {
  const { live } = useStudio();
  const chi2 = live.frame?.header.chi2;
  const last = useProgress(live.watching ?? '').at(-1);
  return (
    <section className="panel">
      <div className="panel-head">
        <span className="kind">{live.config ?? 'no setup'}</span>
        {typeof chi2 === 'number' && <span className="badge">χ² {format(chi2)}</span>}
        {live.watching && (
          <span className="badge">
            following the fit{last ? ` · ${last.stage} · ${last.nevals} evaluations` : ''}
            {last?.final ? ' · done' : ''}
          </span>
        )}
        <div className="actions">
          {live.config !== null && (
            <button className="ghost" onClick={live.stop}>
              Stop
            </button>
          )}
        </div>
      </div>
      {live.config === null && <p className="hint">Open a setup, and press Preview.</p>}
      {live.starting && <p className="hint">Setting the model up…</p>}
      {live.error && <p className="error">{live.error}</p>}
      <div className="rows">
        {rows(live.data, live.frame).map(({ key, title, views }) => (
          <div className="row-item" key={key}>
            <div className="map-head">{title}</div>
            <Comparison views={views} />
          </div>
        ))}
      </div>
    </section>
  );
}

interface Row {
  key: string;
  title: string;
  views: View[];
}

// The arrays of the data and of the result, by data item or image
function rows(data: Frame | null, result: Frame | null): Row[] {
  const byKey = new Map<string, Row>();
  for (const array of [...(data?.arrays ?? []), ...(result?.arrays ?? [])]) {
    const [kind, observation, item, image, title] = ['kind', 'observation', 'item', 'image', 'title'].map(
      (name) => text(array, name),
    );
    const column = kind?.replace(/_image$/, '') ?? '';
    if (!COLUMNS.includes(column)) continue;
    const key = `${observation} · ${item ?? image}`;
    if (!byKey.has(key)) {
      byKey.set(key, { key, title: image === undefined ? key : `${observation} · ${title ?? image}`, views: [] });
    }
    byKey.get(key)!.views.push({ label: column, array, diverging: column === 'residual' });
  }
  for (const row of byKey.values()) {
    row.views.sort((a, b) => COLUMNS.indexOf(a.label) - COLUMNS.indexOf(b.label));
  }
  return [...byKey.values()].filter((row) => row.views.some((v) => v.label === 'model'));
}

// A text of the header of an array (e.g. its kind), if it has it
function text(array: FrameArray, name: string): string | undefined {
  const value = array.info[name];
  return typeof value === 'string' ? value : undefined;
}
