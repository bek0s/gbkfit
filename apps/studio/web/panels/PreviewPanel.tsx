// The live preview: the models of the setup of the live session, at the
// values of its free parameters (see Parameters), as they change.
import { ArrayView } from '../components/ArrayView';
import { useStudio } from '../studio';

export function PreviewPanel() {
  const { live } = useStudio();
  return (
    <section className="panel">
      <div className="panel-head">
        <span className="kind">{live.config ?? 'no setup'}</span>
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
      <div className="maps">
        {live.frame?.arrays.map((array, i) => (
          <div className="map" key={i}>
            <div className="map-head">
              {String(array.info.observation)} · {String(array.info.item)}
            </div>
            <ArrayView data={array.data} shape={array.info.shape} />
          </div>
        ))}
      </div>
    </section>
  );
}
