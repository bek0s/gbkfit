// The free parameters of the live session: a slider for each (over its
// range, min to max), whose values the preview follows.
import { format } from '../components/ImageView';
import { useStudio } from '../studio';

// The steps of a slider over its range
const STEPS = 500;

export function Parameters() {
  const { live } = useStudio();
  const free = Object.entries(live.info?.free ?? {});
  return (
    <aside className="side right">
      <div className="colhead">
        Parameters <span className="sub">{live.watching ? 'best of the fit so far' : `${free.length} free`}</span>
      </div>
      <div className="params">
        {free.map(([name, properties]) => {
          const min = Number(properties.min);
          const max = Number(properties.max);
          const value = live.values[name] ?? Number(properties.value);
          const bounded = Number.isFinite(min) && Number.isFinite(max) && max > min;
          return (
            <label className="param" key={name}>
              <span className="name">{name}</span>
              <span className="num">{format(value)}</span>
              {bounded ? (
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={(max - min) / STEPS}
                  value={value}
                  disabled={live.watching !== null}
                  onChange={(event) => live.set({ [name]: Number(event.target.value) })}
                />
              ) : (
                <input
                  type="number"
                  value={value}
                  disabled={live.watching !== null}
                  onChange={(event) => live.set({ [name]: Number(event.target.value) })}
                />
              )}
            </label>
          );
        })}
        {live.info !== null && free.length === 0 && (
          <p className="hint">The setup has no free parameters.</p>
        )}
      </div>
    </aside>
  );
}
