// Arrays of one data item side by side (e.g. its data, model and residual):
// those not diverging share the range of their colours, and the planes of
// cubes are chosen together.
import { useMemo, useState } from 'react';
import type { FrameArray } from '../api/frames';
import { ImageView, range } from './ImageView';
import { ArrayView } from './ArrayView';

export interface View {
  label: string;
  array: FrameArray;
  diverging: boolean;
}

export function Comparison({ views }: { views: View[] }) {
  const shape = views[0].array.info.shape;
  const shared = useMemo(() => {
    const ranges = views.filter((v) => !v.diverging).map((v) => range(v.array.data, false));
    const finite = ranges.filter(([low, high]) => low <= high && (low !== 0 || high !== 0));
    return finite.length === 0
      ? undefined
      : ([Math.min(...finite.map((r) => r[0])), Math.max(...finite.map((r) => r[1]))] as [number, number]);
  }, [views]);
  const [plane, setPlane] = useState(Math.floor(shape[0] / 2));

  if (shape.length !== 3) {
    return (
      <div className="comparison">
        {views.map((view) => (
          <div className="view" key={view.label}>
            <div className="view-label">{view.label}</div>
            <ArrayView
              data={view.array.data}
              shape={view.array.info.shape}
              diverging={view.diverging}
              limits={view.diverging ? undefined : shared}
            />
          </div>
        ))}
      </div>
    );
  }
  const [planes, rows, columns] = shape;
  const shown = Math.min(plane, planes - 1);
  const size = rows * columns;
  return (
    <>
      <div className="comparison">
        {views.map((view) => (
          <div className="view" key={view.label}>
            <div className="view-label">{view.label}</div>
            <ImageView
              data={view.array.data.subarray(shown * size, (shown + 1) * size)}
              shape={[rows, columns]}
              diverging={view.diverging}
              limits={view.diverging ? range(view.array.data, true) : shared}
            />
          </div>
        ))}
      </div>
      <label className="plane">
        <span>plane</span>
        <input
          type="range"
          min={0}
          max={planes - 1}
          value={shown}
          onChange={(event) => setPlane(Number(event.target.value))}
        />
        <span className="num">
          {shown + 1} / {planes}
        </span>
      </label>
    </>
  );
}
