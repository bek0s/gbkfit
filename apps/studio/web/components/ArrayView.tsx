// A view of an array of any dimension: a line of its values (1-D), an
// image (2-D), or an image of a plane along its first axis, chosen with a
// slider (3-D: e.g. the channels of a cube).
import { useState } from 'react';
import { format, ImageView } from './ImageView';

interface Props {
  data: Float32Array;
  shape: number[];
  diverging?: boolean;
}

export function ArrayView({ data, shape, diverging = false }: Props) {
  if (shape.length === 1) return <LineView data={data} />;
  if (shape.length === 2) return <ImageView data={data} shape={shape} diverging={diverging} />;
  return <CubeView data={data} shape={shape} diverging={diverging} />;
}

function CubeView({ data, shape, diverging }: Required<Props>) {
  const [planes, rows, columns] = shape;
  const [plane, setPlane] = useState(Math.floor(planes / 2));
  const shown = Math.min(plane, planes - 1);
  const size = rows * columns;
  return (
    <div className="cube">
      <ImageView
        data={data.subarray(shown * size, (shown + 1) * size)}
        shape={[rows, columns]}
        diverging={diverging}
      />
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
    </div>
  );
}

function LineView({ data }: { data: Float32Array }) {
  const finite = Array.from(data).filter(Number.isFinite);
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  const y = (value: number) => (high > low ? 90 - (80 * (value - low)) / (high - low) : 50);
  const points = Array.from(data)
    .map((value, i) => (Number.isFinite(value) ? `${(100 * i) / Math.max(data.length - 1, 1)},${y(value)}` : null))
    .filter((point) => point !== null)
    .join(' ');
  return (
    <figure className="line">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline points={points} />
      </svg>
      <figcaption>
        <span>{format(low)}</span>
        <span>{format(high)}</span>
      </figcaption>
    </figure>
  );
}
