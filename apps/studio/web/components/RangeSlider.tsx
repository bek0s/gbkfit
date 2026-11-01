// A slider of one value, or of a range (a low and a high handle), from
// min to max: a handle at its end means no value there (null), as an
// open range. A handle is dragged; a press on the track moves the nearer
// handle there. Its ends are labelled, and so is a mark (e.g. 0).
import { useRef } from 'react';

export function RangeSlider({
  min,
  max,
  step,
  values,
  onChange,
  format,
  mark,
  label,
}: {
  min: number;
  max: number;
  step: number;
  // The low and the high value (null: open), or one value alone
  values: [number | null, number | null] | [number | null];
  onChange: (values: (number | null)[]) => void;
  // How the values on the slider are labelled
  format: (value: number) => string;
  mark?: number;
  label: string;
}) {
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef<number | null>(null);
  const range = values.length === 2;
  // Where each handle is (at its end when open)
  const at = values.map((value, i) => value ?? (range && i === 1 ? max : min));
  const fraction = (value: number) => (Math.min(max, Math.max(min, value)) - min) / (max - min);
  const valueAt = (clientX: number) => {
    const box = track.current!.getBoundingClientRect();
    const raw = min + ((clientX - box.left) / box.width) * (max - min);
    return Math.min(max, Math.max(min, Math.round(raw / step) * step));
  };
  // A handle moved to a value: at its end (or past the other), open
  const move = (handle: number, value: number) => {
    const next = [...values] as (number | null)[];
    if (range && handle === 0) next[0] = value <= min ? null : Math.min(value, at[1]);
    else if (range) next[1] = value >= max ? null : Math.max(value, at[0]);
    else next[0] = value <= min ? null : value;
    onChange(next);
  };
  const [low, high] = range ? [fraction(at[0]), fraction(at[1])] : [0, fraction(at[0])];
  return (
    <div className="range-slider" aria-label={label}>
      <div
        ref={track}
        className="range-track"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          const value = valueAt(event.clientX);
          dragging.current = range && Math.abs(value - at[0]) < Math.abs(value - at[1]) ? 0 : range ? 1 : 0;
          move(dragging.current, value);
        }}
        onPointerMove={(event) => {
          if (dragging.current !== null) move(dragging.current, valueAt(event.clientX));
        }}
        onPointerUp={(event) => {
          // (the value where the pointer is let go: the browser may merge
          // the last moves into the release)
          if (dragging.current !== null) move(dragging.current, valueAt(event.clientX));
          dragging.current = null;
        }}
        onPointerCancel={() => (dragging.current = null)}
      >
        <div className="range-fill" style={{ left: `${100 * low}%`, width: `${100 * (high - low)}%` }} />
        {mark !== undefined && <div className="range-mark" style={{ left: `${100 * fraction(mark)}%` }} />}
        {at.map((value, i) => (
          <div
            key={i}
            className="range-thumb"
            data-open={values[i] === null}
            style={{ left: `${100 * fraction(value)}%` }}
            role="slider"
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={value}
            aria-label={range ? `${label}: ${i === 0 ? 'the low end' : 'the high end'}` : label}
            tabIndex={0}
            onKeyDown={(event) => {
              const by = event.key === 'ArrowRight' || event.key === 'ArrowUp' ? step : event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? -step : 0;
              if (by !== 0) {
                event.preventDefault();
                move(i, value + by);
              }
            }}
          />
        ))}
      </div>
      <div className="range-labels">
        <span>{range ? 'no limit' : 'none'}</span>
        {mark !== undefined && <span style={{ left: `${100 * fraction(mark)}%` }}>{format(mark)}</span>}
        <span>{range ? 'no limit' : format(max)}</span>
      </div>
    </div>
  );
}
