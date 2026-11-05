// A parameter of a galaxy, to set: its value (a slider, and a field to type
// it), and whether a fit frees it (in a range) or keeps it fixed. One of
// several values (e.g. one per ring) has its profile, whose points are
// dragged (or drawn across, as with a pen), and a slider that sets all
// its values at once. One tied to an expression shows it; one set
// otherwise in its params block (e.g. by element) is left as it is.
import { RangeSlider } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { useRef, useState } from 'react';
import { significant } from '../../components/format';
import { StepSlider } from '../../components/ui/sliders';
import { elementValues, sliderReach, validRange, type Param, type Setting } from './params';

export interface ParamFieldProps {
  param: Param;
  label: string;
  setting: Setting;
  // The half-width of the field of the view (arcsec), for the reach of
  // lengths and positions
  field: number;
  // The radii of the rings of its component, for the profile of one value
  // per ring
  rings: number[] | null;
  onValues: (values: number[], gesture: string) => void;
  onFree: (free: boolean) => void;
  onRange: (min: number, max: number) => void;
  onUntie: () => void;
}

// The units as they are written here
export function unitText(unit: string | null): string {
  if (unit === null) return '';
  return { 'km / s': 'km/s', deg: '°', arcsec: '″', 'km / (Mpc s)': 'km/s/Mpc' }[unit] ?? unit.replaceAll(' ', '');
}

// (profiles of more values than this have only the slider of all)
const MOST_POINTS = 64;

export function ParamField(props: ParamFieldProps) {
  const { param, label, setting, field, onFree, onUntie } = props;
  const unit = unitText(param.unit);
  const head = (
    <div className="gx-param-head">
      <span className="gx-param-name" title={[param.desc, `${param.name}${param.range ? ` · valid ${param.range}` : ''}`].filter(Boolean).join('\n')}>
        {param.title ? (
          <>
            <span className="gx-param-title">{param.title}</span>
            <code>{label}</code>
          </>
        ) : (
          <code className="gx-param-symbol">{label}</code>
        )}
        {param.size > 1 && <small>{props.rings?.length === param.size ? 'per ring' : `${param.size} values`}</small>}
      </span>
      {(setting.kind === 'fixed' || setting.kind === 'free') && (
        <button
          className="gx-free"
          data-on={setting.kind === 'free' || undefined}
          onClick={() => onFree(setting.kind !== 'free')}
          title={setting.kind === 'free' ? 'Free in a fit, in its range: click to fix it' : 'Fixed in a fit: click to free it'}
        >
          {setting.kind === 'free' ? 'Free' : 'Fixed'}
        </button>
      )}
    </div>
  );
  if (setting.kind === 'tied') {
    return (
      <div className="gx-param" data-kind="tied">
        {head}
        <div className="gx-tied">
          <code>= {setting.expression}</code>
          <button className="gx-link" onClick={onUntie}>
            untie
          </button>
        </div>
      </div>
    );
  }
  if (setting.kind === 'other') {
    return (
      <div className="gx-param" data-kind="other">
        {head}
        <p className="note">Set in its params block (by element, solved, or by a transform).</p>
      </div>
    );
  }
  return (
    <div className="gx-param" data-kind={setting.kind}>
      {head}
      {param.size > 1 && param.size <= MOST_POINTS ? <Profile {...props} setting={setting} unit={unit} /> : <Single {...props} setting={setting} unit={unit} />}
      {setting.kind === 'free' && <FitRange {...props} setting={setting} field={field} />}
    </div>
  );
}

// A value as typed (set as it is entered or left; Escape: as it was),
// clamped to the valid range
function ValueField({ value, onCommit, param, unit }: { value: number | undefined; onCommit: (value: number) => void; param: Param; unit: string }) {
  const [text, setText] = useState<string | null>(null);
  const [low, high] = validRange(param);
  const commit = () => {
    if (text === null) return;
    const number = Number(text.replace('−', '-'));
    if (text.trim() !== '' && Number.isFinite(number)) onCommit(Math.min(high, Math.max(low, number)));
    setText(null);
  };
  return (
    <label className="gx-value">
      <input
        value={text ?? (value === undefined ? '' : significant(value, 5))}
        onChange={(event) => setText(event.currentTarget.value)}
        onFocus={(event) => event.currentTarget.select()}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === 'Enter') event.currentTarget.blur();
          if (event.key === 'Escape') {
            setText(null);
            event.currentTarget.blur();
          }
        }}
        aria-label={`The value of ${param.name}`}
        inputMode="decimal"
      />
      {unit && <span>{unit}</span>}
    </label>
  );
}

// One value (of a scalar, or of all the values of a vector at once)
function Single({ param, setting, field, onValues, unit }: ParamFieldProps & { setting: Extract<Setting, { values: number[] }>; unit: string }) {
  const reach = sliderReach(param, field);
  const value = setting.values[0] ?? param.default ?? reach.min;
  const set = (v: number) => onValues(Array(Math.max(1, param.size > 1 ? param.size : 1)).fill(v), param.name);
  return (
    <div className="gx-single">
      <div className="gx-slider">
        <StepSlider min={reach.min} max={reach.max} step={reach.step} value={value} onChange={set} format={(v) => significant(v, 4)} label={param.name} />
      </div>
      <ValueField value={value} onCommit={set} param={param} unit={unit} />
      {param.default !== null && value !== param.default && (
        <button className="gx-reset" title={`Back to its default, ${significant(param.default, 4)}`} onClick={() => set(param.default as number)}>
          ↺
        </button>
      )}
    </div>
  );
}

// The range of a free parameter in a fit: its ends on a slider
function FitRange({ param, setting, field, onRange }: ParamFieldProps & { setting: Extract<Setting, { kind: 'free' }> }) {
  const reach = sliderReach(param, field);
  const low = Math.min(reach.min, setting.min ?? reach.min);
  const high = Math.max(reach.max, setting.max ?? reach.max);
  const [min, max] = [setting.min ?? low, setting.max ?? high];
  return (
    <div className="gx-fit-range">
      <span>in a fit</span>
      <RangeSlider
        min={low}
        max={high}
        step={reach.step}
        minRange={reach.step}
        value={[min, max]}
        onChange={([a, b]) => onRange(a, b)}
        label={(v) => significant(v, 4)}
        size="xs"
        className="ui-slider gx-range"
        thumbFromLabel={`${param.name}: the least in a fit`}
        thumbToLabel={`${param.name}: the most in a fit`}
      />
      <span className="gx-fit-ends">
        {significant(min, 4)} – {significant(max, 4)}
      </span>
    </div>
  );
}

// The profile of a parameter of several values (one per ring: at their
// radii; else in their order): a point per value, dragged up and down; a
// drag across draws them. Its slider sets them all.
function Profile({ param, setting, field, rings, onValues, unit }: ParamFieldProps & { setting: Extract<Setting, { values: number[] }>; unit: string }) {
  const { ref, width } = useElementSize();
  const reach = sliderReach(param, field);
  const values = elementValues(setting.values.length ? setting.values : [param.default ?? reach.min], param.size);
  const places = rings && rings.length === param.size ? rings : values.map((_, i) => i);
  const [hover, setHover] = useState<number | null>(null);
  const drawing = useRef<{ last: number; values: number[] } | null>(null);
  const height = 96;
  const pad = { left: 6, right: 6, top: 8, bottom: 8 };
  const lowest = Math.min(reach.min, ...values);
  const highest = Math.max(reach.max, ...values);
  const span = places[places.length - 1] - places[0] || 1;
  const x = (p: number) => pad.left + ((p - places[0]) / span) * (width - pad.left - pad.right);
  const y = (v: number) => pad.top + ((highest - v) / (highest - lowest || 1)) * (height - pad.top - pad.bottom);
  const valueAt = (py: number) => {
    const v = highest - ((py - pad.top) / (height - pad.top - pad.bottom)) * (highest - lowest);
    const [low, high] = validRange(param);
    const stepped = Math.round(v / reach.step) * reach.step;
    return Math.min(Math.min(high, reach.max), Math.max(Math.max(low, reach.min), Number(stepped.toPrecision(12))));
  };
  const nearestAt = (px: number) => {
    let best = 0;
    for (let i = 1; i < places.length; i++) if (Math.abs(x(places[i]) - px) < Math.abs(x(places[best]) - px)) best = i;
    return best;
  };
  const pointer = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top] as const;
  };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const uniform = values.every((v) => v === values[0]);
  const path = places.map((p, i) => `${i ? 'L' : 'M'}${x(p).toFixed(1)},${y(values[i]).toFixed(1)}`).join('');
  return (
    <div className="gx-profile">
      <div className="gx-profile-plot" ref={ref}>
        {width > 0 && (
          <svg
            width={width}
            height={height}
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              const [px, py] = pointer(event);
              const i = nearestAt(px);
              const next = [...values];
              next[i] = valueAt(py);
              drawing.current = { last: i, values: next };
              onValues(next, `${param.name}/profile`);
            }}
            onPointerMove={(event) => {
              const [px, py] = pointer(event);
              const i = nearestAt(px);
              setHover(i);
              const draw = drawing.current;
              if (!draw) return;
              const next = [...draw.values];
              const v = valueAt(py);
              // (the rings passed over since the last move: on the line
              // from the last to this one)
              const [a, va] = [draw.last, next[draw.last]];
              for (let j = Math.min(a, i); j <= Math.max(a, i); j++) next[j] = a === i ? v : va + ((v - va) * (j - a)) / (i - a);
              drawing.current = { last: i, values: next };
              onValues(next, `${param.name}/profile`);
            }}
            onPointerUp={() => (drawing.current = null)}
            onPointerCancel={() => (drawing.current = null)}
            onPointerLeave={() => drawing.current === null && setHover(null)}
          >
            {lowest < 0 && highest > 0 && <line className="gx-axis" x1={pad.left} x2={width - pad.right} y1={y(0)} y2={y(0)} />}
            <path className="gx-profile-line" d={path} />
            {places.map((p, i) => (
              <circle key={i} className="gx-profile-point" data-on={hover === i || undefined} cx={x(p)} cy={y(values[i])} r={hover === i ? 4.5 : 3} />
            ))}
          </svg>
        )}
      </div>
      <span className="gx-profile-read">
        {hover !== null
          ? `${rings && rings.length === param.size ? `ring at ${significant(places[hover], 3)}″` : `value ${hover + 1}`} · ${significant(values[hover], 4)} ${unit}`
          : 'drag a point, or draw across them'}
      </span>
      <div className="gx-single">
        <div className="gx-slider">
          <StepSlider
            min={reach.min}
            max={reach.max}
            step={reach.step}
            value={mean}
            onChange={(v) => onValues(Array(param.size).fill(v), param.name)}
            format={(v) => `all ${significant(v, 4)}`}
            label={`${param.name}: every value`}
          />
        </div>
        <ValueField value={uniform ? values[0] : mean} onCommit={(v) => onValues(Array(param.size).fill(v), param.name)} param={param} unit={unit} />
      </div>
    </div>
  );
}
