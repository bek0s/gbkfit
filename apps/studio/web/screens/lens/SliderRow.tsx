// A parameter as a row: its title, its value (typed), its unit, whether a fit frees it, and a slider under them
// over a span: the fit range of a free parameter (its ends set it, saved),
// else a span of the screen's own (its ends editable, not saved), at first
// that of the parameter's valid range, or of what the picture shows (the
// field, of a position), or about its value, set as the row opens and kept
// (a span that followed the value would run away from a drag). A
// periodic parameter (e.g. a
// position angle) wraps. The name and the slider show once the row is
// opened (a click on it; again, it closes), as the galaxy screen's rows.
import { useEffect, useRef, useState } from 'react';
import { isObject } from './draft';

// A parameter as the server describes it (see LensInfo)
export interface LensParam {
  name: string;
  title?: string | null;
  unit: string | null;
  default: number | null;
  range?: string | null;
  period?: number | null;
  owner: [string, number];
}

// The value of a parameter: a number, or free in a range ({value, min,
// max}; see the params block)
export type ParamValue = number | { value?: number; min?: number | null; max?: number | null } | undefined;

const UNITS: Record<string, string> = { arcsec: '″', deg: '°' };

export function unitText(unit: string | null): string {
  return unit ? (UNITS[unit] ?? unit) : '';
}

// The number of a parameter's value (its default if none)
export function numberOf(param: LensParam, value: ParamValue): number {
  const v = isObject(value) ? value.value : value;
  return typeof v === 'number' ? v : (param.default ?? 0);
}

// The span of the slider of a parameter of no fit range (see the module):
// field: the half-width of the picture (arcsec)
export function naturalSpan(param: LensParam, value: number, field: number): [number, number] {
  if (param.period) return [0, param.period];
  const [lo, hi] = parseRange(param.range);
  if (Number.isFinite(lo) && Number.isFinite(hi)) return [lo, hi];
  if (param.unit === 'arcsec') return [Number.isFinite(lo) ? lo : -field, field];
  const reach = Math.max(4 * Math.abs(value), 0.1);
  return [Number.isFinite(lo) ? lo : value - reach, Number.isFinite(hi) ? hi : value + reach];
}

// The ends of a range as the server writes it (e.g. '[0, inf)')
function parseRange(range: string | null | undefined): [number, number] {
  const match = range?.match(/^[[(]\s*([^,]+),\s*([^\])]+)[\])]$/);
  const read = (text: string | undefined, fallback: number) => {
    if (text === undefined) return fallback;
    const t = text.trim();
    return t === 'inf' ? Infinity : t === '-inf' ? -Infinity : Number(t);
  };
  return [read(match?.[1], -Infinity), read(match?.[2], Infinity)];
}

// A value dragged over a span, rounded to a thousandth of the span (a
// power of ten: e.g. 0.001 of a redshift over 0 to 4): a drag says no more
// (a value typed is kept as typed)
export function snap(value: number, span: [number, number]): number {
  const step = 10 ** Math.floor(Math.log10(Math.abs(span[1] - span[0]) / 1000 || 1e-3));
  return Number((Math.round(value / step) * step).toFixed(Math.max(0, -Math.floor(Math.log10(step)))));
}

export function fmt(value: number): string {
  const size = Math.abs(value);
  return size >= 100 ? value.toFixed(0) : size >= 10 ? value.toFixed(1) : size >= 0.01 || size === 0 ? value.toFixed(2) : String(Number(value.toPrecision(2)));
}

export function SliderRow({
  param,
  value,
  field,
  open,
  onOpen,
  onChange,
}: {
  param: LensParam;
  value: ParamValue;
  field: number;
  // Whether its name and slider show, and what opens or closes it
  open: boolean;
  onOpen: () => void;
  onChange: (value: ParamValue) => void;
}) {
  const free = isObject(value);
  const number = numberOf(param, value);
  // (the span of a fixed parameter: its natural one as the row opens, or as
  // its ends were set on the screen; kept, not of the value as it moves)
  const [own, setOwn] = useState<[number, number] | null>(null);
  const natural = naturalSpan(param, number, field);
  useEffect(() => {
    if (open && own === null) setOwn(natural);
  }, [open, own, natural]);
  const span: [number, number] = free ? [value.min ?? natural[0], value.max ?? natural[1]] : (own ?? natural);
  const short = param.name.slice(param.name.indexOf('_') + 1);
  const set = (v: number) => {
    const wrapped = param.period ? ((v % param.period) + param.period) % param.period : v;
    const [lo, hi] = parseRange(param.range);
    const kept = Math.min(Math.max(wrapped, lo), hi);
    onChange(free ? { ...value, value: kept } : kept);
  };
  const setEnd = (end: 0 | 1, v: number | undefined) => {
    if (v === undefined) return;
    if (free) onChange({ ...value, [end === 0 ? 'min' : 'max']: v });
    else setOwn(end === 0 ? [v, span[1]] : [span[0], v]);
  };
  const toggle = () => (free ? onChange(value.value ?? number) : onChange({ value: number, min: span[0], max: span[1] }));
  const at = Math.max(0, Math.min(1, (number - span[0]) / (span[1] - span[0] || 1)));
  return (
    <div
      className="lp-row"
      data-open={open || undefined}
      onClick={(event) => {
        // (not a click on what it holds that takes it: its value, its
        // state, its slider)
        if (!(event.target as HTMLElement).closest('input, button, .lp-track')) onOpen();
      }}
    >
      <div className="lp-title">
        <b>{param.title ?? short}</b>
        {open && <small>{param.name}</small>}
      </div>
      <NumberInput value={number} label={param.title ?? param.name} onChange={set} />
      <span className="lp-unit">{unitText(param.unit)}</span>
      <button type="button" className="lp-state" data-free={free || undefined} title={free ? 'Free in a fit: click to fix it' : 'Fixed: click to free it in a fit'} onClick={toggle}>
        {free ? 'free' : 'fixed'}
      </button>
      {open && (
        <div className="lp-slide" data-free={free || undefined}>
          <End value={span[0]} title={free ? 'Least value in a fit' : 'Where the slider starts'} onChange={(v) => setEnd(0, v)} />
          <Track at={at} onAt={(t) => set(snap(span[0] + t * (span[1] - span[0]), span))} />
          <End value={span[1]} title={free ? 'Greatest value in a fit' : 'Where the slider ends'} onChange={(v) => setEnd(1, v)} />
        </div>
      )}
    </div>
  );
}

// A number, typed: a click (or Tab) selects it, to type another over it;
// one typed away is cleared, if it can be (see useScrub for the drag)
export function NumberInput({ value, label, onChange, onClear, placeholder }: { value: number | undefined; label: string; onChange: (value: number) => void; onClear?: () => void; placeholder?: string }) {
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <input
      className="lp-num"
      aria-label={label}
      inputMode="decimal"
      value={typing ?? (value === undefined ? '' : fmt(value))}
      placeholder={placeholder}
      onFocus={(event) => event.currentTarget.select()}
      onChange={(event) => {
        setTyping(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() === '') onClear?.();
        else if (Number.isFinite(parsed)) onChange(parsed);
      }}
      onBlur={() => setTyping(null)}
      onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.blur()}
    />
  );
}

// What makes an element (e.g. the label of a number) drag a value
// sideways: the span across 200 pixels, rounded (see snap)
// (a click without a drag stays a click, e.g. of the row it is in; a drag
// is no click)
export function useScrub(value: number | undefined, span: [number, number], onChange: (value: number) => void) {
  const start = useRef<{ x: number; v: number } | null>(null);
  const dragged = useRef(false);
  return {
    className: 'lp-scrub',
    title: 'Drag sideways to change it',
    onPointerDown: (event: React.PointerEvent<HTMLElement>) => {
      start.current = { x: event.clientX, v: value ?? 0 };
      dragged.current = false;
      event.currentTarget.setPointerCapture(event.pointerId);
      event.preventDefault();
    },
    onPointerMove: (event: React.PointerEvent<HTMLElement>) => {
      const s = start.current;
      if (!s || !(event.buttons & 1)) return;
      if (!dragged.current && Math.abs(event.clientX - s.x) < 3) return;
      dragged.current = true;
      onChange(snap(s.v + ((event.clientX - s.x) / 200) * (span[1] - span[0]), span));
    },
    onPointerUp: () => {
      start.current = null;
    },
    onClick: (event: React.MouseEvent<HTMLElement>) => {
      if (dragged.current) event.stopPropagation();
    },
  };
}

// An end of a span: its number, typed in place
function End({ value, title, onChange }: { value: number; title: string; onChange: (value: number | undefined) => void }) {
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <input
      className="lp-end"
      title={title}
      aria-label={title}
      value={typing ?? fmt(value)}
      onChange={(event) => setTyping(event.target.value)}
      onBlur={() => {
        if (typing !== null && typing.trim() !== '' && Number.isFinite(Number(typing))) onChange(Number(typing));
        setTyping(null);
      }}
      onKeyDown={(event) => event.key === 'Enter' && event.currentTarget.blur()}
    />
  );
}

// A track, its knob at a fraction of it; a drag or a click moves it
export function Track({ at, onAt }: { at: number; onAt: (at: number) => void }) {
  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    onAt(Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)));
  };
  return (
    <div
      className="lp-track"
      role="slider"
      aria-valuenow={at}
      onPointerDown={(event) => {
        // (no text selected by the drag: a drag that starts on a selection
        // drags the text instead, and the browser cancels the slider's)
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        move(event);
      }}
      onPointerMove={(event) => event.buttons & 1 && move(event)}
    >
      <div className="lp-fill" style={{ width: `${at * 100}%` }} />
      <div className="lp-knob" style={{ left: `${at * 100}%` }} />
    </div>
  );
}
