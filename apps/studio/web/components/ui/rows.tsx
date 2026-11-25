// The rows of the sheets of the screens that edit a model (of a lens, of
// a galaxy), on one grid for the whole sheet, so that every value starts
// at one edge: what a row is (its title, and under it a note, or its name
// as it opens), its value in a box, its unit, and how a fit takes it (a
// pill). A row of a parameter (ParamRow): its value typed in its box, its
// pill free or fixed (a click frees or fixes it), and, opened (a click on
// it; again, it closes), a slider under them over a span: the fit range
// of a free parameter (its ends set it, saved), else a span of the
// screen's own (its ends editable, not saved), at first its natural one
// (as the screen has it: e.g. its valid range, or about its value), set
// as the row opens and kept (a span that followed the value would run
// away from a drag). A value is settled as it is set (see settled).
import { useEffect, useRef, useState, type HTMLAttributes, type ReactNode, type Ref } from 'react';
import { fmt, significant } from '../format';
import { isValid, settled, snap, type Bounds } from '../values';
import { DropMenu, type MenuEntry } from './overlays';

// The parameter of a row: its name (e.g. gas_vsys) and title, and what
// settles its values
export interface RowParam extends Bounds {
  name: string;
  title?: string | null;
}

// The value of a parameter as its row shows it: its number, whether a fit
// frees it, and of a free one its range in a fit (none: its natural span)
export interface RowValue {
  number: number;
  free: boolean;
  min?: number | null;
  max?: number | null;
}

// A row: what it is (its title, and a line under it), its box, its unit
// and its pill, or what changes it (tools) in their place, or its box
// across them (wide), or across its unit alone (wideBox, e.g. a choice of
// long titles); what it holds across the row under them (e.g. a slider).
// One that opens (onOpen) is opened by a click on it, but on what it
// holds that takes a click (its box, its pill, its slider).
export function Row({
  title,
  titleProps,
  sub,
  box,
  unit,
  pill,
  tools,
  wide,
  wideBox,
  open,
  onOpen,
  current,
  bad,
  className,
  ref,
  children,
  ...rest
}: {
  title: ReactNode;
  // (of its title: e.g. what drags its value, a tooltip)
  titleProps?: HTMLAttributes<HTMLElement>;
  sub?: ReactNode;
  box?: ReactNode;
  unit?: ReactNode;
  pill?: ReactNode;
  tools?: ReactNode;
  wide?: boolean;
  wideBox?: boolean;
  open?: boolean;
  onOpen?: () => void;
  // (the row the keys act on, in a list)
  current?: boolean;
  bad?: boolean;
  className?: string;
  ref?: Ref<HTMLDivElement>;
  children?: ReactNode;
} & Omit<HTMLAttributes<HTMLDivElement>, 'title' | 'onClick'>) {
  return (
    <div
      {...rest}
      ref={ref}
      className={['sh-row', onOpen && 'sh-opens', className].filter(Boolean).join(' ')}
      data-open={open || undefined}
      data-current={current || undefined}
      data-bad={bad || undefined}
      onClick={
        onOpen &&
        ((event) => {
          if (!(event.target as HTMLElement).closest('input, button, textarea, .sh-track')) onOpen();
        })
      }
    >
      <div className="sh-title">
        <b {...titleProps}>{title}</b>
        {sub && <small>{sub}</small>}
      </div>
      {wide ? (
        <div className="sh-wide">{box}</div>
      ) : wideBox ? (
        <>
          <div className="sh-wide-box">{box}</div>
          {pill || <span className="sh-pill-room" />}
        </>
      ) : (
        <>
          {box || <span />}
          {tools !== undefined ? (
            <span className="sh-tools">{tools}</span>
          ) : (
            <>
              {/* (a unit longer than its cell cut, all of it in its tooltip) */}
              <span className="sh-unit" title={typeof unit === 'string' && unit.length > 6 ? unit : undefined}>
                {unit}
              </span>
              {/* (the room of a pill, which a sheet without pills drops) */}
              {pill || <span className="sh-pill-room" />}
            </>
          )}
        </>
      )}
      {children}
    </div>
  );
}

// How a fit takes a parameter, a pill: free (a dot of the accent) or
// fixed; a click frees or fixes it
export function StatePill({ free, onToggle }: { free: boolean; onToggle: () => void }) {
  return (
    <button type="button" className="sh-state" data-free={free || undefined} title={free ? 'Free in a fit: click to fix it' : 'Fixed: click to free it in a fit'} onClick={onToggle}>
      {free ? 'free' : 'fixed'}
    </button>
  );
}

// A parameter as a row (see the module): its title (else its own), its
// name under it as it opens (or a note given, always); its box, unless
// another is given (e.g. a field of its tie, as it is typed)
export function ParamRow({
  param,
  value,
  natural,
  unit,
  open,
  onOpen,
  onValue,
  onFree,
  onRange,
  title,
  titleProps,
  note,
  current,
  box,
  onDone,
}: {
  param: RowParam;
  value: RowValue;
  // The span of its slider of no fit range (see the module)
  natural: [number, number];
  // Its unit, as written (e.g. km/s)
  unit: string;
  open: boolean;
  onOpen: () => void;
  onValue: (value: number) => void;
  // (freed: its fit range the span of its slider)
  onFree: (free: boolean, span: [number, number]) => void;
  onRange: (end: 'min' | 'max', value: number) => void;
  title?: ReactNode;
  titleProps?: HTMLAttributes<HTMLElement>;
  note?: ReactNode;
  current?: boolean;
  box?: ReactNode;
  // (as the value typed is left)
  onDone?: () => void;
}) {
  const { number, free } = value;
  // (the span of a fixed parameter: its natural one as the row opens, or
  // as its ends were set on the screen; kept, not of the value as it moves)
  const [own, setOwn] = useState<[number, number] | null>(null);
  useEffect(() => {
    if (open && own === null) setOwn(natural);
  }, [open, own, natural]);
  const span: [number, number] = free ? [value.min ?? natural[0], value.max ?? natural[1]] : (own ?? natural);
  const set = (v: number) => onValue(settled(v, param));
  const setEnd = (end: 0 | 1, v: number | undefined) => {
    if (v === undefined) return;
    if (free) onRange(end === 0 ? 'min' : 'max', v);
    else setOwn(end === 0 ? [v, span[1]] : [span[0], v]);
  };
  const at = Math.max(0, Math.min(1, (number - span[0]) / (span[1] - span[0] || 1)));
  return (
    <Row
      title={title ?? param.title ?? param.name}
      titleProps={titleProps}
      sub={note ?? (open && param.name)}
      open={open}
      onOpen={onOpen}
      current={current}
      box={box ?? <NumberInput value={number} label={param.title ?? param.name} valid={(v) => isValid(v, param)} onChange={set} onDone={onDone} />}
      unit={unit}
      pill={<StatePill free={free} onToggle={() => onFree(!free, span)} />}
    >
      {open && (
        <div className="sh-slide" data-free={free || undefined}>
          <End value={span[0]} title={free ? 'Least value in a fit' : 'Where the slider starts'} onChange={(v) => setEnd(0, v)} />
          <Track at={at} onAt={(t) => set(snap(span[0] + t * (span[1] - span[0]), span))} />
          <End value={span[1]} title={free ? 'Greatest value in a fit' : 'Where the slider ends'} onChange={(v) => setEnd(1, v)} />
        </div>
      )}
    </Row>
  );
}

// A number, typed: a click (or Tab) selects it, to type another over it;
// set as it is typed once it is a valid one (not before: e.g. not 0 on
// the way to 0.5), cleared once typed away, if it can be; Enter keeps
// it, Escape puts back the value it had as it took the focus (both leave
// it); shown to 6 digits, small and large ones with an exponent (see
// significant), not rounded as in a summary (see useScrub for the drag);
// what is done as it is left, if given
export function NumberInput({
  value,
  label,
  onChange,
  onClear,
  onDone,
  placeholder,
  valid = Number.isFinite,
}: {
  value: number | undefined;
  label: string;
  onChange: (value: number) => void;
  onClear?: () => void;
  onDone?: () => void;
  placeholder?: string;
  valid?: (value: number) => boolean;
}) {
  const [typing, setTyping] = useState<string | null>(null);
  // (the value as it took the focus, which Escape puts back)
  const before = useRef<number | undefined>(undefined);
  return (
    <input
      className="sh-num"
      aria-label={label}
      inputMode="decimal"
      value={typing ?? (value === undefined ? '' : significant(value, 6))}
      placeholder={placeholder}
      onFocus={(event) => {
        before.current = value;
        event.currentTarget.select();
      }}
      onChange={(event) => {
        setTyping(event.target.value);
        const text = event.target.value.trim();
        const parsed = Number(text);
        if (text === '') onClear?.();
        else if (valid(parsed)) onChange(parsed);
      }}
      onBlur={() => {
        setTyping(null);
        onDone?.();
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape' && typing !== null) {
          setTyping(null);
          if (before.current === undefined) onClear?.();
          else if (before.current !== value) onChange(before.current);
        }
        if (event.key === 'Enter' || event.key === 'Escape') event.currentTarget.blur();
      }}
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
    className: 'sh-scrub',
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

// An end of a span: its number, typed in place, set as it is left (or
// entered); Escape leaves it as it was
function End({ value, title, onChange }: { value: number; title: string; onChange: (value: number | undefined) => void }) {
  const [typing, setTyping] = useState<string | null>(null);
  // (left by Escape: what is typed is not set)
  const discard = useRef(false);
  return (
    <input
      className="sh-end"
      title={title}
      aria-label={title}
      value={typing ?? fmt(value)}
      onChange={(event) => setTyping(event.target.value)}
      onBlur={() => {
        if (!discard.current && typing !== null && typing.trim() !== '' && Number.isFinite(Number(typing))) onChange(Number(typing));
        discard.current = false;
        setTyping(null);
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') discard.current = true;
        if (event.key === 'Enter' || event.key === 'Escape') event.currentTarget.blur();
      }}
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
      className="sh-track"
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
      <div className="sh-fill" style={{ width: `${at * 100}%` }} />
      <div className="sh-knob" style={{ left: `${at * 100}%` }} />
    </div>
  );
}

// A choice, in the box of a row: what is chosen, and a menu of the
// choices
export function ChoiceBox({ label, text, title, entries, width = 240 }: { label: string; text: ReactNode; title?: string; entries: MenuEntry[]; width?: number }) {
  return (
    <DropMenu
      label={label}
      width={width}
      entries={entries}
      target={
        <button type="button" className="sh-num gm-choice" title={title ?? (typeof text === 'string' ? text : undefined)}>
          <span>{text}</span>
        </button>
      }
    />
  );
}
