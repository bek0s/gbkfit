// The parameters of a galaxy as rows of its sheet (see ui/rows), in its
// model and in the list of its parameters. One of one value, fixed or
// free: its value typed in its box, its pill free or fixed, and, as its
// row opens, a slider over its fit range or its natural span (see
// naturalSpan); its fit range kept as it is typed (not grown to hold the
// numbers on the way, e.g. 1 and 14 of 140), grown to hold it as it is
// left. One of several values (e.g. one per ring): the range of them in
// its box; one tied to an expression: its value greyed in a dashed box,
// its tie under its title; one set otherwise in its params block (e.g. by
// element): a dash. A click on one of these opens it to set (see
// ParamField): its values (a profile, of several), whether a fit frees it
// and in what range, or its tie.
import { Popover } from '@mantine/core';
import type { ReactNode } from 'react';
import { significant } from '../../components/format';
import { ParamRow, Row, StatePill } from '../../components/ui/rows';
import type { GalaxyDraft } from './draft';
import type { Options } from './options';
import { ParamField } from './ParamField';
import { unitText } from '../../components/units';
import { naturalSpan, settingOf, shortName, valueAt, withFreedom, withRange, withValues, type Param, type Properties, type Setting } from './params';

export interface NumbersContext {
  draft: GalaxyDraft;
  options: Options;
  params: Param[];
  // The half-width of the field of the view (arcsec), for the reach of the
  // sliders of lengths and positions
  field: number;
}

// Values in a few characters: one, or the first and the last of several
// (e.g. along the rings), or a dash
function valuesText(values: number[]): string {
  if (values.length === 0) return '—';
  const [first, last] = [values[0], values[values.length - 1]];
  return values.every((v) => v === first) ? significant(first, 4) : `${significant(first, 3)} … ${significant(last, 3)}`;
}

// The text of the value of a setting: its values, a tie, or what the
// screen does not edit
export function valueText(setting: Setting): string {
  if (setting.kind === 'tied') return `= ${setting.expression}`;
  if (setting.kind === 'other') return 'set in its params';
  return valuesText(setting.values);
}

// The values of a parameter tied to another, or to an element of it (e.g.
// 'gas_vsys', 'gas_posa[0]'): those of the other; none of an expression
// (they are the fit's), or of a tie to a tie
function tiedValues(expression: string, properties: Properties, params: Param[]): number[] {
  const match = /^\s*([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?\s*$/.exec(expression);
  const other = match && params.find((p) => p.name === match[1]);
  const setting = other ? settingOf(properties, other) : null;
  if (!match || !setting || (setting.kind !== 'fixed' && setting.kind !== 'free')) return [];
  if (match[2] === undefined) return setting.values;
  const value = valueAt(setting.values, Number(match[2]));
  return value === undefined ? [] : [value];
}

// What sets the property of a parameter (undefined: none, its default);
// the edits of one gesture (e.g. the drag of a slider) one step of history
function propertySetter(draft: GalaxyDraft, param: Param) {
  return (next: unknown, gesture?: string) =>
    draft.edit((s) => {
      const { [param.name]: _, ...rest } = s.properties;
      return { ...s, properties: next === undefined ? rest : { ...rest, [param.name]: next } };
    }, { gesture });
}

// A parameter as a row (see the module): its title (else its short name),
// and under it its name as it opens (always, named), or a note given;
// limit: the most its lengths reach (e.g. the height of a trait)
export function ParamLine({
  ctx,
  param,
  rings,
  limit,
  open,
  onOpen,
  named,
  current,
  title,
  note,
  box,
}: {
  ctx: NumbersContext;
  param: Param;
  rings: number[] | null;
  limit?: number;
  open: boolean;
  onOpen: () => void;
  named?: boolean;
  current?: boolean;
  // (e.g. a menu, of the distance)
  title?: ReactNode;
  note?: ReactNode;
  // (in place of its box: e.g. its tie, as it is typed)
  box?: ReactNode;
}) {
  const { draft, options, params } = ctx;
  const properties = draft.state?.properties ?? {};
  const setting = settingOf(properties, param);
  const property = properties[param.name];
  const set = propertySetter(draft, param);
  const field = limit === undefined ? ctx.field : Math.min(ctx.field, limit);
  const heading = title ?? param.title ?? shortName(param, options, params);
  const about = { title: param.desc ?? undefined };
  const unit = setting.kind === 'other' ? '' : unitText(param.unit);
  if (param.size === 1 && (setting.kind === 'fixed' || setting.kind === 'free')) {
    const number = setting.values[0] ?? param.default ?? 0;
    const natural = naturalSpan(param, [number], field);
    return (
      <ParamRow
        param={param}
        title={heading}
        titleProps={about}
        value={setting.kind === 'free' ? { number, free: true, min: setting.min, max: setting.max } : { number, free: false }}
        natural={natural}
        unit={unit}
        open={open}
        onOpen={onOpen}
        onValue={(v) => set(withValues(property, param, [v], false), param.name)}
        onDone={() => setting.kind === 'free' && (number < (setting.min ?? number) || number > (setting.max ?? number)) && set(withValues(property, param, [number]), param.name)}
        onFree={(free, span) => set(withFreedom(property, param, setting, free, span))}
        onRange={(end, v) => {
          if (setting.kind !== 'free') return;
          const [min, max] = [setting.min ?? natural[0], setting.max ?? natural[1]];
          set(withRange(property, end === 'min' ? v : min, end === 'max' ? v : max), `${param.name}/range`);
        }}
        note={note ?? (named ? param.name : undefined)}
        current={current}
        box={box}
      />
    );
  }
  const values = setting.kind === 'fixed' || setting.kind === 'free' ? setting.values : setting.kind === 'tied' ? tiedValues(setting.expression, properties, params) : [];
  const several = param.size > 1 ? `${param.name} · ${rings?.length === param.size ? 'per ring' : `${param.size} values`}` : param.name;
  const sub = setting.kind === 'tied' ? `${param.name} = ${setting.expression}` : setting.kind === 'other' ? `${param.name} · set in its params` : (note ?? (named || open ? several : undefined));
  return (
    <Popover opened={open} onChange={(next) => next !== open && onOpen()} position="bottom-end" offset={4} width={320} shadow="md" withinPortal trapFocus={false} classNames={{ dropdown: 'gm-pop' }}>
      <Popover.Target>
        <Row
          title={heading}
          titleProps={about}
          sub={sub}
          open={open}
          onOpen={onOpen}
          // (Escape closes what sets it, the focus on its box)
          onKeyDown={(event) => event.key === 'Escape' && open && onOpen()}
          current={current}
          box={
            box ?? (
              <button type="button" className="sh-num gm-box" data-kind={setting.kind === 'fixed' || setting.kind === 'free' ? 'many' : setting.kind} title={valueText(setting)} onClick={onOpen}>
                {valuesText(values)}
              </button>
            )
          }
          unit={unit}
          pill={
            setting.kind === 'fixed' || setting.kind === 'free' ? (
              <StatePill free={setting.kind === 'free'} onToggle={() => set(withFreedom(property, param, setting, setting.kind !== 'free', naturalSpan(param, setting.values, field)))} />
            ) : (
              <span className="sh-state" data-kind={setting.kind}>
                {setting.kind === 'tied' ? 'tied' : 'set'}
              </span>
            )
          }
        />
      </Popover.Target>
      <Popover.Dropdown>
        <ParamEditor ctx={ctx} param={param} rings={rings} limit={limit} />
      </Popover.Dropdown>
    </Popover>
  );
}

// A parameter to set (see ParamField): its value, whether a fit frees it
// and in what range, or its tie
export function ParamEditor({ ctx, param, rings, limit }: { ctx: NumbersContext; param: Param; rings: number[] | null; limit?: number }) {
  const { draft, options, params } = ctx;
  const properties = draft.state?.properties ?? {};
  const setting = settingOf(properties, param);
  const property = properties[param.name];
  const set = propertySetter(draft, param);
  const field = limit === undefined ? ctx.field : Math.min(ctx.field, limit);
  return (
    <ParamField
      param={param}
      label={shortName(param, options, params)}
      setting={setting}
      field={field}
      rings={rings}
      onValues={(values, gesture) => set(withValues(property, param, values), gesture)}
      onFree={(free) => setting.kind !== 'tied' && setting.kind !== 'other' && set(withFreedom(property, param, setting, free, naturalSpan(param, setting.values, field)))}
      onRange={(min, max) => set(withRange(property, min, max), `${param.name}/range`)}
      onUntie={() => set(undefined)}
    />
  );
}

// The text of a parameter's value (see valueText), for lines that list them
// (a unit of a sign, e.g. °, right after the value)
export function paramText(ctx: NumbersContext, param: Param): string {
  const setting = settingOf(ctx.draft.state?.properties ?? {}, param);
  const unit = setting.kind === 'tied' || setting.kind === 'other' ? '' : unitText(param.unit);
  return `${shortName(param, ctx.options, ctx.params)} ${valueText(setting)}${unit.length > 1 ? ' ' : ''}${unit}`;
}

// How many of some parameters a fit frees
export function freeCount(ctx: NumbersContext, params: Param[]): number {
  const properties = ctx.draft.state?.properties ?? {};
  return params.filter((param) => settingOf(properties, param).kind === 'free').length;
}
