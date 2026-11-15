// The parameters of a part of a galaxy, as numbers: each its short name
// and its value (tinted if a fit frees it, outlined if tied to an
// expression, marked if it has a value per ring), in a line or a line each
// with its title. A click opens it to set (see ParamField): its value,
// whether a fit frees it and in what range, or its tie.
import { Popover } from '@mantine/core';
import { significant } from '../../components/format';
import type { GalaxyDraft } from './draft';
import { RingMark } from './Marks';
import type { Options } from './options';
import { ParamField, unitText } from './ParamField';
import { freeRange, settingOf, shortName, withFreedom, withRange, withValues, type Param, type Setting } from './params';

export interface NumbersContext {
  draft: GalaxyDraft;
  options: Options;
  params: Param[];
  // The half-width of the field of the view (arcsec), for the reach of the
  // sliders of lengths and positions
  field: number;
}

// The text of the value of a setting: one value, the range of several, a
// tie, or what the screen does not edit
export function valueText(setting: Setting): string {
  if (setting.kind === 'tied') return `= ${setting.expression}`;
  if (setting.kind === 'other') return 'set in its params';
  const values = setting.values;
  if (values.length === 0) return '—';
  const low = Math.min(...values);
  const high = Math.max(...values);
  return low === high ? significant(low, 4) : `${significant(low, 3)}…${significant(high, 3)}`;
}

// (bare: without its name, where what it is is said, e.g. in its tile)
export function NumberChip({ ctx, param, rings, limit, bare }: { ctx: NumbersContext; param: Param; rings: number[] | null; limit?: number; bare?: boolean }) {
  const { draft, options, params } = ctx;
  const setting = settingOf(draft.state?.properties ?? {}, param);
  const label = shortName(param, options, params);
  const unit = setting.kind === 'tied' || setting.kind === 'other' ? '' : unitText(param.unit);
  return (
    <Popover position="bottom-start" offset={4} width={320} shadow="md" withinPortal trapFocus={false} classNames={{ dropdown: 'gm-pop' }}>
      <Popover.Target>
        <button
          className="gm-num"
          data-kind={setting.kind}
          data-ring={param.size > 1 || undefined}
          title={[param.title, param.name, param.desc].filter(Boolean).join(' · ')}
        >
          {!bare && <i>{label}</i>}
          {valueText(setting)}
          {unit && <span className="gm-unit">{unit}</span>}
          {param.size > 1 && <RingMark />}
        </button>
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
  const { draft, options, params, field } = ctx;
  const properties = draft.state?.properties ?? {};
  const setting = settingOf(properties, param);
  const property = properties[param.name];
  const setProperty = (next: unknown, gesture?: string) =>
    draft.edit((s) => {
      const { [param.name]: _, ...rest } = s.properties;
      return { ...s, properties: next === undefined ? rest : { ...rest, [param.name]: next } };
    }, { gesture });
  return (
    <ParamField
      param={param}
      label={shortName(param, options, params)}
      setting={setting}
      field={limit === undefined ? field : Math.min(field, limit)}
      rings={rings}
      onValues={(values, gesture) => setProperty(withValues(property, param, values), gesture)}
      onFree={(free) => setProperty(withFreedom(property, param, setting, free, freeRange(param, field)))}
      onRange={(min, max) => setProperty(withRange(property, min, max), `${param.name}/range`)}
      onUntie={() => setProperty(undefined)}
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

// The parameters of a part, a line each in the columns of the card: its
// short name and its title, and its number (see NumberChip); at a depth
// under the line of what owns them (0: beside it, e.g. in a frame; 2: under
// a line that is itself under another, e.g. the height of a trait)
export function NumberRows({ ctx, owned, rings, limit, depth = 1 }: { ctx: NumbersContext; owned: Param[]; rings: number[] | null; limit?: number; depth?: 0 | 1 | 2 }) {
  return (
    <>
      {owned.map((param) => (
        <div key={param.name} className="gm-prow" data-depth={depth}>
          <span className="gm-cell-head">
            <code className="gm-pn">{shortName(param, ctx.options, ctx.params)}</code>
            <span className="gm-pt">{param.title}</span>
          </span>
          <span className="gm-cell-nums">
            <NumberChip ctx={ctx} param={param} rings={rings} limit={limit} bare />
          </span>
        </div>
      ))}
    </>
  );
}
