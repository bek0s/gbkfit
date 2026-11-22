// The properties of the data of a dataset, as GBKFIT reads them (from the
// arrays and their headers): of their grid (the dataset is on the pixels
// of the data of its first item), their shape, pixels, spectral axis and
// beam, and the axes it leaves out; and the unit and the noise of each
// item. (The header itself, as it is in the file: see HeaderView.)
import { Fragment } from 'react';
import type { ArrayInfo } from '../../api/client';
import { kept, keptAxes } from '../../components/arrays';
import { significant } from '../../components/format';
import { isReading, NO_ARRAYS, presentSlots, restOf, slotTitle, SPECTRAL_TYPES, spectrumOf, type Resolved, type SpectralType } from './model';

// Of a dataset as it is given (see Resolved)
export function DataProperties({ resolved }: { resolved: Resolved }) {
  const { kind, draft, arrays } = resolved;
  const items = presentSlots(kind, draft).flatMap((slot) => {
    const array = (arrays[slot.key] ?? NO_ARRAYS).found.data;
    return array ? [{ title: slotTitle(slot, kind).replace(/^Moment \d · /, ''), array }] : [];
  });
  return (
    <PropertyList
      array={items[0]?.array}
      items={kind === 'moments' ? items : undefined}
      rest={kind === 'cube' ? restOf(draft.spectral) : null}
      type={kind === 'cube' ? draft.spectral.type : null}
      waiting={isReading(resolved)}
    />
  );
}

function PropertyList({
  array,
  items,
  rest,
  type = null,
  waiting,
}: {
  // The data of the grid, and those of each item, by its title (none: of
  // one item, that of the grid)
  array: ArrayInfo | undefined;
  items?: { title: string; array: ArrayInfo }[];
  rest: string | null;
  // (what the spectral axis is said to be, where its own type does not
  // say what it is)
  type?: SpectralType | null;
  waiting?: boolean;
}) {
  if (array === undefined) {
    return <p className="note">{waiting ? 'Reading…' : '—'}</p>;
  }
  const flat = array.shape.map((size, i) => [size, array.axes[i]] as const).filter(([size]) => size === 1);
  const [nx, ny, nz] = kept(array);
  const axis = keptAxes(array)[2];
  const spectrum = spectrumOf(array, type);
  const said = SPECTRAL_TYPES.find((t) => t.type === type);
  const step = array.step;
  return (
    <>
      <dl className="ds-facts">
        <dt>Shape</dt>
        <dd>{nz === undefined ? kept(array).join(' × ') : `${nx} × ${ny} px × ${nz} channels`}</dd>
        <dt>Pixel</dt>
        <dd>{step ? `${significant(Math.abs(step[0]), 3)}″ × ${significant(Math.abs(step[1]), 3)}″` : '—'}</dd>
        {nz !== undefined && (
          <>
            <dt>Channels</dt>
            <dd>
              {axis || 'no type'}
              {said && ` → ${said.type}`}
              {spectrum && rest && ` → velocities of ${rest}`}
              {step && step.length === 3 && ` · ${significant(Math.abs(step[2]), 3)} km/s`}
            </dd>
          </>
        )}
        <dt>Beam</dt>
        <dd>{beamText(array.beam)}</dd>
        {(items ?? [{ title: null, array }]).map(({ title, array: of }) => (
          <Fragment key={title ?? ''}>
            <dt>{title ?? 'Unit'}</dt>
            <dd>
              {of.unit ?? 'no unit'}
              {of.rms !== null && ` · noise ${significant(of.rms, 3)}`}
            </dd>
          </Fragment>
        ))}
      </dl>
      {flat.length > 0 && <p className="ds-fixed">Left out: {flat.map(([, name]) => name || 'no type').join(', ')} (length 1)</p>}
    </>
  );
}

function beamText(beam: ArrayInfo['beam']): string {
  if (!beam) return '—';
  const { bmaj, bmin, bpa } = beam;
  if (Array.isArray(bmaj)) return `one per channel, ${significant(Math.min(...bmaj), 3)}″ to ${significant(Math.max(...bmaj), 3)}″`;
  return `${significant(bmaj, 3)}″ × ${significant(Number(bmin), 3)}″ at ${significant(Number(bpa), 3)}°`;
}
