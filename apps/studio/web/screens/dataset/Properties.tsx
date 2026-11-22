// What GBKFIT reads from the header of the data of a dataset (of its
// first item): their shape, pixels, spectral axis, unit, beam and noise,
// and the axes it leaves out.
import type { ArrayInfo } from '../../api/client';
import { significant } from '../../components/format';
import { kept, keptAxes, SPECTRAL_TYPES, spectrumOf, type SpectralType } from './model';

export function HeaderFacts({
  array,
  rest,
  type = null,
  waiting,
}: {
  array: ArrayInfo | undefined;
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
        <dt>Unit</dt>
        <dd>{array.unit ?? '—'}</dd>
        <dt>Beam</dt>
        <dd>{beamText(array.beam)}</dd>
        {array.rms !== null && (
          <>
            <dt>Noise</dt>
            <dd>
              {significant(array.rms, 3)} {array.unit ?? ''}
            </dd>
          </>
        )}
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
