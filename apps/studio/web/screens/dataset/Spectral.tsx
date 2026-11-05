// The spectral axis of a cube, in the dock: what its header leaves to the
// user. The convention of its velocities, if the header does not say
// (e.g. VELO-HEL without VELREF), and, for channels of wavelengths or
// frequencies (e.g. an IFU cube), the line whose velocities they become:
// its rest, at the redshift of the galaxy (the dataset reads the cube in
// velocities of that rest; its file stays as it is).
import { useEffect } from 'react';
import { significant } from '../../components/format';
import { Choice, NumberField, Segments } from '../../components/ui/inputs';
import { keptAxes, LINE_UNITS, restOf, spectrumOf, type Resolved, type Spectral } from './model';

// Whether a cube has anything for the user to say of its spectral axis
export function hasSpectralChoices({ kind, draft, arrays }: Resolved): boolean {
  const data = arrays.cube?.found.data;
  if (kind !== 'cube' || data === undefined) return false;
  return data.ambiguous_velocity || spectrumOf(data) !== null || draft.spectral.convention !== null;
}

export function SpectralFields({ resolved, onChange }: { resolved: Resolved; onChange?: (spectral: Spectral) => void }) {
  const { draft, arrays } = resolved;
  const { spectral } = draft;
  const data = arrays.cube?.found.data;
  const spectrum = spectrumOf(data);
  const axis = data ? keptAxes(data)[2] : '';
  // (the unit of the line: one of those of the axis)
  const units = spectrum ? LINE_UNITS[spectrum] : [];
  useEffect(() => {
    if (onChange && spectrum && !units.includes(spectral.unit)) onChange({ ...spectral, unit: units[0] });
  });
  if (data === undefined) return null;
  const rest = restOf(spectral);
  const width = data.step && data.step.length === 3 ? Math.abs(data.step[2]) : null;
  return (
    <div className="ds-form">
      {(data.ambiguous_velocity || spectral.convention !== null) && (
        <>
          <p className="ds-text">
            {data.ambiguous_velocity
              ? `The header says velocities (${axis}) but not whether they are radio or optical. MIRIAD and AIPS write radio ones.`
              : 'The velocities of the header are read as:'}
          </p>
          <div className="ds-field">
            <span className="ds-label">Velocities</span>
            <Segments
              items={(['radio', 'optical'] as const).map((value) => ({ value, label: value === 'radio' ? 'Radio' : 'Optical', disabled: !onChange }))}
              value={spectral.convention}
              label="The convention of the velocities"
              onChange={(convention) => onChange?.({ ...spectral, convention })}
            />
          </div>
        </>
      )}
      {spectrum && (
        <>
          <p className="ds-text">
            The channels are {spectrum === 'wave' ? 'wavelengths' : 'frequencies'} ({axis}
            {axis === 'AWAV' ? ', in air' : axis === 'WAVE' ? ', in vacuum' : ''}). GBKFIT reads them as velocities of a line
            {onChange ? `: give its rest${axis === 'AWAV' ? ' wavelength in air' : ''}, and the redshift of the galaxy.` : ':'}
          </p>
          <div className="ds-field">
            <span className="ds-label">Line</span>
            <div className="ds-inline">
              {onChange ? (
                <>
                  <NumberField value={spectral.line} placeholder={spectrum === 'wave' ? '6562.8' : '1420.405752'} label="The rest of the line" onChange={(line) => onChange({ ...spectral, line })} />
                  <Choice items={units.map((unit) => ({ value: unit, label: unit }))} value={spectral.unit || units[0]} label="The unit of the line" onChange={(unit) => onChange({ ...spectral, unit })} />
                </>
              ) : (
                <span className="ds-mono">{spectral.line ? `${spectral.line} ${spectral.unit}` : '—'}</span>
              )}
            </div>
          </div>
          {(onChange || spectral.redshift) && (
            <div className="ds-field">
              <span className="ds-label">Redshift</span>
              {onChange ? (
                <NumberField value={spectral.redshift} placeholder="0" label="The redshift" onChange={(redshift) => onChange({ ...spectral, redshift })} />
              ) : (
                <span className="ds-mono">{spectral.redshift}</span>
              )}
            </div>
          )}
          {rest && onChange && (
            <p className="ds-result">
              Velocities of {rest}
              {width !== null && ` · channels of ${significant(width, 3)} km/s`}
            </p>
          )}
        </>
      )}
    </div>
  );
}
