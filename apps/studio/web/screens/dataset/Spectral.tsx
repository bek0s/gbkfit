// The spectral axis of a cube, in the dock: what its header leaves to the
// user. What the axis is, if its header gives it a type FITS does not
// define (e.g. CALIFA's WAVELENGTH): chosen, nothing suggested, as such a
// type says nothing of what its values are. The convention of its
// velocities, if the header does not say
// (e.g. VELO-HEL without VELREF), and, for channels of wavelengths or
// frequencies (e.g. an IFU cube), the line whose velocities they become:
// one of the catalogue of GBKFIT (its rest in air or in vacuum, as the
// type of the axis says) or a rest typed, at the redshift of the galaxy
// (the dataset reads the cube in velocities of that rest; its file stays
// as it is).
import { useEffect } from 'react';
import { useLines } from '../../api/queries';
import { significant } from '../../components/format';
import { Choice, NumberField, Segments } from '../../components/ui/inputs';
import { catalogueLineOf, catalogueRests, keptAxes, LINE_UNITS, mediumOf, restOf, SPECTRAL_TYPES, spectrumOf, type Resolved, type Spectral, type SpectralType } from './model';

// Whether a cube has anything for the user to say of its spectral axis
export function hasSpectralChoices({ kind, draft, arrays }: Resolved): boolean {
  const data = arrays.cube?.found.data;
  if (kind !== 'cube' || data === undefined) return false;
  return Boolean(data.nonstandard_type) || data.ambiguous_velocity || spectrumOf(data, draft.spectral.type) !== null || draft.spectral.convention !== null;
}

export function SpectralFields({ resolved, onChange }: { resolved: Resolved; onChange?: (spectral: Spectral) => void }) {
  const { draft, arrays } = resolved;
  const { spectral } = draft;
  const data = arrays.cube?.found.data;
  const spectrum = spectrumOf(data, spectral.type);
  // (the type of the axis: of its header, or else as said)
  const axis = data ? (data.nonstandard_type ? (spectral.type ?? '') : keptAxes(data)[2]) : '';
  const said = SPECTRAL_TYPES.find(({ type }) => type === spectral.type);
  // (the unit of the line: one of those of the axis)
  const units = spectrum ? LINE_UNITS[spectrum] : [];
  useEffect(() => {
    if (onChange && spectrum && !units.includes(spectral.unit)) onChange({ ...spectral, unit: units[0] });
  });
  // (the lines of the catalogue, in the medium of the axis, and the one
  // whose rest is typed)
  const catalogue = useLines().data ?? [];
  const medium = mediumOf(axis);
  const unit = spectral.unit || units[0];
  const restsIn = (unit: string) => (spectrum ? catalogueRests(catalogue, spectrum, medium, unit) : []);
  const rests = restsIn(unit);
  const chosen = catalogueLineOf(rests, spectral.line);
  // (another unit keeps a line of the catalogue, in that unit)
  const setUnit = (unit: string) => {
    const kept = chosen === null ? undefined : restsIn(unit).find(({ name }) => name === chosen);
    onChange?.({ ...spectral, unit, line: kept ? kept.rest : spectral.line });
  };
  if (data === undefined) return null;
  const rest = restOf(spectral);
  const width = data.step && data.step.length === 3 ? Math.abs(data.step[2]) : null;
  return (
    <div className="ds-form">
      {data.nonstandard_type && (
        <>
          <p className="ds-text">{data.nonstandard_type} is not a FITS axis type.</p>
          <div className="ds-field">
            <span className="ds-label">Axis</span>
            {onChange ? (
              <Choice
                items={SPECTRAL_TYPES.map(({ type, text }) => ({ value: type, label: `${text} (${type})` }))}
                value={spectral.type ?? ''}
                placeholder="Choose"
                label="What the spectral axis is"
                onChange={(type) => onChange({ ...spectral, type: type as SpectralType })}
              />
            ) : (
              <span className="ds-mono">{said ? `${said.text} (${said.type})` : '—'}</span>
            )}
          </div>
        </>
      )}
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
            {onChange
              ? `Choose the line or give its rest ${spectrum === 'wave' ? `wavelength${medium ? ` in ${medium}` : ''}` : 'frequency'}, and the redshift.`
              : 'The line:'}
          </p>
          {onChange ? (
            <>
              <div className="ds-field">
                <span className="ds-label">Line</span>
                <Choice
                  items={rests.map(({ name, rest }) => ({ value: name, label: name, description: `${rest} ${unit}` }))}
                  value={chosen ?? ''}
                  placeholder={spectral.line.trim() ? 'Other' : 'Choose'}
                  label="The line"
                  onChange={(name) => onChange({ ...spectral, unit, line: rests.find((line) => line.name === name)?.rest ?? spectral.line })}
                />
              </div>
              <div className="ds-field">
                <span className="ds-label">Rest</span>
                <div className="ds-inline">
                  <NumberField value={spectral.line} label="The rest of the line" onChange={(line) => onChange({ ...spectral, line })} />
                  <Choice items={units.map((unit) => ({ value: unit, label: unit }))} value={unit} label="The unit of the line" onChange={setUnit} />
                </div>
              </div>
            </>
          ) : (
            <div className="ds-field">
              <span className="ds-label">Line</span>
              <span className="ds-mono">{spectral.line ? `${chosen ? `${chosen}, ` : ''}${spectral.line} ${spectral.unit}` : '—'}</span>
            </div>
          )}
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
