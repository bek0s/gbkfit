// The spectral axis of a cube, in the dock: a row for each of its options
// (its type, its unit, the convention of its velocities, and the line that
// is 0 km/s, at the redshift of the galaxy). Each shows what the header
// says, or a field, once the library finds a problem of that option (see
// the server's ArrayProblem) or one is given: the field says the problem
// until it is gone. No rule is the page's: the library describes the file
// again with each option given, and says the next problem, if any.
import type { ReactNode } from 'react';
import { useEffect } from 'react';
import type { ArrayProblem } from '../../api/client';
import { useLines } from '../../api/queries';
import { Choice, NumberField, Segments } from '../../components/ui/inputs';
import { catalogueLineOf, catalogueRests, keptAxes, keptAxisUnits, LINE_UNITS, mediumOf, restOf, SPECTRAL_TYPES, spectrumOf, type Resolved, type Spectral, type SpectralType } from './model';

// The problem of an array at an option (e.g. 'spectral.unit'), if any
function problemAt(problems: ArrayProblem[], option: string): ArrayProblem | null {
  return problems.find((problem) => problem.path.join('.') === option) ?? null;
}

// Whether a cube has anything for the user to say of its spectral axis: a
// problem of it, or an option of it given
export function hasSpectralChoices({ kind, draft, arrays }: Resolved): boolean {
  const data = arrays.cube?.found.data;
  if (kind !== 'cube' || data === undefined) return false;
  const { type, convention, line, axisUnit } = draft.spectral;
  return data.problems.some((problem) => problem.path[0] === 'spectral') || type !== null || convention !== null || line.trim() !== '' || axisUnit !== '';
}

export function SpectralFields({ resolved, onChange }: { resolved: Resolved; onChange?: (spectral: Spectral) => void }) {
  const { draft, arrays } = resolved;
  const { spectral } = draft;
  const data = arrays.cube?.found.data;
  const spectrum = spectrumOf(data, spectral.type);
  // (the type of the axis: as given, or of its header)
  const axis = spectral.type ?? (data ? keptAxes(data)[2] : '');
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
  const at = (option: string) => problemAt(data.problems, option);
  const [typeProblem, unitProblem, conventionProblem, restProblem] = ['spectral.type', 'spectral.unit', 'spectral.convention', 'spectral.rest'].map(at);
  const rest = restOf(spectral);
  const restWord = spectrum === 'wave' ? 'wavelength' : 'frequency';
  const others = spectrum ? othersNear(rests, chosen, Number(spectral.line), spectrum) : [];
  const fixed = (text: string) => <span className="sp-fixed">{text || '—'}</span>;
  return (
    <div className="sp-rows">
      <Row label="Type" problem={typeProblem}>
        {!(typeProblem || spectral.type) ? (
          fixed(axis)
        ) : onChange ? (
          <Choice
            items={SPECTRAL_TYPES.map(({ type, text }) => ({ value: type, label: `${text} (${type})` }))}
            value={spectral.type ?? ''}
            placeholder={typeProblem?.message ?? 'Choose'}
            label="The type of the spectral axis"
            onChange={(type) => onChange({ ...spectral, type: type as SpectralType })}
          />
        ) : (
          fixed(said ? `${said.text} (${said.type})` : '')
        )}
      </Row>
      <Row label="Unit" problem={unitProblem}>
        {!(unitProblem || spectral.axisUnit) ? (
          fixed(keptAxisUnits(data)[2] ?? '')
        ) : onChange ? (
          <Choice
            items={AXIS_UNITS}
            value={spectral.axisUnit}
            placeholder={unitProblem?.message ?? 'Choose'}
            label="The unit of the spectral axis"
            onChange={(axisUnit) => onChange({ ...spectral, axisUnit })}
          />
        ) : (
          fixed(spectral.axisUnit)
        )}
      </Row>
      {(conventionProblem || spectral.convention !== null) && (
        <Row label="Velocities" problem={conventionProblem}>
          <div className="sp-inline">
            <Segments
              items={(['radio', 'optical'] as const).map((value) => ({ value, label: value === 'radio' ? 'Radio' : 'Optical', disabled: !onChange }))}
              value={spectral.convention}
              label="The convention of the velocities"
              onChange={(convention) => onChange?.({ ...spectral, convention })}
            />
            {conventionProblem && <span className="sp-flag">{conventionProblem.message}</span>}
          </div>
        </Row>
      )}
      {(restProblem || spectral.line.trim() !== '') && (
        <>
          <Row label="Line" problem={restProblem}>
            {onChange ? (
              <Choice
                items={rests.map(({ name, rest }) => ({ value: name, label: name, description: `${rest} ${unit}` }))}
                value={chosen ?? ''}
                placeholder={restProblem?.message ?? (spectral.line.trim() ? 'Other' : 'Choose')}
                label="The line"
                onChange={(name) => onChange({ ...spectral, unit, line: rests.find((line) => line.name === name)?.rest ?? spectral.line })}
              />
            ) : (
              fixed(spectral.line ? `${chosen ? `${chosen}, ` : ''}${spectral.line} ${spectral.unit}` : '')
            )}
          </Row>
          {onChange && (
            <Row label="Rest">
              <div className="ds-inline">
                <NumberField value={spectral.line} label={`The rest ${restWord} of the line`} onChange={(line) => onChange({ ...spectral, line })} />
                <Choice items={units.map((unit) => ({ value: unit, label: unit }))} value={unit} label="The unit of the line" onChange={setUnit} />
              </div>
            </Row>
          )}
          {(onChange || spectral.redshift) && (
            <Row label="Redshift">
              <div className="sp-pair">
                {onChange ? <NumberField value={spectral.redshift} placeholder="0" label="The redshift of the galaxy" onChange={(redshift) => onChange({ ...spectral, redshift })} /> : fixed(spectral.redshift)}
                {/* (the rest that is 0 km/s, at the redshift) */}
                <span className="sp-result">{rest ? `0 km/s = ${restText(rest)}` : ''}</span>
              </div>
            </Row>
          )}
          {/* (the lines of the catalogue near the one chosen: its line is
              always there, so that choosing one moves nothing) */}
          <p className="sp-near">{others.length > 0 && `Near: ${others.map(({ name, velocity }) => `${name} ${velocity > 0 ? '+' : '−'}${Math.round(Math.abs(velocity))} km/s`).join(', ')}`}</p>
        </>
      )}
    </div>
  );
}

// A row of the axis: its label, and its value or field; marked while the
// library finds a problem of its option (which its field says)
function Row({ label, problem = null, children }: { label: string; problem?: ArrayProblem | null; children: ReactNode }) {
  return (
    <div className="sp-row" data-attention={problem ? true : undefined}>
      <span className="sp-label">{label}</span>
      <div className="sp-value">{children}</div>
    </div>
  );
}

// The units a spectral axis can be given, by what its values are (the
// library says which fit its type)
const AXIS_UNITS = [
  ...['km/s', 'm/s'].map((value) => ({ value, label: value, group: 'Velocities' })),
  ...['Angstrom', 'nm', 'um', 'm'].map((value) => ({ value, label: value, group: 'Wavelengths' })),
  ...['GHz', 'MHz', 'kHz', 'Hz'].map((value) => ({ value, label: value, group: 'Frequencies' })),
];

// The speed of light (km/s)
const LIGHT = 299792.458;

// The lines of the catalogue near the line chosen (within NEAR km/s of it,
// as the velocities of the axis read them: optical of wavelengths, radio
// of frequencies), but it, nearest first
const NEAR = 2500;

function othersNear(rests: { name: string; rest: string }[], chosen: string | null, line: number, spectrum: 'wave' | 'freq'): { name: string; velocity: number }[] {
  if (!(line > 0)) return [];
  return rests
    .filter(({ name, rest }) => name !== chosen && Number(rest) !== line)
    .map(({ name, rest }) => ({ name, velocity: spectrum === 'wave' ? LIGHT * (Number(rest) / line - 1) : LIGHT * (1 - Number(rest) / line) }))
    .filter(({ velocity }) => Math.abs(velocity) <= NEAR)
    .sort((a, b) => Math.abs(a.velocity) - Math.abs(b.velocity))
    .slice(0, 4);
}

// A rest as said to the user: to 7 digits, in the symbols of its unit
function restText(rest: string): string {
  const [value, unit] = rest.split(' ');
  return `${Number(Number(value).toPrecision(7))} ${{ Angstrom: 'Å', micron: 'µm' }[unit] ?? unit}`;
}
