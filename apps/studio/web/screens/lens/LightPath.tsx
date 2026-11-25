// The path of the light of a lens: you, the lens at its redshift and the
// source its quoted strengths are of at its own, apart as their distances
// are (comoving, of a flat LCDM of matter density 0.3: a sketch, not the
// cosmology of a setup), with the rays bent at the lens.
import { significant } from '../../components/format';

const MATTER = 0.3;

function distance(z: number): number {
  let sum = 0;
  const steps = 200;
  for (let i = 0; i < steps; i++) {
    const zz = ((i + 0.5) * z) / steps;
    sum += 1 / Math.sqrt(MATTER * (1 + zz) ** 3 + (1 - MATTER));
  }
  return (sum * z) / steps;
}

// A redshift as written (to 6 digits; ? if none)
function zOf(z: number | undefined): string {
  return z === undefined ? '?' : significant(z, 6);
}

export function LightPath({ lens, source }: { lens: number | undefined; source: number | undefined }) {
  const width = 360;
  const x0 = 12;
  const far = distance(Math.max(source ?? 2, (lens ?? 0.5) + 0.2));
  const at = (z: number) => x0 + (distance(z) / far) * (width - 2 * x0);
  const xl = lens === undefined ? width / 3 : at(lens);
  const xs = width - x0;
  return (
    <svg className="lp-path" viewBox={`0 0 ${width} 60`} role="img" aria-label={`The path of the light: you, the lens at z ${zOf(lens)}, the source at z ${zOf(source)}`}>
      <line className="lp-path-axis" x1={x0} y1={31} x2={xs} y2={31} />
      <path className="lp-path-ray" d={`M${x0} 31 Q ${xl} 13 ${xs} 31`} />
      <path className="lp-path-ray" d={`M${x0} 31 Q ${xl} 49 ${xs} 31`} />
      <circle cx={x0} cy={31} r={3.5} className="lp-path-you" />
      <ellipse cx={xl} cy={31} rx={4} ry={9} className="lp-path-lens" />
      <circle cx={xs} cy={31} r={4.5} className="lp-path-source" />
      <text className="lp-path-z" x={xl} y={10} textAnchor="middle">z {zOf(lens)}</text>
      <text className="lp-path-z" x={xs} y={10} textAnchor="end">z {zOf(source)}</text>
      <text className="lp-path-label" x={x0} y={57} textAnchor="start">you</text>
      <text className="lp-path-label" x={xl} y={57} textAnchor="middle">lens</text>
      <text className="lp-path-label" x={xs} y={57} textAnchor="end">source</text>
    </svg>
  );
}
