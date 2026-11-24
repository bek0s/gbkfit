// The units of values as the Studio writes them (see unitText)

// The units as the Studio writes them, everywhere: by their symbols
// (arcsec as ″, deg as °, Angstrom as Å, um as µm), a scale as a power
// of ten (e.g. '1e-10 m / s2' as '×10⁻¹⁰ m/s²'), powers raised, products
// by a dot
export function unitText(unit: string | null): string {
  if (unit === null) return '';
  const known = { 'km / s': 'km/s', deg: '°', arcsec: '″', arcmin: '′', Angstrom: 'Å', um: 'µm', micron: 'µm', 'km / (Mpc s)': 'km/s/Mpc', solMass: 'M☉', 'dex(solMass)': 'dex M☉', 'solMass / solLum': 'M☉/L☉' }[unit];
  if (known) return known;
  const scaled = /^1e([+-]?\d+)\s+(.*)$/.exec(unit);
  const rest = (scaled ? scaled[2] : unit)
    .replace(/\s*\/\s*/g, '/')
    .replace(/\s+/g, '·')
    .replace(/solMass/g, 'M☉')
    .replace(/solLum/g, 'L☉')
    .replace(/([a-zA-Z])(-?\d+)/g, (_, base: string, power: string) => base + raised(power));
  return scaled ? `×10${raised(String(Number(scaled[1])))} ${rest}` : rest;
}

// A number as superscript digits
function raised(digits: string): string {
  return [...digits].map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)] ?? (c === '-' ? '⁻' : c)).join('');
}
