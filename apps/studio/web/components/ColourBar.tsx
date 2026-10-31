// The colour bar of a map: its ramp upright, from the low end of its range
// at the bottom to the high end at the top, and its unit.
import { rampGradient } from './colours';
import { significant } from './format';

export function ColourBar({ range, diverging, unit }: { range: [number, number]; diverging: boolean; unit: string }) {
  return (
    <div className="colour-bar">
      <span>{significant(range[1], 3)}</span>
      <i style={{ background: rampGradient(diverging, 'to top') }} />
      <span>{significant(range[0], 3)}</span>
      <span className="colour-unit">{unit}</span>
    </div>
  );
}
