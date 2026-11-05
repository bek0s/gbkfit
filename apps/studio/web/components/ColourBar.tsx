// The colour bar of a map: its ramp upright, from the low end of its range
// at the bottom to the high end at the top, and its unit. Its ramp is as
// stretched on the map (see Stretch). A click on it opens a menu that
// resets the stretch, and, of a sequential ramp, chooses the colour map of
// all the maps (each with its swatch), and inverts it (see useColourMap).
import { COLOUR_MAPS, isStretched, mapGradient, rampGradient, setColourMap, UNSTRETCHED, useColourMap, useScheme, type Stretch } from './colours';
import { significant } from './format';
import { DropMenu, type MenuEntry } from './ui/overlays';

export function ColourBar({
  range,
  diverging,
  unit,
  horizontal = false,
  stretch = UNSTRETCHED,
  onReset,
}: {
  range: [number, number];
  diverging: boolean;
  unit: string;
  // (lying under a map, from low on the left to high on the right)
  horizontal?: boolean;
  stretch?: Stretch;
  onReset?: () => void;
}) {
  // (in the colours of the theme and scheme of the page, and the colour map
  // chosen, as they change)
  useScheme();
  const chosen = useColourMap();
  const ramp = (direction: string) => {
    const background = rampGradient(diverging, direction, stretch);
    const entries: MenuEntry[] = [
      ...(onReset && isStretched(stretch) ? [{ label: 'Reset stretch', onClick: onReset }] : []),
      ...(diverging
        ? []
        : [
            { section: 'Colour map' },
            ...COLOUR_MAPS.map(({ name, label }) => ({
              label,
              current: chosen.map === name,
              picture: <span className="colour-swatch" style={{ background: mapGradient({ ...chosen, map: name }) }} />,
              onClick: () => setColourMap({ ...chosen, map: name }),
            })),
            { section: 'Order' },
            { label: 'Inverted', current: chosen.inverted, onClick: () => setColourMap({ ...chosen, inverted: !chosen.inverted }) },
          ]),
    ];
    if (entries.length === 0) return <i style={{ background }} title="Two colours about the middle (a value such as 0): the same whatever the colour map" />;
    return (
      <DropMenu
        label="Colours"
        at={horizontal ? 'start' : 'side'}
        entries={entries}
        target={<button type="button" className="colour-ramp" style={{ background }} title={diverging ? 'Reset the stretch' : 'Choose the colour map'} aria-label="Colours" />}
      />
    );
  };
  const label = <span className="colour-unit">{unit}</span>;
  if (horizontal) {
    return (
      <div className="colour-bar horizontal">
        <span>{significant(range[0], 3)}</span>
        {ramp('to right')}
        <span>{significant(range[1], 3)}</span>
        {label}
      </div>
    );
  }
  return (
    <div className="colour-bar">
      <span>{significant(range[1], 3)}</span>
      {ramp('to top')}
      <span>{significant(range[0], 3)}</span>
      {label}
    </div>
  );
}
