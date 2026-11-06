// The zoom of a map, as icons over its corner: to the selection, or out to
// the whole map (as a double click). Its other gestures need no tool (see
// MapView): the wheel zooms about the pointer, a drag draws a region, a
// drag with the middle button or Space held pans, and one with the right
// button stretches the colours; the title of the corner says so.
import { ZoomSelectionIcon, ZoomWholeIcon } from './icons';
import { IconButton } from './ui/basics';

export const GESTURES = 'Drag: a region · wheel: zoom · middle drag or Space drag: pan · right drag: colours · double click: whole';

export function MapTools({
  onZoomToSelection,
  onWhole,
}: {
  // How the map zooms to the selection, and out to the whole map (none:
  // nothing selected, or not zoomed)
  onZoomToSelection?: () => void;
  onWhole?: () => void;
}) {
  return (
    <div className="map-tools" title={GESTURES}>
      <IconButton label="Zoom to the selection" disabled={onZoomToSelection === undefined} onClick={() => onZoomToSelection?.()}>
        <ZoomSelectionIcon />
      </IconButton>
      <IconButton label="The whole map (double click)" disabled={onWhole === undefined} onClick={() => onWhole?.()}>
        <ZoomWholeIcon />
      </IconButton>
    </div>
  );
}
