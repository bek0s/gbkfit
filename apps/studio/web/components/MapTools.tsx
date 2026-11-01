// The tools of a map, as icons over its corner: select (a drag selects a
// region) or pan (a drag moves the zoomed map); and zoom to the selection
// (the wheel zooms about the pointer) or out to the whole map.
import { PanIcon, SelectIcon, ZoomSelectionIcon, ZoomWholeIcon } from './icons';
import { IconButton } from './ui/basics';

export type Tool = 'select' | 'pan';

export function MapTools({
  tool,
  onTool,
  selectTitle = 'Select: a drag selects a region of the map',
  onZoomToSelection,
  onWhole,
}: {
  tool: Tool;
  onTool: (tool: Tool) => void;
  selectTitle?: string;
  // How the map zooms to the selection, and out to the whole map (none:
  // nothing selected, or not zoomed)
  onZoomToSelection?: () => void;
  onWhole?: () => void;
}) {
  return (
    <div className="map-tools">
      <IconButton label={selectTitle} pressed={tool === 'select'} onClick={() => onTool('select')}>
        <SelectIcon />
      </IconButton>
      <IconButton label="Pan: a drag moves the zoomed map" pressed={tool === 'pan'} onClick={() => onTool('pan')}>
        <PanIcon />
      </IconButton>
      <span className="map-tools-gap" />
      <IconButton
        label="Zoom to the selection (the wheel zooms about the pointer)"
        disabled={onZoomToSelection === undefined}
        onClick={() => onZoomToSelection?.()}
      >
        <ZoomSelectionIcon />
      </IconButton>
      <IconButton label="The whole map" disabled={onWhole === undefined} onClick={() => onWhole?.()}>
        <ZoomWholeIcon />
      </IconButton>
    </div>
  );
}
