// Maps on one grid (e.g. moment maps), to look at: one at a time, chosen on
// a rail at its left, with its colour bar, as large as the space left
// allows; its data, or the map of their errors or of their mask (its
// layers). The map has two tools (see MapTools): select (a drag selects a
// region) and pan (a drag moves the zoomed map). The selection (a region;
// never kept) shows as a chip in the head, and the map zooms to it (or
// with the wheel) and back; a crop (an edit) is outlined, and the pixels
// the edits leave out are blank on the data. Over its corner, its file;
// under it, the value of the pixel under the pointer.
import { useState, type ReactNode } from 'react';
import { useChannel } from '../api/queries';
import { ColourBar } from './ColourBar';
import { robustRange, useFitted, type Layer } from './CubeViewer';
import { significant } from './format';
import { MapTools, type Tool } from './MapTools';
import { MapView, type Region } from './MapView';
import { Chip, NavItem } from './ui/basics';
import { Segments } from './ui/inputs';

// A map: its key, its title on the rail (and its short name), how its
// colours run (sequential, diverging about its median, or about 0), and
// its layers (its data first)
export interface MapItem {
  key: string;
  title: string;
  short: string;
  colours: 'sequential' | 'median' | 'zero';
  layers: Layer[];
}

// The fewest pixels a side of the window of a zoomed map has
const LEAST = 8;

export function MapsViewer({
  maps,
  shape,
  selection,
  onSelection,
  crop = null,
  out = null,
  actions,
}: {
  maps: MapItem[];
  // The shape of the maps (x, y)
  shape: [number, number];
  selection: Region | null;
  onSelection: (selection: Region | null) => void;
  crop?: Region | null;
  // The pixels the edits leave out (1; null: none), of every map
  out?: Uint8Array | null;
  actions?: ReactNode;
}) {
  const [nx, ny] = shape;
  const [mapKey, setMapKey] = useState(maps[0]?.key);
  const [layerKey, setLayerKey] = useState<Layer['key']>('data');
  const [tool, setTool] = useState<Tool>('select');
  const [zoom, setZoom] = useState<Region | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  // (a map gone, the first; a layer it has not, its data)
  const map = maps.find((m) => m.key === mapKey) ?? maps[0];
  const layer = map.layers.find((l) => l.key === layerKey) ?? map.layers[0];
  const plane = useChannel(layer.path, layer.hdu, 0);
  const shownPixels = zoom ?? ([0, nx - 1, 0, ny - 1] as Region);
  const fitted = useFitted(shownPixels[1] - shownPixels[0] + 1, shownPixels[3] - shownPixels[2] + 1);
  const values = plane.data?.plane;
  // (the mask, 0 to 1; the data and the errors over the range of their
  // values, as the map's colours run)
  const colours = layer.key === 'data' ? map.colours : 'sequential';
  const range: [number, number] | undefined = values && (layer.key === 'mask' ? [0, 1] : robustRange(values.values, colours));
  const selecting = tool === 'select';
  // Zoom about a pixel by a factor, keeping it where it is on the screen;
  // out to the whole map, the window is none
  const zoomAt = (x: number, y: number, factor: number) => {
    const [x0, x1, y0, y1] = shownPixels;
    const w = Math.min(nx, Math.max(LEAST, Math.round((x1 - x0 + 1) / factor)));
    const h = Math.min(ny, Math.max(LEAST, Math.round((y1 - y0 + 1) / factor)));
    if (w >= nx && h >= ny) return setZoom(null);
    const left = Math.min(nx - w, Math.max(0, Math.round(x - ((x - x0) * w) / (x1 - x0 + 1))));
    const bottom = Math.min(ny - h, Math.max(0, Math.round(y - ((y - y0) * h) / (y1 - y0 + 1))));
    setZoom([left, left + w - 1, bottom, bottom + h - 1]);
  };
  // Pan the window by pixels, within the map
  const panBy = (dx: number, dy: number) => {
    if (zoom === null) return;
    const [x0, x1, y0, y1] = zoom;
    const mx = Math.min(nx - 1 - x1, Math.max(-x0, dx));
    const my = Math.min(ny - 1 - y1, Math.max(-y0, dy));
    setZoom([x0 + mx, x1 + mx, y0 + my, y1 + my]);
  };
  const hovered = hover && values ? values.values[hover[1] * nx + hover[0]] : undefined;
  return (
    <div className="card viewer">
      <div className="card-head viewer-head">
        {map.layers.length > 1 ? (
          <Segments items={map.layers.map((l) => ({ value: l.key, label: l.title }))} value={layer.key} onChange={setLayerKey} label="The map shown" />
        ) : (
          <h2>{map.title}</h2>
        )}
        <div className="selection-chips" aria-label="The selection">
          {selection !== null && (
            <Chip title="The region selected on the map" removeLabel="Clear the region" onRemove={() => onSelection(null)}>
              x {selection[0]}–{selection[1]} · y {selection[2]}–{selection[3]}
            </Chip>
          )}
        </div>
        {actions}
      </div>
      <div className="viewer-body">
        <nav className="rail" aria-label="The maps">
          <div className="rail-label">maps</div>
          {maps.map((m) => (
            <NavItem key={m.key} label={m.title} left={<span className="order">{m.short}</span>} active={m.key === map.key} onClick={() => setMapKey(m.key)} />
          ))}
        </nav>
        <div className="viewer-main">
          <div className="viewer-stage" ref={fitted.stage}>
            {plane.error && <p className="error">{plane.error.message}</p>}
            {values && range && fitted.size && (
              <div className="viewer-map">
                <div className="viewer-canvas" style={{ width: fitted.size[0], height: fitted.size[1] }}>
                  <MapView
                    values={values.values}
                    shape={values.shape}
                    range={range}
                    diverging={colours !== 'sequential'}
                    out={layer.key === 'data' ? out : null}
                    window={zoom}
                    drag={selecting ? 'region' : 'pan'}
                    region={selection}
                    outline={crop}
                    onRegion={selecting ? onSelection : undefined}
                    onZoom={zoomAt}
                    onPan={panBy}
                    onHover={setHover}
                    label={`The ${layer.title.toLowerCase()} of ${map.title}`}
                  />
                  <div className="map-source" title={layer.path}>
                    {layer.path.split('/').pop()}
                  </div>
                  <MapTools
                    tool={tool}
                    onTool={setTool}
                    onZoomToSelection={selection ? () => setZoom(selection) : undefined}
                    onWhole={zoom ? () => setZoom(null) : undefined}
                  />
                </div>
                <ColourBar range={range} diverging={colours !== 'sequential'} unit={layer.unit ?? ''} />
              </div>
            )}
          </div>
          <div className="map-status">
            {hover !== null
              ? `pixel ${hover[0]}, ${hover[1]} · ${hovered === undefined || !Number.isFinite(hovered) ? 'no value' : `${significant(hovered, 4)} ${layer.unit ?? ''}`}`
              : ''}
          </div>
        </div>
      </div>
    </div>
  );
}
