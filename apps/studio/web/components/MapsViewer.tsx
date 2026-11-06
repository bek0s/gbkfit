// Maps on one grid (e.g. moment maps), to look at: one at a time, chosen in
// its head (as segments, or of many, from a list), with its colour bar, as
// large as the space left
// allows; its data, or the map of their errors or of their mask (its
// layers). Its gestures need no tool (see MapView): a drag selects a
// region, a drag with the middle button or Space held pans. The selection (a region;
// never kept) shows as a chip in the head, and the map zooms to it (or
// with the wheel) and back; a crop (an edit) is outlined, and the pixels
// the edits leave out are blank on the data. Over its corner, its file;
// under it, the value of the pixel under the pointer.
import { useEffect, useState, type ReactNode } from 'react';
import { useChannel } from '../api/queries';
import { ColourBar } from './ColourBar';
import { UNSTRETCHED, useStretches } from './colours';
import { robustRange, useFitted, type Layer } from './CubeViewer';
import { significant } from './format';
import { MapTools } from './MapTools';
import { RegionsBar, useSkyRegions } from './RegionsBar';
import type { SkyFrame } from './sky';
import { MapView, type Region } from './MapView';
import { Choice, Segments } from './ui/inputs';

// A map: its key, its title, how its colours run (sequential, diverging
// about its median, or about 0), and its layers (its data first)
export interface MapItem {
  key: string;
  title: string;
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
  project = null,
  frame = null,
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
  // The project, whose regions of the sky (see RegionsBar) are on the map,
  // and the pixels of the map on the sky (none: a selection of its own)
  project?: string | null;
  frame?: SkyFrame | null;
}) {
  const [nx, ny] = shape;
  // (the regions of the sky on it; the one chosen is the selection)
  const regions = useSkyRegions(project, frame, [nx, ny]);
  const activePixels = regions.active?.pixels ?? null;
  const activeKey = activePixels ? activePixels.join() : '';
  useEffect(() => {
    onSelection(activePixels);
  }, [activeKey]);
  const [mapKey, setMapKey] = useState(maps[0]?.key);
  const [layerKey, setLayerKey] = useState<Layer['key']>('data');
  const [zoom, setZoom] = useState<Region | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);
  // (a map gone, the first; a layer it has not, its data)
  const map = maps.find((m) => m.key === mapKey) ?? maps[0];
  const layer = map.layers.find((l) => l.key === layerKey) ?? map.layers[0];
  // (the colours of each, as the right button stretches them on the map)
  const [stretch, setStretch] = useStretches(`${map.key}/${layer.key}`);
  const plane = useChannel(layer.path, layer.hdu, 0);
  const shownPixels = zoom ?? ([0, nx - 1, 0, ny - 1] as Region);
  const fitted = useFitted(shownPixels[1] - shownPixels[0] + 1, shownPixels[3] - shownPixels[2] + 1);
  const values = plane.data?.plane;
  // (the mask, 0 to 1; the data and the errors over the range of their
  // values, as the map's colours run)
  const colours = layer.key === 'data' ? map.colours : 'sequential';
  const range: [number, number] | undefined = values && (layer.key === 'mask' ? [0, 1] : robustRange(values.values, colours));
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
    <div className="viewer">
      <div className="viewer-head">
        {maps.length > 4 ? (
          <Choice items={maps.map((m) => ({ value: m.key, label: m.title }))} value={map.key} onChange={setMapKey} label="The map shown" />
        ) : maps.length > 1 ? (
          <Segments items={maps.map((m) => ({ value: m.key, label: m.title }))} value={map.key} onChange={setMapKey} label="The map shown" />
        ) : (
          <h2>{map.title}</h2>
        )}
        {map.layers.length > 1 && (
          <Segments items={map.layers.map((l) => ({ value: l.key, label: l.title }))} value={layer.key} onChange={setLayerKey} label="The layer of the map shown" />
        )}
        <div className="selection-chips" aria-label="The selection">
          <RegionsBar regions={regions} values={values?.values} nx={nx} unit={layer.unit ?? ''} />
        </div>
        {actions}
      </div>
      <div className="viewer-body">
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
                    region={selection}
                    regions={regions.placed.filter((r) => r.pixels && r.id !== regions.active?.id).map((r) => r.pixels!)}
                    outline={crop}
                    onRegion={(region) => {
                      regions.add(region);
                      onSelection(region);
                    }}
                    onZoom={zoomAt}
                    onPan={panBy}
                    onWhole={() => setZoom(null)}
                    onHover={setHover}
                    stretch={stretch}
                    onStretch={setStretch}
                    label={`The ${layer.title.toLowerCase()} of ${map.title}`}
                  />
                  <div className="map-source" title={layer.path}>
                    {layer.path.split('/').pop()}
                  </div>
                  <MapTools
                    onZoomToSelection={selection ? () => setZoom(selection) : undefined}
                    onWhole={zoom ? () => setZoom(null) : undefined}
                  />
                </div>
                <div className="map-side" style={{ height: fitted.size[1] }}>
                  <ColourBar
                    range={range}
                    diverging={colours !== 'sequential'}
                    unit={layer.unit ?? ''}
                    stretch={stretch}
                    onReset={() => setStretch(UNSTRETCHED)}
                  />
                </div>
              </div>
            )}
          </div>
          <div className="map-status">
            {hover !== null
              ? `Pixel ${hover[0]}, ${hover[1]} · ${hovered === undefined || !Number.isFinite(hovered) ? 'no value' : `${significant(hovered, 4)} ${layer.unit ?? ''}`}`
              : ''}
          </div>
        </div>
      </div>
    </div>
  );
}
