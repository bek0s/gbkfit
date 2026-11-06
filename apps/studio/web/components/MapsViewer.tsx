// Maps on one grid (e.g. moment maps), to look at: one at a time, chosen in
// its head (as segments, or of many, from a list); its data, or the map of
// their errors or of their mask (its layers); in a map panel (see
// MapPanel: as every map of the Studio). A crop (an edit) is outlined, and
// the pixels the edits leave out are blank on the data. Over its corner,
// its file.
import { useState, type ReactNode } from 'react';
import { useChannel } from '../api/queries';
import type { Layer } from './CubeViewer';
import { MapPanel, robustRange } from './MapPanel';
import type { Region } from './MapView';
import type { SkyFrame } from './sky';
import { Choice, Segments } from './ui/inputs';

// A map: its key, its title, how its colours run (sequential, diverging
// about its median, or about 0), and its layers (its data first)
export interface MapItem {
  key: string;
  title: string;
  colours: 'sequential' | 'median' | 'zero';
  layers: Layer[];
}

export function MapsViewer({
  maps,
  shape,
  selection,
  onSelection,
  crop = null,
  out = null,
  actions,
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
  // The pixels of the maps on the sky (none: not known)
  frame?: SkyFrame | null;
}) {
  const [nx, ny] = shape;
  const [mapKey, setMapKey] = useState(maps[0]?.key);
  const [layerKey, setLayerKey] = useState<Layer['key']>('data');
  // (a map gone, the first; a layer it has not, its data)
  const map = maps.find((m) => m.key === mapKey) ?? maps[0];
  const layer = map.layers.find((l) => l.key === layerKey) ?? map.layers[0];
  const plane = useChannel(layer.path, layer.hdu, 0);
  const values = plane.data?.plane;
  // (the mask, 0 to 1; the data and the errors over the range of their
  // values, as the map's colours run)
  const colours = layer.key === 'data' ? map.colours : 'sequential';
  const range: [number, number] = values ? (layer.key === 'mask' ? [0, 1] : robustRange(values.values, colours)) : [0, 1];
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
        <span className="viewer-head-gap" />
        {actions}
      </div>
      <div className="viewer-body">
        <div className="viewer-main">
          <MapPanel
            values={values?.values ?? null}
            shape={[ny, nx]}
            range={range}
            diverging={colours !== 'sequential'}
            unit={layer.unit ?? ''}
            stretchKey={`${map.key}/${layer.key}`}
            frame={frame}
            selection={selection}
            onSelection={onSelection}
            out={layer.key === 'data' ? out : null}
            outline={crop}
            label={`The ${layer.title.toLowerCase()} of ${map.title}`}
            note={plane.error ? <p className="error">{plane.error.message}</p> : null}
            tags={
              <span className="map-source" title={layer.path}>
                {layer.path.split('/').pop()}
              </span>
            }
          />
        </div>
      </div>
    </div>
  );
}
