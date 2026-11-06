// A map of values as every viewer of the Studio shows it, the same on every
// screen (the data of a dataset, a channel or a moment of a cube, a galaxy
// as it is or as an observation sees it, the data of an observation): the
// map, as large as its place allows, with its colour bar at its side, its
// tools over its top right corner, and under it one line, always there: at
// the left, what the pointer is over (its pixel, its place on the sky in
// arcsec from the reference pixel, as xpos and ypos are measured, and its
// value), else the size of what is shown; at the right, the selection
// with the statistics of the map in it, and a button that clears it.
//
// Its gestures (see MapView): the wheel zooms about the pointer, a drag
// pans the map zoomed, a drag with the right button stretches its
// colours, a double click shows it whole, a click picks a pixel (if the
// viewer has a use for one). The select tool (a button, pressed while it
// is on) makes a drag draw a box instead: one box, the selection, which
// the next replaces; the button at its statistics, or Esc, clears it.
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { ColourBar } from './ColourBar';
import { UNSTRETCHED, useStretches } from './colours';
import { significant } from './format';
import { ClearIcon, SelectIcon, ZoomSelectionIcon, ZoomWholeIcon } from './icons';
import { MapView, rescale, useZoom, type Region } from './MapView';
import { regionStats, statsText } from './regions';
import type { SkyFrame } from './sky';
import { IconButton } from './ui/basics';

// The width of the side of a map (its colour bar), with its gap, and the
// height of the line under it, with its gap
const SIDE = 76;
const UNDER = 46;

const GESTURES = 'Wheel: zoom · drag: pan · right drag: colours · double click: whole';

export interface MapPanelProps {
  // The values drawn (ny by nx; none yet: the note shows instead), the
  // shape of the map (ny, nx), the range of the colours, and whether they
  // diverge about its middle
  values: Float32Array | null;
  shape: [number, number];
  range: [number, number];
  diverging?: boolean;
  // The values read under the pointer and in the selection, if not those
  // drawn (e.g. a light drawn on a scale), and their unit
  read?: Float32Array;
  unit: string;
  // The range of the colour bar, if not that of the colours (e.g. a light
  // drawn on a scale), and what goes under it
  bar?: [number, number];
  side?: ReactNode;
  // (the stretches of the colours are kept for each key, e.g. each map)
  stretchKey: string;
  // The pixels on the sky (none: not known), and whether the pixels are
  // worth naming (not of a render, whose pixels change with its size)
  frame?: SkyFrame | null;
  pixels?: boolean;
  // The selection (one box of pixels; none if not given: of the map
  // alone), and how it changes
  selection?: Region | null;
  onSelection?: (selection: Region | null) => void;
  // A pixel picked (marked), and what a click picks (none: nothing)
  mark?: [number, number];
  onPick?: (x: number, y: number) => void;
  // (the pixel under the pointer, as it moves)
  onHover?: (pixel: [number, number] | null) => void;
  kept?: [number | null, number | null] | null;
  out?: Uint8Array | null;
  outline?: Region | null;
  contour?: string | null;
  // (drawn smoothly: a render, rather than pixels of data)
  smooth?: boolean;
  label: string;
  // What shows in place of the map while it has no values, what goes over
  // its top left corner (e.g. its file), and what is drawn over it (of the
  // size of the map on the screen, and the pixels shown)
  note?: ReactNode;
  tags?: ReactNode;
  over?: (size: [number, number], shown: Region) => ReactNode;
}

export function MapPanel({
  values,
  shape,
  range,
  diverging = false,
  read,
  unit,
  bar,
  side,
  stretchKey,
  frame = null,
  pixels = true,
  selection,
  onSelection,
  mark,
  onPick,
  onHover,
  kept = null,
  out = null,
  outline = null,
  contour = null,
  smooth = false,
  label,
  note,
  tags,
  over,
}: MapPanelProps) {
  const [ny, nx] = shape;
  const { zoomed, whole, zoomAt, zoomTo, panBy, shown } = useZoom(nx, ny);
  // (a square, the same for every map, whatever its shape, and as it
  // zooms: see useZoom)
  const fitted = useFitted(1, 1, SIDE, UNDER);
  const [stretch, setStretch] = useStretches(stretchKey);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [selecting, setSelecting] = useState(false);
  // (a selection of the map alone, if the viewer keeps none: kept with
  // the shape it is of, e.g. of a render made again at another size)
  const [own, setOwn] = useState<{ box: Region; of: [number, number] } | null>(null);
  const controlled = onSelection !== undefined;
  const selected = controlled ? (selection ?? null) : own && rescale(own.box, own.of, [nx, ny]);
  const select = (box: Region | null) => (controlled ? onSelection(box) : setOwn(box && { box, of: [nx, ny] }));
  // (Esc clears it)
  const clear = useRef(() => {});
  clear.current = () => select(null);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && clear.current();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const readValues = read ?? values;
  const hovered = (at: [number, number]) => {
    const value = readValues?.[at[1] * nx + at[0]];
    return value === undefined || !Number.isFinite(value) ? 'no value' : `${significant(value, 4)}${unit ? ` ${unit}` : ''}`;
  };
  // (the part of the map shown, within it)
  const [w, h] = [Math.min(nx - 1, shown[1]) - Math.max(0, shown[0]) + 1, Math.min(ny - 1, shown[3]) - Math.max(0, shown[2]) + 1];
  return (
    <div className="map-panel">
      <div className="viewer-stage" ref={fitted.stage}>
        {!values && note}
        {values && fitted.size && (
          <div className="map-block" style={{ width: fitted.size[0] + SIDE }}>
          <div className="viewer-map">
            <div className="viewer-canvas" data-smooth={smooth || undefined} style={{ width: fitted.size[0], height: fitted.size[1] }}>
              <MapView
                values={values}
                shape={shape}
                range={range}
                diverging={diverging}
                kept={kept}
                out={out}
                window={shown}
                mark={mark}
                region={selected}
                outline={outline}
                contour={contour}
                onPick={onPick}
                onRegion={selecting ? select : undefined}
                onZoom={zoomAt}
                onPan={panBy}
                onWhole={whole}
                onHover={(at) => {
                  setHover(at);
                  onHover?.(at);
                }}
                stretch={stretch}
                onStretch={setStretch}
                label={label}
              />
              {over?.(fitted.size, shown)}
              {tags && <div className="map-tags">{tags}</div>}
              <div className="map-tools" title={GESTURES}>
                <IconButton label={selecting ? 'Select: on (a drag draws a box)' : 'Select a box of the map'} pressed={selecting} onClick={() => setSelecting(!selecting)}>
                  <SelectIcon />
                </IconButton>
                <span className="map-tools-gap" />
                <IconButton label="Zoom to the selection" disabled={selected === null} onClick={() => selected && zoomTo(selected)}>
                  <ZoomSelectionIcon />
                </IconButton>
                <IconButton label="The whole map (double click)" disabled={!zoomed} onClick={whole}>
                  <ZoomWholeIcon />
                </IconButton>
              </div>
            </div>
            <div className="map-side" style={{ height: fitted.size[1] }}>
              <ColourBar range={bar ?? range} diverging={diverging} unit={unit} stretch={stretch} onReset={() => setStretch(UNSTRETCHED)} />
              {side}
            </div>
          </div>
          <div className="map-status">
            <span className="map-status-at" data-hover={hover !== null || undefined}>
              {hover !== null ? (
                <>
                  {pixels && <span>Pixel {hover[0]}, {hover[1]} ·</span>}
                  {frame && <span title="Arcsec on the sky from the reference pixel, as the centre of a galaxy (xpos, ypos) is set">{skyText(hover, frame)} ·</span>}
                  <b>{hovered(hover)}</b>
                </>
              ) : (
                <span>{sizeText(w, h, frame, pixels)}</span>
              )}
            </span>
            <span className="map-status-selection">
              {selected !== null ? (
                <>
                  <span>{selectionText(selected, readValues, nx, unit, pixels ? null : frame)}</span>
                  <IconButton label="Clear the selection (Esc)" onClick={() => select(null)}>
                    <ClearIcon />
                  </IconButton>
                </>
              ) : (
                selecting && <span className="map-status-hint">Drag a box on the map</span>
              )}
            </span>
          </div>
          </div>
        )}
      </div>
    </div>
  );
}

// (a pixel on the sky: arcsec from the reference pixel)
function skyText([x, y]: [number, number], { step, rpix }: SkyFrame): string {
  return `(${signed((x - rpix[0]) * step[0])}″, ${signed((y - rpix[1]) * step[1])}″)`;
}

// (the selection: its size, in pixels, or else on the sky; and the
// statistics of the values in it)
function selectionText(box: Region, values: Float32Array | null | undefined, nx: number, unit: string, sky: SkyFrame | null): string {
  const [w, h] = [box[1] - box[0] + 1, box[3] - box[2] + 1];
  const size = sky ? sizeText(w, h, sky, false) : `${w} × ${h} px`;
  return `Selection ${size}${values ? ` · ${statsText(regionStats(values, nx, box), unit, w * h)}` : ''}`;
}

function sizeText(w: number, h: number, frame: SkyFrame | null, pixels: boolean): string {
  const sky = frame && `${significant(w * Math.abs(frame.step[0]), 3)}″ × ${significant(h * Math.abs(frame.step[1]), 3)}″`;
  return [pixels && `${w} × ${h} px`, sky].filter(Boolean).join(' · ');
}

function signed(value: number): string {
  const text = significant(Math.abs(value), 3);
  return value < -1e-9 ? `−${text}` : text;
}

// The size of a map of nx by ny pixels in the space of its stage, less that
// of its side and of what is under it (undefined until the stage is
// measured)
export function useFitted(nx: number, ny: number, side = SIDE, under = 0) {
  const stage = useRef<HTMLDivElement>(null);
  const [space, setSpace] = useState<[number, number] | null>(null);
  useLayoutEffect(() => {
    const element = stage.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setSpace([entry.contentRect.width, entry.contentRect.height]));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  if (space === null) return { stage, size: undefined };
  const scale = Math.max(0, Math.min((space[0] - side) / nx, (space[1] - under) / ny));
  return { stage, size: [Math.floor(nx * scale), Math.floor(ny * scale)] as [number, number] };
}

// The range of the colours of a map: from the 1st to the 99th percentile
// of its values; for one diverging, about its median or 0, as wide on
// each side
export function robustRange(values: Float32Array, colours: 'sequential' | 'median' | 'zero'): [number, number] {
  const sorted = Array.from(values.filter(Number.isFinite)).sort((a, b) => a - b);
  if (sorted.length === 0) return [0, 1];
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const [low, high] = [at(0.01), at(0.99)];
  if (colours === 'sequential') return [low, high];
  const middle = colours === 'median' ? at(0.5) : 0;
  const half = Math.max(middle - low, high - middle) || 1;
  return [middle - half, middle + half];
}
