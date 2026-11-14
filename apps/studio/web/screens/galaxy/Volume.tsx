// The light of a galaxy in 3d (see the server's /views/galaxy-volume),
// face-on so that z is along its axis: as slices along z (its plane, and
// two heights above it), each a map of the light at that height on one
// scale, so that the light falling off with height shows; and as a volume
// to turn (see Volume3D, loaded only as it is shown).
import { useElementSize } from '@mantine/hooks';
import { lazy, Suspense, useEffect, useRef } from 'react';
import type { GalaxyVolume } from '../../api/queries';
import { rampColour, rampStops, useColourMap } from '../../components/colours';
import { significant } from '../../components/format';

export const Volume3D = lazy(() => import('./Volume3D'));

// The volume in 3d, to turn, as it loads
export function VolumeView({ volume, error }: { volume: GalaxyVolume | undefined; error: Error | null }) {
  if (error) return <p className="gx-error gx-view-error">{error.message}</p>;
  if (!volume) return <div className="gx-map-wait" />;
  return (
    <Suspense fallback={<div className="gx-map-wait" />}>
      <Volume3D volume={volume} />
    </Suspense>
  );
}

// Slices of the volume along z: at its plane, and at two heights above it
// (where its light falls to a half and a tenth), on the scale of the
// brightest
export function Slices({ volume, error, kpcPerArcsec }: { volume: GalaxyVolume | undefined; error: Error | null; kpcPerArcsec: number | null }) {
  const { ref, width, height } = useElementSize();
  if (error) return <p className="gx-error gx-view-error">{error.message}</p>;
  if (!volume) return <div className="gx-map-wait" />;
  const [nz, ny, nx] = volume.shape;
  const zero = Math.round(Math.min(Math.max(volume.zero, 0), nz - 1));
  // (the plane, and above it the heights where the light of a plane falls
  // to half and to a tenth of that of the plane)
  const totals = Array.from({ length: nz }, (_, z) => {
    let sum = 0;
    for (let i = 0; i < nx * ny; i++) sum += Math.max(0, volume.values[z * nx * ny + i]);
    return sum;
  });
  const above = (fraction: number) => {
    for (let z = zero + 1; z < nz; z++) if (totals[z] <= fraction * totals[zero]) return z;
    return nz - 1;
  };
  const planes = (nz > 1 ? [zero, above(0.5), above(0.1)] : [zero]).filter((z, i, all) => all.indexOf(z) === i);
  let peak = 0;
  for (const z of planes) for (let i = 0; i < nx * ny; i++) peak = Math.max(peak, volume.values[z * nx * ny + i]);
  // (the slices side by side, as large as the view allows, square)
  const side = Math.max(60, Math.min((width - 12 * (planes.length - 1)) / planes.length, height - 28));
  return (
    <div className="gx-slices" ref={ref}>
      {planes.map((z) => {
        const lift = (z - zero) * volume.depthStep;
        const text = kpcPerArcsec === null ? `${significant(lift, 3)}″` : `${significant(lift * kpcPerArcsec, 3)} kpc`;
        return (
          <figure key={z} style={{ width: side }}>
            <Slice values={volume.values.subarray(z * nx * ny, (z + 1) * nx * ny)} nx={nx} ny={ny} peak={peak} side={side} />
            <figcaption>z = {z === zero ? '0' : text}</figcaption>
          </figure>
        );
      })}
    </div>
  );
}

// A slice as a picture, its light on a square-root scale of the peak
function Slice({ values, nx, ny, peak, side }: { values: Float32Array; nx: number; ny: number; peak: number; side: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const colourMap = useColourMap();
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const stops = rampStops(false);
    const image = context.createImageData(nx, ny);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const v = values[y * nx + x];
        // (the first row is the bottom of the sky: drawn at the bottom)
        const o = ((ny - 1 - y) * nx + x) * 4;
        const [r, g, b] = rampColour(stops, peak > 0 ? Math.sqrt(Math.max(0, v) / peak) : 0);
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [values, nx, ny, peak, colourMap]);
  return <canvas ref={canvas} width={nx} height={ny} style={{ width: side, height: side }} />;
}
