// A position-velocity diagram of a galaxy (see the server's /views/pv): the
// light of its cube along a slit through its centre, at each position
// (arcsec along the slit, positive towards its position angle: its
// receding side) and velocity, on a square-root scale of the brightest,
// in the colours of the colour map; over it, the line of sight velocity
// of the rotation of its main component along its major axis (vsys + v(R)
// sin i, of the sign of the side), if given. A readout follows the
// pointer.
import { useElementSize } from '@mantine/hooks';
import { useEffect, useRef, useState } from 'react';
import type { PVDiagram } from '../../api/queries';
import { rampColour, rampStops, useColourMap } from '../../components/colours';
import { significant } from '../../components/format';
import { ticks } from './Curves';

const PAD = { left: 48, right: 14, top: 10, bottom: 24 };

export function PVPlot({ pv, model }: { pv: PVDiagram; model: { position: number; velocity: number }[] | null }) {
  const { ref, width, height } = useElementSize();
  const canvas = useRef<HTMLCanvasElement>(null);
  const colourMap = useColourMap();
  const [at, setAt] = useState<{ i: number; j: number } | null>(null);
  const npositions = pv.positions.length;
  const nchannels = pv.spectral.length;
  const scale = pv.step ?? 1;
  const unit = pv.step === null ? 'px' : '″';
  const positions = pv.positions.map((p) => p * scale);
  const [pLow, pHigh] = [positions[0] ?? 0, positions[npositions - 1] ?? 1];
  const [vLow, vHigh] = [Math.min(...pv.spectral), Math.max(...pv.spectral)];
  const plotW = Math.max(10, width - PAD.left - PAD.right);
  const plotH = Math.max(10, height - PAD.top - PAD.bottom);
  const x = (p: number) => PAD.left + ((p - pLow) / (pHigh - pLow || 1)) * plotW;
  const y = (v: number) => PAD.top + ((vHigh - v) / (vHigh - vLow || 1)) * plotH;

  // (the diagram as a picture: a pixel for each position and channel, the
  // highest velocity at the top)
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context || npositions === 0 || nchannels === 0) return;
    let peak = 0;
    for (const v of pv.values) if (Number.isFinite(v)) peak = Math.max(peak, v);
    const stops = rampStops(false);
    const image = context.createImageData(npositions, nchannels);
    const rising = pv.spectral[nchannels - 1] >= pv.spectral[0];
    for (let c = 0; c < nchannels; c++) {
      const row = rising ? nchannels - 1 - c : c;
      for (let i = 0; i < npositions; i++) {
        const v = pv.values[c * npositions + i];
        const o = (row * npositions + i) * 4;
        const [r, g, b] = rampColour(stops, peak > 0 && Number.isFinite(v) ? Math.sqrt(Math.max(0, v) / peak) : 0);
        image.data.set([r, g, b, Number.isFinite(v) ? 255 : 0], o);
      }
    }
    context.putImageData(image, 0, 0);
  }, [pv, npositions, nchannels, colourMap]);

  const modelPath = model
    ?.filter((m) => m.position >= pLow && m.position <= pHigh && Number.isFinite(m.velocity))
    .map((m, k) => `${k ? 'L' : 'M'}${x(m.position).toFixed(1)},${y(Math.min(vHigh, Math.max(vLow, m.velocity))).toFixed(1)}`)
    .join('');
  const read = at && { position: positions[at.i], velocity: pv.spectral[at.j], value: pv.values[at.j * npositions + at.i] };
  return (
    <div className="gx-plot gx-pv" ref={ref}>
      {/* (always there, so that it is drawn as the diagram comes) */}
      <canvas ref={canvas} width={npositions} height={nchannels} style={{ position: 'absolute', left: PAD.left, top: PAD.top, width: plotW, height: plotH, imageRendering: 'pixelated' }} />
      {width > 0 && height > 0 && (
        <>
          <svg
            width={width}
            height={height}
            style={{ position: 'absolute', inset: 0 }}
            onPointerMove={(event) => {
              const box = event.currentTarget.getBoundingClientRect();
              const fx = (event.clientX - box.left - PAD.left) / plotW;
              const fy = (event.clientY - box.top - PAD.top) / plotH;
              if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return setAt(null);
              const i = Math.min(npositions - 1, Math.round(fx * (npositions - 1)));
              const v = vHigh - fy * (vHigh - vLow);
              let j = 0;
              for (let c = 1; c < nchannels; c++) if (Math.abs(pv.spectral[c] - v) < Math.abs(pv.spectral[j] - v)) j = c;
              setAt({ i, j });
            }}
            onPointerLeave={() => setAt(null)}
          >
            <rect className="gx-pv-frame" x={PAD.left} y={PAD.top} width={plotW} height={plotH} />
            {ticks(vLow, vHigh, 4).map((t) => (
              <text key={`v${t}`} className="gx-tick" x={PAD.left - 6} y={y(t) + 3.5} textAnchor="end">
                {significant(t, 3)}
              </text>
            ))}
            {ticks(pLow, pHigh, Math.max(2, Math.floor(plotW / 70))).map((t) => (
              <text key={`p${t}`} className="gx-tick" x={x(t)} y={height - 7} textAnchor="middle">
                {significant(t, 3)}
                {unit}
              </text>
            ))}
            {modelPath && <path className="gx-pv-model" d={modelPath} />}
            {at && <line className="gx-pv-cursor" x1={x(positions[at.i])} x2={x(positions[at.i])} y1={PAD.top} y2={PAD.top + plotH} />}
          </svg>
          {read && (
            <div className="gx-curves-read" style={{ left: Math.min(width - 190, Math.max(0, x(read.position) + 10)) }}>
              <b>
                {significant(read.position, 3)}
                {unit} · {significant(read.velocity, 4)} {pv.unit ?? ''}
              </b>
              <span>
                light <em>{Number.isFinite(read.value) ? significant(read.value, 3) : '—'}</em>
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
