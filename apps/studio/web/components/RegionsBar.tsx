// The regions of the sky of the project on a map (see sky.ts; kept by the
// server, shared by every map of the project): drawn on it (a drag adds
// one) and listed beside it, each with the statistics of the map in it.
// One is the selection: the last drawn, or one chosen in the list; the map
// dims the rest of it, and what follows the selection (e.g. a total
// spectrum, a crop) is of it. Esc clears the selection; × removes a region
// from the project. A map without coordinates on the sky has a selection
// of its own alone, not kept.
import { useEffect, useState } from 'react';
import { useRegions, useSetRegions, type SkyRegion } from '../api/queries';
import type { Region } from './MapView';
import { regionStats, statsText } from './regions';
import { toPixels, toSky, type SkyFrame } from './sky';
import { Chip } from './ui/basics';

export interface Placed {
  id: string;
  name: string;
  // (its pixels on this map; none if it is beyond it)
  pixels: Region | null;
}

export function useSkyRegions(project: string | null, frame: SkyFrame | null, shape: [number, number]) {
  const kept = useRegions(frame ? project : null).data ?? [];
  const set = useSetRegions(project);
  const [activeId, setActiveId] = useState<string | null>(null);
  // (of a map without coordinates: its own, not kept)
  const [own, setOwn] = useState<Region | null>(null);
  const placed: Placed[] = frame ? kept.map((region) => ({ id: region.id, name: region.name, pixels: toPixels(region.box, frame, shape) })) : own ? [{ id: 'own', name: 'Selection', pixels: own }] : [];
  const active = placed.find((region) => region.id === (frame ? activeId : 'own') && region.pixels !== null) ?? null;
  const add = (pixels: Region) => {
    if (!frame) return setOwn(pixels);
    const id = crypto.randomUUID();
    set.mutate([...kept, { id, name: freshName(kept), box: toSky(pixels, frame) }]);
    setActiveId(id);
  };
  const remove = (id: string) => {
    if (!frame) return setOwn(null);
    set.mutate(kept.filter((region) => region.id !== id));
    if (activeId === id) setActiveId(null);
  };
  const choose = (id: string | null) => (frame ? setActiveId(id) : id === null && setOwn(null));
  // (Esc clears the selection)
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && (frame ? setActiveId(null) : setOwn(null));
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [frame]);
  return { placed, active, add, remove, choose };
}

// (R1, R2, ...: the first not taken)
function freshName(regions: SkyRegion[]): string {
  const names = new Set(regions.map((region) => region.name));
  let n = regions.length + 1;
  while (names.has(`R${n}`)) n += 1;
  return `R${n}`;
}

// The regions on a map, as chips: each its name, and the statistics of the
// map in it (of its values, nx across, in a unit); a click chooses it as
// the selection (again: none), × removes it
export function RegionsBar({
  regions,
  values,
  nx,
  unit,
}: {
  regions: ReturnType<typeof useSkyRegions>;
  values: ArrayLike<number> | undefined;
  nx: number;
  unit: string;
}) {
  const on = regions.placed.filter((region) => region.pixels !== null);
  if (on.length === 0) return null;
  return (
    <div className="regions-bar" aria-label="The regions on the map">
      {on.map((region) => {
        const active = regions.active?.id === region.id;
        return (
          <Chip
            key={region.id}
            title={active ? 'The selection: a click lets it go (Esc too)' : 'A click makes it the selection'}
            removeLabel={`Remove the region ${region.name}`}
            onRemove={() => regions.remove(region.id)}
          >
            <button type="button" className="region-chip" data-active={active || undefined} onClick={() => regions.choose(active ? null : region.id)}>
              <b>{region.name}</b>
              {values && ` ${statsText(regionStats(values, nx, region.pixels!), unit)}`}
            </button>
          </Chip>
        );
      })}
    </div>
  );
}
