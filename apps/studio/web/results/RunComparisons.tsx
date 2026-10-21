// The data, models and residuals of a run side by side, by data item (and
// by image of one, e.g. the moment maps of a cube): their arrays are read
// from its files, a plane at a time for cubes (to keep large cubes light).
import { useQueries } from '@tanstack/react-query';
import { useState } from 'react';
import { api, data, type Run } from '../api/client';
import { decode, type FrameArray } from '../api/frames';
import { Comparison } from '../components/Comparison';

type Product = Run['products'][number];

// The columns of a data item, by the kinds of its products (those of its
// images end in _image)
const COLUMNS = ['data', 'model', 'residual'];

export function RunComparisons({ run, products }: { run: string; products: Product[] }) {
  const groups = new Map<string, Product[]>();
  for (const product of products) {
    const column = String(product.kind).replace(/_image$/, '');
    if (!COLUMNS.includes(column)) continue;
    const ids = product.ids as Record<string, string>;
    const key = `${ids.observation} · ${ids.item ?? ids.image}`;
    groups.set(key, [...(groups.get(key) ?? []), product]);
  }
  return (
    <>
      {[...groups].map(([key, group]) => (
        <div className="row-item" key={key}>
          <div className="map-head">{key}</div>
          <RunComparison run={run} products={sortByColumn(group)} />
        </div>
      ))}
    </>
  );
}

function RunComparison({ run, products }: { run: string; products: Product[] }) {
  const shape = (products[0].shape as number[] | undefined) ?? [];
  const planes = shape.length === 3 ? shape[0] : 0;
  const [plane, setPlane] = useState(Math.floor(planes / 2));
  const arrays = useQueries({
    queries: products.map((product) => {
      const path = `${run}/${String(product.path)}`;
      const query = planes > 0 ? { plane } : {};
      return {
        queryKey: ['array', path, query],
        queryFn: async () =>
          decode(
            await data(
              api.GET('/api/v1/arrays/{path}', {
                params: { path: { path }, query },
                parseAs: 'arrayBuffer',
              }),
            ),
          ).arrays[0],
        placeholderData: (previous: FrameArray | undefined) => previous,
      };
    }),
  });
  const failed = arrays.find((a) => a.error);
  if (failed) return <p className="error">{failed.error!.message}</p>;
  if (arrays.some((a) => !a.data)) return <p className="hint">Reading…</p>;
  const views = products.map((product, i) => {
    const column = String(product.kind).replace(/_image$/, '');
    return { label: column, array: arrays[i].data!, diverging: column === 'residual' };
  });
  return (
    <>
      <Comparison views={views} />
      {planes > 0 && (
        <label className="plane">
          <span>plane</span>
          <input
            type="range"
            min={0}
            max={planes - 1}
            value={plane}
            onChange={(event) => setPlane(Number(event.target.value))}
          />
          <span className="num">
            {plane + 1} / {planes}
          </span>
        </label>
      )}
    </>
  );
}

function sortByColumn(products: Product[]): Product[] {
  const column = (p: Product) => COLUMNS.indexOf(String(p.kind).replace(/_image$/, ''));
  return [...products].sort((a, b) => column(a) - column(b));
}
