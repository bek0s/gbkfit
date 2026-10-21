// A run (see gbkfit.outputs): its status and its products, by kind; a
// product of data (FITS) shows as an image, with its title and description.
import { useQuery } from '@tanstack/react-query';
import type { IDockviewPanelProps } from 'dockview-react';
import { useState } from 'react';
import { api, data, type Run } from '../api/client';
import { decode } from '../api/frames';
import { ArrayView } from '../components/ArrayView';

type Product = Run['products'][number];

// The kinds of products whose values are about 0 (shown diverging)
const DIVERGING = new Set(['residual', 'wresidual']);

export function RunPanel({ params }: IDockviewPanelProps<{ path: string }>) {
  const { path } = params;
  const run = useQuery({
    queryKey: ['run', path],
    queryFn: () => data(api.GET('/api/v1/runs/{path}', { params: { path: { path } } })),
  });
  const [selected, setSelected] = useState<Product | null>(null);
  const products = run.data?.products ?? [];
  const kinds = [...new Set(products.map((p) => String(p.kind)))];
  return (
    <section className="panel">
      <div className="panel-head">
        <span className="kind">{path}</span>
        {run.data && <span className="badge">{run.data.task} · {run.data.status}</span>}
      </div>
      {run.error && <p className="error">{run.error.message}</p>}
      <div className="run">
        <div className="products">
          {kinds.map((kind) => (
            <div key={kind}>
              <div className="product-kind">{kind}</div>
              {products
                .filter((p) => p.kind === kind)
                .map((p) => (
                  <button
                    key={String(p.path)}
                    className={p === selected ? 'row file selected' : 'row file'}
                    onClick={() => setSelected(p)}
                    title={String(p.description ?? '')}
                  >
                    {String(p.title ?? p.path)}
                  </button>
                ))}
            </div>
          ))}
        </div>
        {selected && <ProductView run={path} product={selected} />}
      </div>
    </section>
  );
}

function ProductView({ run, product }: { run: string; product: Product }) {
  const path = `${run}/${String(product.path)}`;
  const isArray = product.format === 'fits';
  const array = useQuery({
    queryKey: ['array', path],
    enabled: isArray,
    queryFn: async () =>
      decode(
        await data(
          api.GET('/api/v1/arrays/{path}', { params: { path: { path } }, parseAs: 'arrayBuffer' }),
        ),
      ),
  });
  return (
    <div className="product">
      <div className="map-head">{String(product.title ?? product.path)}</div>
      {typeof product.description === 'string' && <p className="hint">{product.description}</p>}
      {!isArray && <p className="hint">{String(product.path)} ({String(product.format)})</p>}
      {array.error && <p className="error">{array.error.message}</p>}
      {array.data?.arrays.map((a, i) => (
        <ArrayView
          key={i}
          data={a.data}
          shape={a.info.shape}
          diverging={DIVERGING.has(String(product.kind))}
        />
      ))}
    </div>
  );
}
