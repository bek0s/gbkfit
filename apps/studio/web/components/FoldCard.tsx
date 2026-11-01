// A card that folds to its head (a click on its title), as the page
// remembers by its key; its actions (e.g. a link) stay in its head.
import type { ReactNode } from 'react';
import { useRemembered } from '../shell/remember';

export function FoldCard({
  id,
  title,
  actions,
  bodyClass = '',
  children,
}: {
  id: string;
  title: string;
  actions?: ReactNode;
  bodyClass?: string;
  children: ReactNode;
}) {
  const [folded, setFolded] = useRemembered(`fold:${id}`, false);
  return (
    <div className="card" data-folded={folded}>
      <div className="card-head fold-head">
        <button className="fold-title" aria-expanded={!folded} onClick={() => setFolded(!folded)}>
          <span className="chevron">▶</span>
          <h2>{title}</h2>
        </button>
        {!folded && actions}
      </div>
      {!folded && <div className={`card-body ${bodyClass}`}>{children}</div>}
    </div>
  );
}
