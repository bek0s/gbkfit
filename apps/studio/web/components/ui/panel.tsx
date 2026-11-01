// A panel (a Paper of Mantine; see theme.ts) that folds to its head (a
// click on its title), as the page remembers by its key; its actions (e.g.
// a clear) stay in its head while it is open.
import { Collapse, Paper, UnstyledButton } from '@mantine/core';
import type { ReactNode } from 'react';
import { useRemembered } from '../../shell/remember';

export function Panel({
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
    <Paper withBorder className="card ui-panel" data-folded={folded}>
      <div className="card-head ui-panel-head">
        <UnstyledButton className="ui-panel-title" aria-expanded={!folded} onClick={() => setFolded(!folded)}>
          <span className="ui-chevron" data-opened={!folded}>
            ▶
          </span>
          <h2>{title}</h2>
        </UnstyledButton>
        {!folded && actions}
      </div>
      <Collapse expanded={!folded}>
        <div className={`card-body ${bodyClass}`}>{children}</div>
      </Collapse>
    </Paper>
  );
}
