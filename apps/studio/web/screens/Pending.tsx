// A screen not built yet: its title, and that it comes in a later step;
// of a block of the project, its menu (rename, delete).
import type { ReactNode } from 'react';
import { BlockMenu } from '../shell/BlockMenu';
import { useStudio } from '../studio';

export function Pending({ eyebrow, title, folder }: { eyebrow: string; title: string; folder?: string }) {
  const { project } = useStudio();
  let actions: ReactNode = null;
  if (folder !== undefined && project !== null) {
    actions = (
      <div className="head-actions">
        <BlockMenu project={project} folder={folder} name={title} />
      </div>
    );
  }
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
        </div>
        {actions}
      </div>
      <div className="screen-body">
        <p className="note">This screen comes in a later step.</p>
      </div>
    </section>
  );
}
