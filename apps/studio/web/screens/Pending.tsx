// A screen not built yet: its title, and that it comes in a later step;
// of a block of the project, its head (its name renamed in place, its
// menu: rename, delete).
import { BlockLinks } from '../shell/BlockLinks';
import { useBlockHead } from '../shell/BlockMenu';
import { useStudio } from '../studio';

export function Pending({ eyebrow, title, folder }: { eyebrow: string; title: string; folder?: string }) {
  const { project } = useStudio();
  const head = useBlockHead(project ?? '', folder ?? '', title);
  const block = folder !== undefined && project !== null;
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          {block ? head.title : <h1>{title}</h1>}
        </div>
        {block && <BlockLinks folder={folder} name={title} />}
        {block && <div className="head-actions">{head.menu}</div>}
      </div>
      <div className="screen-body">
        <p className="note">This screen comes in a later step.</p>
      </div>
    </section>
  );
}
