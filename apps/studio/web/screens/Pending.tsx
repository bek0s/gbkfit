// A screen not built yet: its title, and that it comes in a later step;
// of a block of the project, its head (its name renamed in place, its
// menu: rename, delete).
import { useBlockHead } from '../shell/BlockMenu';
import { ScreenHead } from '../shell/ScreenHead';
import { useStudio } from '../studio';

export function Pending({ eyebrow, title, folder }: { eyebrow: string; title: string; folder?: string }) {
  const { project } = useStudio();
  const head = useBlockHead(project ?? '', folder ?? '', title);
  const block = folder !== undefined && project !== null;
  return (
    <section className="screen">
      <ScreenHead eyebrow={eyebrow} title={block ? head.title : title}>
        {block ? head.menu : undefined}
      </ScreenHead>
      <div className="screen-body">
        <p className="note">This screen comes in a later step.</p>
      </div>
    </section>
  );
}
