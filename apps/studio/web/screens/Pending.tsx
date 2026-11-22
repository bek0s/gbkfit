// A screen not built yet: its title, and that it comes in a later step;
// of a block of the project, its head (its name renamed in place, its
// menu: rename, delete).
import { useBlockHead } from '../shell/BlockMenu';
import { ScreenHead } from '../shell/ScreenHead';
import type { SectionKey } from '../shell/sections';
import { useStudio } from '../studio';

// (up: the section whose overview it goes back to, if any)
export function Pending({ eyebrow, title, folder, up }: { eyebrow: string; title: string; folder?: string; up?: SectionKey }) {
  const { project } = useStudio();
  const head = useBlockHead(project ?? '', folder ?? '', title);
  const block = folder !== undefined && project !== null;
  return (
    <section className="screen">
      <ScreenHead eyebrow={eyebrow} up={up} tags={up ? [eyebrow] : []} title={block ? head.title : title}>
        {block ? head.menu : undefined}
      </ScreenHead>
      <div className="screen-body">
        <p className="note">This screen comes in a later step.</p>
      </div>
    </section>
  );
}
