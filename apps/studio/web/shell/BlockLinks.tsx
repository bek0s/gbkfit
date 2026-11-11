// The links of a block, in the head of its screen: the blocks it uses
// (e.g. the dataset of an observation) and those that use it (e.g. the
// fits of a galaxy), from the links of its project (see the server's
// ProjectInfo.links); each opens its block. A block it names that is
// missing (e.g. deleted) is marked.
import { useProject } from '../api/queries';
import { useStudio } from '../studio';
import { kindOf } from './sections';

// The links shown of each side before "+ n more"
const SHOWN = 3;

interface Linked {
  folder: string;
  name: string;
  missing: boolean;
}

export function BlockLinks({ folder, name }: { folder: string; name: string }) {
  const { project, go } = useStudio();
  const links = useProject(project).data?.links ?? [];
  const uses: Linked[] = links
    .filter((link) => link.source.folder === folder && link.source.name === name)
    .map((link) => ({ ...link.target, missing: link.missing }));
  const usedBy: Linked[] = links
    .filter((link) => link.target.folder === folder && link.target.name === name)
    .map((link) => ({ ...link.source, missing: false }));
  if (uses.length === 0 && usedBy.length === 0) return null;
  const chips = (items: Linked[]) => (
    <>
      {items.slice(0, SHOWN).map((item) => (
        <button
          key={`${item.folder}/${item.name}`}
          type="button"
          className="link-chip"
          data-missing={item.missing || undefined}
          title={item.missing ? `The ${kindOf(item.folder).toLowerCase()} ${item.name} is missing` : `Open the ${kindOf(item.folder).toLowerCase()} ${item.name}`}
          disabled={item.missing}
          onClick={() => go({ kind: 'block', folder: item.folder, name: item.name })}
        >
          <small>{kindOf(item.folder)}</small>
          {item.name}
        </button>
      ))}
      {items.length > SHOWN && <span className="link-more">+ {items.length - SHOWN} more</span>}
    </>
  );
  return (
    <div className="block-links">
      {uses.length > 0 && (
        <span className="link-side">
          <span className="link-label">uses</span>
          {chips(uses)}
        </span>
      )}
      {usedBy.length > 0 && (
        <span className="link-side">
          <span className="link-label">used by</span>
          {chips(usedBy)}
        </span>
      )}
    </div>
  );
}
