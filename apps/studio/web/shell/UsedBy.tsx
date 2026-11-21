// The blocks of the project that name a block (e.g. the observations that
// see a dataset, or have a PSF in their instrument), a section of the dock
// of its screen; each opens from it, but fits (seen in the Fit section).
import { useReferences } from '../api/queries';
import { NavItem } from '../components/ui/basics';
import { DockSection } from '../components/ui/dock';
import { useStudio } from '../studio';
import { kindOf } from './sections';

export function UsedBy({ path }: { path: string }) {
  const { go } = useStudio();
  const references = useReferences(path).data ?? [];
  return (
    <DockSection id="used-by" title="Used by">
      {references.length === 0 ? (
        <p className="note">No observation uses it yet.</p>
      ) : (
        references.map((r) => (
          <NavItem
            key={`${r.folder}/${r.name}`}
            mono
            label={r.name}
            right={<small>{kindOf(r.folder)}</small>}
            onClick={() => r.folder !== 'setups' && go({ kind: 'block', folder: r.folder, name: r.name })}
          />
        ))
      )}
    </DockSection>
  );
}
