// The rail: the sections of the Studio, down the left edge (see
// sections.ts). Home on top; the blocks a project is built of; a divider;
// the tools that use them; the settings at the foot. A section of blocks
// opens the block of it seen last, or else its first, or else itself,
// empty.
import type { ReactNode } from 'react';
import { useProject } from '../api/queries';
import { CompareIcon, DataIcon, FitIcon, HomeIcon, ModelIcon, ObservationIcon } from '../components/icons';
import { SettingsButton } from '../components/ui/settings';
import { useStudio } from '../studio';
import type { Screen } from './navigation';
import { SECTIONS, sectionOf, type Section, type SectionKey } from './sections';

const ICONS: Record<SectionKey, ReactNode> = {
  home: <HomeIcon />,
  data: <DataIcon />,
  galaxies: <ModelIcon />,
  observations: <ObservationIcon />,
  compare: <CompareIcon />,
  fit: <FitIcon />,
};

export function Rail() {
  const { project, screen } = useStudio();
  const current = sectionOf(screen);
  const open = useOpenSection();
  const button = (section: Section) => (
    <button
      key={section.key}
      type="button"
      className="rail-item"
      data-tool={section.tool || undefined}
      aria-current={section.key === current ? 'page' : undefined}
      disabled={project === null}
      title={section.title}
      onClick={() => open(section.key)}
    >
      {ICONS[section.key]}
      <span>{section.title}</span>
    </button>
  );
  return (
    <nav className="rail" aria-label="The sections of the Studio">
      {SECTIONS.filter((section) => !section.tool).map(button)}
      <span className="rail-divider" role="separator" />
      {SECTIONS.filter((section) => section.tool).map(button)}
      <span className="rail-foot">
        <SettingsButton />
      </span>
    </nav>
  );
}

// Open a section: home and the tools at once; a section of blocks at the
// block of it seen last, or its first block, or itself, empty
export function useOpenSection(): (key: SectionKey) => void {
  const { project, visited, go } = useStudio();
  const info = useProject(project);
  return (key: SectionKey) => {
    if (key === 'home' || key === 'compare' || key === 'fit') {
      go({ kind: key });
      return;
    }
    const section = SECTIONS.find((each) => each.key === key)!;
    const last = visited.find((seen) => sectionOf(seen) === key && seen.kind === 'block');
    if (last) {
      go(last);
      return;
    }
    const first: Screen | undefined = section.groups
      .map((group) => ({ folder: group.folder, name: info.data?.pieces[group.folder]?.[0] }))
      .filter((found): found is { folder: string; name: string } => found.name !== undefined)
      .map(({ folder, name }) => ({ kind: 'block' as const, folder, name }))[0];
    go(first ?? { kind: 'section', key });
  };
}
