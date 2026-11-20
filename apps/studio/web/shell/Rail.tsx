// The rail: the sections of the Studio, down the left edge (see
// sections.ts). Home on top; the blocks a project is built of; a divider;
// the tools that use them; the settings at the foot. A section opens its
// overview (the screen of all its blocks) if it opens there first; else
// the block of it seen last, or its overview, or its first block, or
// itself, empty. Clicked, it also shows its list (onPeek), until the mouse
// leaves the rail and the list (onLeave); but on a block of a section with
// an overview, it goes up to the overview. (Hovering shows nothing: the mouse
// crosses the rail on its way to other things.)
import type { ReactNode } from 'react';
import { useProject } from '../api/queries';
import { CompareIcon, DataIcon, FitIcon, HomeIcon, LensIcon, ModelIcon, ObservationIcon } from '../components/icons';
import { SettingsButton } from '../components/ui/settings';
import { useStudio } from '../studio';
import type { Screen } from './navigation';
import { SECTIONS, sectionOf, type Section, type SectionKey } from './sections';

const ICONS: Record<SectionKey, ReactNode> = {
  home: <HomeIcon />,
  data: <DataIcon />,
  galaxies: <ModelIcon />,
  lenses: <LensIcon />,
  observations: <ObservationIcon />,
  compare: <CompareIcon />,
  fit: <FitIcon />,
};

export function Rail({ onPeek, onLeave }: { onPeek: (key: SectionKey) => void; onLeave: () => void }) {
  const { project, screen, go } = useStudio();
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
      onClick={() => {
        // (on a block of a section with an overview, its button goes up to
        // the overview, which lists them all; else it opens the section
        // and shows its list)
        if (section.overview && section.key === current && screen.kind === 'block') go({ kind: 'section', key: section.key });
        else {
          open(section.key);
          onPeek(section.key);
        }
      }}
    >
      {ICONS[section.key]}
      <span>{section.title}</span>
    </button>
  );
  return (
    <nav className="rail" aria-label="The sections of the Studio" onMouseLeave={onLeave}>
      {SECTIONS.filter((section) => !section.tool).map(button)}
      <span className="rail-divider" role="separator" />
      {SECTIONS.filter((section) => section.tool).map(button)}
      <span className="rail-foot">
        <SettingsButton />
      </span>
    </nav>
  );
}

// Open a section: home at once; a section with an overview that opens
// first at its overview; another at the block of it seen last, or its
// overview, or its first block, or itself, empty
export function useOpenSection(): (key: SectionKey) => void {
  const { project, visited, go } = useStudio();
  const info = useProject(project);
  return (key: SectionKey) => {
    if (key === 'home') {
      go({ kind: key });
      return;
    }
    const section = SECTIONS.find((each) => each.key === key)!;
    if (section.overview?.opens === 'first') {
      go({ kind: 'section', key });
      return;
    }
    const last = visited.find((seen) => sectionOf(seen) === key && seen.kind === 'block');
    if (last || section.overview) {
      go(last ?? { kind: 'section', key });
      return;
    }
    const first: Screen | undefined = section.groups
      .map((group) => ({ folder: group.folder, name: info.data?.pieces[group.folder]?.[0] }))
      .filter((found): found is { folder: string; name: string } => found.name !== undefined)
      .map(({ folder, name }) => ({ kind: 'block' as const, folder, name }))[0];
    go(first ?? { kind: 'section', key });
  };
}
