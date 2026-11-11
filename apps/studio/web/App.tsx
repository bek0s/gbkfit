// The Studio: the bar (the project, Go to in the middle, New and the
// activity of the server), the rail of its sections down the left, the
// list of a section (opened over the stage from the rail, or pinned
// beside it), and the stage, with one screen at a time.
import { useCallback, useEffect, useRef, useState } from 'react';
import { Logo } from './components/icons';
import { Button } from './components/ui/basics';
import { Dataset, NewDataset } from './screens/dataset/DatasetScreen';
import { isKind } from './screens/dataset/model';
import { GalaxyScreen } from './screens/galaxy/GalaxyScreen';
import { Home } from './screens/Home';
import { ObservationScreen } from './screens/observation/ObservationScreen';
import { Pending } from './screens/Pending';
import { ResponseScreen } from './screens/Response';
import { Welcome } from './screens/Welcome';
import { ActivityPill } from './shell/Activity';
import { Adding, GoTo, NewMenu } from './shell/Making';
import { address, type Screen } from './shell/navigation';
import { ProjectMenu } from './shell/ProjectMenu';
import { Rail } from './shell/Rail';
import { hasList, SectionList, useCtrlKey } from './shell/SectionList';
import { useRemembered } from './shell/remember';
import { kindOf, sectionNamed, sectionOf, type SectionKey } from './shell/sections';
import { StudioProvider, useStudio } from './studio';

export function App() {
  return (
    <StudioProvider>
      <Layout />
    </StudioProvider>
  );
}

function Layout() {
  const { project, screen } = useStudio();
  const [pinned, setPinned] = useRemembered('list-pinned', false);
  const [peeking, setPeeking] = useState<SectionKey | null>(null);
  const closing = useRef<number | undefined>(undefined);
  const current = sectionOf(screen);
  const listed = project !== null && pinned && hasList(current);
  // (the list over the screen stays while the mouse is on it or on the
  // rail, and goes a moment after it leaves both)
  const stay = () => window.clearTimeout(closing.current);
  const leave = () => {
    stay();
    closing.current = window.setTimeout(() => setPeeking(null), 300);
  };
  const peek = (key: SectionKey) => {
    stay();
    setPeeking(!pinned && project !== null && hasList(key) ? key : null);
  };
  const pin = (next: boolean) => {
    setPinned(next);
    setPeeking(null);
  };
  useCtrlKey('b', useCallback(() => pin(!pinned), [pinned])); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && setPeeking(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return (
    <div className="app" data-list={listed ? 'pinned' : 'none'}>
      <header className="bar">
        <div className="bar-start">
          <div className="logo">
            <Logo />
            <span className="logo-text">
              GBKFIT <span>Studio</span>
            </span>
          </div>
          <div className="bar-project">
            <ProjectMenu />
          </div>
        </div>
        <GoTo />
        <div className="bar-end">
          <NewMenu />
          <ActivityPill />
        </div>
      </header>
      <Rail onPeek={peek} onLeave={leave} />
      {listed && <SectionList section={current} pinned onPin={pin} onOpen={() => undefined} />}
      <main className="stage">
        {peeking !== null && (
          <div className="list-flyout" onMouseEnter={stay} onMouseLeave={leave}>
            <SectionList section={peeking} pinned={false} onPin={pin} onOpen={() => setPeeking(null)} />
          </div>
        )}
        {project === null ? <Welcome /> : <ScreenView screen={screen} />}
      </main>
      <Adding />
    </div>
  );
}

function ScreenView({ screen }: { screen: Screen }) {
  switch (screen.kind) {
    case 'home':
      return <Home />;
    case 'section':
      return <EmptySection section={screen.key} />;
    case 'block':
      if (screen.folder === 'datasets') return <Dataset key={address(screen)} name={screen.name} />;
      if (screen.folder === 'psfs') return <ResponseScreen key={address(screen)} family="psf" name={screen.name} />;
      if (screen.folder === 'lsfs') return <ResponseScreen key={address(screen)} family="lsf" name={screen.name} />;
      if (screen.folder === 'galaxies') return <GalaxyScreen key={address(screen)} name={screen.name} />;
      if (screen.folder === 'observations') return <ObservationScreen key={address(screen)} name={screen.name} />;
      return <Pending key={address(screen)} eyebrow={kindOf(screen.folder)} title={screen.name} folder={screen.folder} />;
    case 'new-dataset':
      if (!isKind(screen.data)) return <Pending eyebrow="New dataset" title="This kind of data comes later" />;
      return <NewDataset key={address(screen)} kind={screen.data} />;
    case 'compare':
      return <Pending eyebrow="Compare" title="Models, data and observations side by side" />;
    case 'fit':
      return <Pending eyebrow="Fit" title="Fits and their runs" />;
  }
}

// A section of blocks with none: what it holds, and how to make the first
function EmptySection({ section }: { section: SectionKey }) {
  const { startAdding } = useStudio();
  const { title, groups } = sectionNamed(section);
  return (
    <section className="screen empty-section">
      <div className="empty-section-body">
        <h1>No {title.toLowerCase()} yet</h1>
        <div className="empty-section-actions">
          {groups.map((group) => (
            <Button key={group.folder} kind={group === groups[0] ? 'primary' : undefined} onClick={() => startAdding(group.folder)}>
              New {group.kind.toLowerCase()}…
            </Button>
          ))}
        </div>
      </div>
    </section>
  );
}
