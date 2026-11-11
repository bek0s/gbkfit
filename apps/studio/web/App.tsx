// The Studio: the bar (the project, New, Go to and the activity of the
// server), the rail of its sections down the left, the list of the
// section on the stage (pinned beside it, or hidden and opened over it),
// and the stage, with one screen at a time.
import { useCallback, useState } from 'react';
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
import { hasList, SectionList, useListKey, useListPinned } from './shell/SectionList';
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
  const [pinned, setPinned] = useListPinned();
  const [flyout, setFlyout] = useState(false);
  const listed = project !== null && hasList(sectionOf(screen));
  // (Ctrl+B pins the list or not)
  useListKey(useCallback(() => setPinned(!pinned), [pinned, setPinned]));
  return (
    <div className="app" data-list={listed ? (pinned ? 'pinned' : 'hidden') : 'none'}>
      <header className="bar">
        <div className="logo">
          <Logo />
          <span className="logo-text">
            GBKFIT <span>Studio</span>
          </span>
        </div>
        <div className="bar-project">
          <ProjectMenu />
        </div>
        <span className="bar-space" />
        <NewMenu />
        <GoTo />
        <ActivityPill />
      </header>
      <Rail />
      {listed && pinned && <SectionList flyout={false} onClose={() => undefined} />}
      <main className="stage">
        {listed && !pinned && (
          <button type="button" className="list-handle" title="Open the list (Ctrl+B pins it)" onClick={() => setFlyout(!flyout)}>
            ☰
          </button>
        )}
        {listed && !pinned && flyout && (
          <div className="list-flyout" onMouseLeave={() => setFlyout(false)}>
            <SectionList flyout onClose={() => setFlyout(false)} />
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
