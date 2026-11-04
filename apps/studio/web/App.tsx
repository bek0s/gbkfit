// The Studio: a bar (the tabs of the tasks, the state of the server, and
// the colour scheme), the tree of the blocks of the open project, and the
// stage, with one screen at a time.
import { useServerEvents } from './api/events';
import { useJobs } from './api/queries';
import { Logo } from './components/icons';
import { SchemeMenu } from './components/ui/scheme';
import { Dataset, NewDataset } from './screens/Dataset';
import { GalaxyScreen } from './screens/GalaxyScreen';
import { BlockInLayout, DraftScreen } from './screens/Drafts';
import { Home } from './screens/Home';
import { NewImage } from './screens/Image';
import { NewMoments } from './screens/Moments';
import { Pending } from './screens/Pending';
import { ResponseScreen } from './screens/Response';
import { Welcome } from './screens/Welcome';
import { kindOf } from './shell/blocks';
import { address, type Screen } from './shell/navigation';
import { Tabs } from './shell/Tabs';
import { Tree } from './shell/Tree';
import { StudioProvider, useStudio } from './studio';

export function App() {
  return (
    <StudioProvider>
      <Layout />
    </StudioProvider>
  );
}

function Layout() {
  const { project, screen, folded } = useStudio();
  return (
    <div className={folded ? 'app folded' : 'app'}>
      <header className="bar">
        <div className="logo">
          <Logo />
          <span className="logo-text">
            GBKFIT <span>Studio</span>
          </span>
        </div>
        <Tabs />
        <ServerStatus />
        <SchemeMenu />
      </header>
      <Tree />
      <main className="stage">{project === null ? <Welcome /> : <ScreenView screen={screen} />}</main>
    </div>
  );
}

// Whether the page is connected to the server, and how many jobs run
function ServerStatus() {
  const connected = useServerEvents();
  const running = (useJobs().data ?? []).filter((job) => job.state === 'running').length;
  return (
    <div className="status" data-state={connected ? (running ? 'run' : 'ok') : 'off'}>
      <span className="dot" />
      {connected ? `connected · ${running} running` : 'not connected'}
    </div>
  );
}

function ScreenView({ screen }: { screen: Screen }) {
  switch (screen.kind) {
    case 'home':
      return <Home />;
    case 'block':
      if (screen.folder === 'datasets') return <Dataset key={address(screen)} name={screen.name} />;
      if (screen.folder === 'psfs') return <ResponseScreen key={address(screen)} family="psf" name={screen.name} />;
      if (screen.folder === 'lsfs') return <ResponseScreen key={address(screen)} family="lsf" name={screen.name} />;
      if (screen.folder === 'galaxies') return <GalaxyScreen key={address(screen)} name={screen.name} />;
      if (screen.folder === 'observations') {
        return <BlockInLayout key={address(screen)} folder={screen.folder} name={screen.name} />;
      }
      return <Pending key={address(screen)} eyebrow={kindOf(screen.folder)} title={screen.name} folder={screen.folder} />;
    case 'new-dataset':
      if (screen.data === 'moments') return <NewMoments key={address(screen)} />;
      if (screen.data === 'image') return <NewImage key={address(screen)} />;
      return <NewDataset key={address(screen)} data={screen.data} />;
    case 'draft':
      return <DraftScreen key={address(screen)} id={screen.id} name={screen.name} />;
    case 'observe':
      return <Pending eyebrow="Observe · no data" title="Galaxies through observations" />;
    case 'compare':
      return <Pending eyebrow="Compare with data" title="Galaxies and data" />;
    case 'fit':
      return <Pending eyebrow="Fit" title="Fits and their runs" />;
  }
}
