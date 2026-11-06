// The Studio: a bar (the tabs of the tasks, the state of the server, and
// the settings), the tree of the blocks of the open project, and the
// stage, with one screen at a time.
import { useServerEvents } from './api/events';
import { useJobs } from './api/queries';
import { Logo } from './components/icons';
import { SettingsButton } from './components/ui/settings';
import { Dataset, NewDataset } from './screens/dataset/DatasetScreen';
import { isKind } from './screens/dataset/model';
import { GalaxyScreen } from './screens/galaxy/GalaxyScreen';
import { Home } from './screens/Home';
import { ObservationScreen } from './screens/observation/ObservationScreen';
import { ObserveScreen } from './screens/observe/ObserveScreen';
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
        <SettingsButton />
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
      if (screen.folder === 'observations') return <ObservationScreen key={address(screen)} name={screen.name} />;
      return <Pending key={address(screen)} eyebrow={kindOf(screen.folder)} title={screen.name} folder={screen.folder} />;
    case 'new-dataset':
      if (!isKind(screen.data)) return <Pending eyebrow="New dataset" title="This kind of data comes later" />;
      return <NewDataset key={address(screen)} kind={screen.data} />;
    case 'observe':
      return <ObserveScreen />;
    case 'compare':
      return <Pending eyebrow="Compare with data" title="Galaxies and data" />;
    case 'fit':
      return <Pending eyebrow="Fit" title="Fits and their runs" />;
  }
}
