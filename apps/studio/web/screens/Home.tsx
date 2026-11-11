// The home of the project: what it is and where it stands. A project with
// nothing yet shows the four steps to a first model. One with something
// shows its map (its datasets, observations, fits and galaxies, and the
// links between them: data and models meet in the fits), what runs, the
// next steps (what its links say is missing), its runs, the blocks opened
// last, and its notes (its README.md).
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { components } from '../api/schema';
import { useJobs, useNotes, useProject, useRenders, useSaveNotes } from '../api/queries';
import { Button } from '../components/ui/basics';
import { kindOf } from '../shell/sections';
import { useStudio } from '../studio';

type Link = components['schemas']['PieceLink'];

// The columns of the map: the blocks of each folder, the data and the
// galaxies at the edges and the fits between them, where they meet
const COLUMNS = [
  { folder: 'datasets', title: 'Data' },
  { folder: 'observations', title: 'Observations' },
  { folder: 'setups', title: 'Fits' },
  { folder: 'galaxies', title: 'Galaxies' },
];

// The links the map draws (of a source folder to a target folder); the
// others (e.g. of a fit to its fitter) are not drawn
const DRAWN = new Set(['observations>datasets', 'setups>observations', 'setups>galaxies']);

export function Home() {
  const { project, startAdding } = useStudio();
  const info = useProject(project);
  const names = (folder: string) => info.data?.pieces[folder] ?? [];
  const empty = ['datasets', 'observations', 'galaxies', 'setups'].every((folder) => names(folder).length === 0);
  const counts = [
    [names('datasets').length, 'dataset', 'datasets'],
    [names('observations').length, 'observation', 'observations'],
    [names('galaxies').length, 'galaxy', 'galaxies'],
    [names('setups').length, 'fit', 'fits'],
    [info.data?.runs.length ?? 0, 'run', 'runs'],
  ] as const;
  return (
    <section className="screen home-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Project</div>
          <h1>{project}</h1>
        </div>
        <span className="home-counts">{counts.map(([n, one, many]) => `${n} ${n === 1 ? one : many}`).join(' · ')}</span>
        <div className="head-actions">
          <Button onClick={() => startAdding('datasets')}>+ Add data</Button>
          <Button onClick={() => startAdding('observations')}>+ New observation</Button>
          <Button kind="primary" onClick={() => startAdding('galaxies')}>
            + New galaxy
          </Button>
        </div>
      </div>
      {info.error && <p className="error screen-error">{info.error.message}</p>}
      {info.data && (empty ? <Start /> : <Overview />)}
    </section>
  );
}

// The four steps to a first model, of a project with nothing yet
function Start() {
  const { startAdding } = useStudio();
  const steps = [
    { title: 'Add your data', text: 'A cube or maps from FITS files: the Studio reads the header, finds the spectral axis and the beam, and keeps the originals.', act: 'Add data…', run: () => startAdding('datasets') },
    { title: 'Say how it is seen', text: 'An observation of the data: its PSF and LSF (from the header, or made here), its noise, and the line of sight.', act: 'New observation…', run: () => startAdding('observations') },
    { title: 'Build a galaxy', text: 'Its components (disks, a bulge…), their density, their motions, and the mass that moves them.', act: 'New galaxy…', run: () => startAdding('galaxies') },
    { title: 'Compare, then fit', text: 'See the galaxy through the observation beside the data, move its parameters, then fit it.', act: 'Comes soon', run: null },
  ];
  return (
    <div className="home-start">
      <h2>A new project. Four steps to a first model.</h2>
      <p>Each step makes one thing and links it to the last; take them in any order. Home turns into the map of the project once there is something to show.</p>
      <div className="home-steps">
        {steps.map((step, i) => (
          <div key={step.title} className="home-step" data-off={step.run === null || undefined}>
            <span className="home-step-n">{i + 1}</span>
            <b>{step.title}</b>
            <span>{step.text}</span>
            <Button kind={i === 0 ? 'primary' : undefined} disabled={step.run === null} onClick={() => step.run?.()}>
              {step.act}
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}

function Overview() {
  return (
    <div className="home-grid">
      <ProjectMap />
      <div className="home-side">
        <Now />
        <NextSteps />
        <Notes />
      </div>
      <div className="home-row">
        <Runs />
        <Recent />
      </div>
    </div>
  );
}

function Panel({ title, note, children, className }: { title: string; note?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`card home-panel ${className ?? ''}`}>
      <div className="home-panel-head">
        <b>{title}</b>
        {note && <span>{note}</span>}
      </div>
      <div className="home-panel-body">{children}</div>
    </section>
  );
}

// The map of the project: a column of blocks of each folder, and the
// links between them; a block shows its links as it is hovered, and opens
// as it is clicked. A dataset in no observation, a galaxy in no fit, and
// a block named but missing are marked.
function ProjectMap() {
  const { project, go } = useStudio();
  const info = useProject(project);
  const links = (info.data?.links ?? []).filter((link) => DRAWN.has(`${link.source.folder}>${link.target.folder}`));
  const map = useRef<HTMLDivElement>(null);
  const [paths, setPaths] = useState<{ key: string; d: string; ends: [string, string] }[]>([]);
  const [size, setSize] = useState<[number, number]>([0, 0]);
  const [hovered, setHovered] = useState<string | null>(null);
  const id = (folder: string, name: string) => `${folder}/${name}`;
  const usedBy = (folder: string, name: string) => links.filter((link) => link.target.folder === folder && link.target.name === name);
  const gap = (folder: string, name: string): string | null => {
    if (folder === 'datasets' && usedBy(folder, name).length === 0) return 'in no observation yet';
    if (folder === 'galaxies' && usedBy(folder, name).length === 0) return 'in no fit yet';
    return null;
  };
  // (the edges, drawn once the boxes are laid out, and again as they move)
  useLayoutEffect(() => {
    const element = map.current;
    if (!element) return;
    const draw = () => {
      const box = (key: string) => element.querySelector<HTMLElement>(`[data-node="${CSS.escape(key)}"]`);
      const where = (node: HTMLElement) => {
        let x = 0;
        let y = 0;
        let at: HTMLElement | null = node;
        while (at && at !== element) {
          x += at.offsetLeft;
          y += at.offsetTop;
          at = at.offsetParent as HTMLElement | null;
        }
        return { left: x, right: x + node.offsetWidth, middle: y + node.offsetHeight / 2 };
      };
      setSize([element.clientWidth, element.clientHeight]);
      setPaths(
        links.flatMap((link: Link) => {
          // (drawn from the left column to the right one)
          const source = box(id(link.source.folder, link.source.name));
          const target = box(id(link.target.folder, link.target.name));
          if (!source || !target) return [];
          const a = where(source);
          const b = where(target);
          const [from, to] = a.left < b.left ? [a, b] : [b, a];
          const x1 = from.right;
          const x2 = to.left;
          const dx = (x2 - x1) / 2;
          return [
            {
              key: `${id(link.source.folder, link.source.name)}>${id(link.target.folder, link.target.name)}`,
              d: `M${x1} ${from.middle} C${x1 + dx} ${from.middle} ${x2 - dx} ${to.middle} ${x2} ${to.middle}`,
              ends: [id(link.source.folder, link.source.name), id(link.target.folder, link.target.name)] as [string, string],
            },
          ];
        }),
      );
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(element);
    return () => observer.disconnect();
    // (links change with the project)
  }, [info.data]); // eslint-disable-line react-hooks/exhaustive-deps
  const linked = new Set(hovered === null ? [] : paths.filter((path) => path.ends.includes(hovered)).flatMap((path) => path.ends));
  return (
    <Panel title="How the project fits together" note="data and galaxies meet in the fits · hover a box to follow its links" className="home-map">
      <div className="map" ref={map}>
        <svg className="map-edges" viewBox={`0 0 ${size[0]} ${size[1]}`} aria-hidden="true">
          {paths.map((path) => (
            <path key={path.key} d={path.d} data-lit={(hovered !== null && path.ends.includes(hovered)) || undefined} />
          ))}
        </svg>
        {COLUMNS.map((column) => (
          <div key={column.folder} className="map-column">
            <small>{column.title}</small>
            {(info.data?.pieces[column.folder] ?? []).map((name) => {
              const key = id(column.folder, name);
              const note = gap(column.folder, name);
              return (
                <button
                  key={key}
                  type="button"
                  className="map-node"
                  data-node={key}
                  data-gap={note !== null || undefined}
                  data-lit={hovered === key || undefined}
                  data-dim={(hovered !== null && !linked.has(key) && hovered !== key) || undefined}
                  onMouseEnter={() => setHovered(key)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(key)}
                  onBlur={() => setHovered(null)}
                  onClick={() => go({ kind: 'block', folder: column.folder, name })}
                >
                  <b>{name}</b>
                  <span>{note ?? describe(column.folder, name, info.data?.links ?? [])}</span>
                </button>
              );
            })}
            {(info.data?.pieces[column.folder] ?? []).length === 0 && <p className="map-none">None yet.</p>}
          </div>
        ))}
      </div>
    </Panel>
  );
}

// A word on a block of the map: what it uses or how many use it
function describe(folder: string, name: string, links: Link[]): string {
  const uses = links.filter((link) => link.source.folder === folder && link.source.name === name);
  const used = links.filter((link) => link.target.folder === folder && link.target.name === name);
  if (folder === 'observations') {
    const data = uses.find((link) => link.target.folder === 'datasets');
    return data ? `of ${data.target.name}` : 'a model of no data';
  }
  if (folder === 'setups') {
    const galaxies = uses.filter((link) => link.target.folder === 'galaxies').length;
    return `${galaxies} ${galaxies === 1 ? 'galaxy' : 'galaxies'}`;
  }
  return `used by ${used.length}`;
}

// What runs, in brief (the Activity drawer of the bar has it all)
function Now() {
  const renders = useRenders().data ?? [];
  const jobs = (useJobs().data ?? []).filter((job) => !job.preview && (job.state === 'running' || job.state === 'queued'));
  const items = [
    ...renders.map((render) => ({ key: render.id, title: 'Render', text: `${render.what} · ${render.stage} · ${Math.round(render.elapsed)} s` })),
    ...jobs.map((job) => ({ key: job.id, title: job.task === 'prep' ? 'Prepare' : job.task, text: `${job.state === 'queued' ? 'waiting' : 'running'}` })),
  ];
  return (
    <Panel title="Now" note="the Activity drawer, in brief">
      {items.length === 0 ? <p className="note">Nothing runs.</p> : items.slice(0, 4).map((item) => (
        <div key={item.key} className="home-line">
          <b>{item.title}</b>
          <span>{item.text}</span>
        </div>
      ))}
    </Panel>
  );
}

// The next steps, from the links of the project: what nothing uses yet,
// and what is named but missing
function NextSteps() {
  const { project, go, startAdding } = useStudio();
  const info = useProject(project);
  const links = info.data?.links ?? [];
  const names = (folder: string) => info.data?.pieces[folder] ?? [];
  const used = (folder: string, name: string) => links.some((link) => link.target.folder === folder && link.target.name === name);
  const steps: { key: string; text: string; act: string; run: () => void; tone: 'err' | 'warn' | 'go' }[] = [
    ...links
      .filter((link) => link.missing)
      .map((link) => ({
        key: `missing/${link.source.folder}/${link.source.name}/${link.target.name}`,
        text: `The ${kindOf(link.source.folder).toLowerCase()} ${link.source.name} names the ${kindOf(link.target.folder).toLowerCase()} ${link.target.name}, which is missing.`,
        act: 'Open it',
        run: () => go({ kind: 'block', folder: link.source.folder, name: link.source.name }),
        tone: 'err' as const,
      })),
    ...names('datasets')
      .filter((name) => !used('datasets', name))
      .map((name) => ({
        key: `observe/${name}`,
        text: `The dataset ${name} is in no observation, so nothing can use it.`,
        act: 'Observe it',
        run: () => startAdding('observations'),
        tone: 'warn' as const,
      })),
    ...names('galaxies')
      .filter((name) => !used('galaxies', name))
      .map((name) => ({
        key: `fit/${name}`,
        text: `The galaxy ${name} is in no fit yet.`,
        act: 'Open it',
        run: () => go({ kind: 'block', folder: 'galaxies', name }),
        tone: 'go' as const,
      })),
  ];
  return (
    <Panel title="Next steps" note="from the links of the project">
      {steps.length === 0 ? <p className="note">Everything is used, and nothing is missing.</p> : steps.slice(0, 6).map((step) => (
        <div key={step.key} className="home-next" data-tone={step.tone}>
          <span className="dot" />
          <span>{step.text}</span>
          <Button kind="small" onClick={step.run}>
            {step.act}
          </Button>
        </div>
      ))}
      {steps.length > 6 && <p className="note">and {steps.length - 6} more</p>}
    </Panel>
  );
}

function Runs() {
  const { project } = useStudio();
  const runs = [...(useProject(project).data?.runs ?? [])].reverse();
  return (
    <Panel title="Runs" note="of the fits of the project">
      {runs.length === 0 ? <p className="note">No run yet.</p> : runs.slice(0, 6).map((run) => (
        <div key={run} className="home-line">
          <b className="mono">{run.split('/').slice(-2).join('/')}</b>
        </div>
      ))}
    </Panel>
  );
}

function Recent() {
  const { visited, go } = useStudio();
  const blocks = visited.filter((screen) => screen.kind === 'block').slice(0, 6);
  return (
    <Panel title="Recent" note="opened last">
      {blocks.length === 0 ? <p className="note">Nothing opened yet.</p> : blocks.map((screen) => screen.kind === 'block' && (
        <button key={`${screen.folder}/${screen.name}`} type="button" className="home-recent" onClick={() => go(screen)}>
          <small>{kindOf(screen.folder)}</small>
          <b>{screen.name}</b>
        </button>
      ))}
    </Panel>
  );
}

// The notes of the project (its README.md): what it asks, and what next
function Notes() {
  const { project } = useStudio();
  const notes = useNotes(project);
  const save = useSaveNotes(project ?? '');
  const [text, setText] = useState<string | null>(null);
  useEffect(() => setText(null), [notes.data]);
  const shown = text ?? notes.data ?? '';
  const changed = text !== null && text !== (notes.data ?? '');
  return (
    <Panel title="Notes" note="README.md of the project">
      <textarea
        className="home-notes"
        value={shown}
        placeholder="What the project asks, and what comes next."
        aria-label="The notes of the project"
        onChange={(event) => setText(event.target.value)}
      />
      {save.error && <p className="error">{save.error.message}</p>}
      {changed && (
        <div className="home-notes-actions">
          <Button onClick={() => setText(null)}>Revert</Button>
          <Button kind="primary" disabled={save.isPending} onClick={() => save.mutate(shown)}>
            Save the notes
          </Button>
        </div>
      )}
    </Panel>
  );
}
