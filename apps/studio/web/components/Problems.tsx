// The display of the problems of a screen (those the library finds in what
// the user gave, and what the screen still needs). A screen puts them in a
// ProblemsProvider; each field that can fix some registers its path
// (useProblemHome), and each problem has one home: the deepest field whose
// path starts its path, else the field that catches a prefix of it (e.g.
// the files of a dataset catch an unreadable file). A home is marked by a
// rule at its side in the colour of its worst problem, and says its
// problems in a tooltip: fields keep their height, so nothing moves as
// problems come and go. The counter in the head of the screen counts them
// and lists them, each leading to its field.
import { Popover, Tooltip } from '@mantine/core';
import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';

export type Severity = 'error' | 'warning';

export interface Problem {
  severity: Severity;
  path: (string | number)[];
  message: string;
}

type Path = Problem['path'];

// A field that fixes problems: its path, its name in the list of the
// counter, the prefix whose otherwise homeless problems it takes (if any),
// and its element (to scroll to and focus)
interface Home {
  path: Path;
  label: string;
  catches?: Path;
  element: HTMLElement | null;
}

interface Registry {
  problems: readonly Problem[];
  // The home of each problem (the key of its path; null: no field)
  homes: Map<Problem, string | null>;
  register: (key: string, home: Home) => () => void;
  lookup: (key: string) => Home | undefined;
}

const Context = createContext<Registry | null>(null);

const keyOf = (path: Path) => JSON.stringify(path);

const startsWith = (path: Path, prefix: Path) => prefix.length <= path.length && prefix.every((part, i) => part === path[i]);

// The home of a problem among the fields: the deepest whose path starts
// its path, else the deepest catching a prefix of it (null: none)
export function homeOf(path: Path, homes: readonly Pick<Home, 'path' | 'catches'>[]): Path | null {
  let best: Path | null = null;
  for (const home of homes) if (startsWith(path, home.path) && (best === null || home.path.length > best.length)) best = home.path;
  if (best !== null) return best;
  let caught: Pick<Home, 'path' | 'catches'> | null = null;
  for (const home of homes) {
    if (home.catches && startsWith(path, home.catches) && (caught === null || home.catches.length > caught.catches!.length)) caught = home;
  }
  return caught?.path ?? null;
}

// Errors first, then warnings, each in the order given
function bySeverity(problems: readonly Problem[]): Problem[] {
  return [...problems].sort((a, b) => Number(a.severity === 'warning') - Number(b.severity === 'warning'));
}

export function ProblemsProvider({ problems, children }: { problems: readonly Problem[]; children: ReactNode }) {
  // (the fields registered, by the key of their path; as they mount and
  // go, the problems find their homes again)
  const fields = useRef(new Map<string, Home>());
  const [version, setVersion] = useState(0);
  const register = useCallback((key: string, home: Home) => {
    fields.current.set(key, home);
    setVersion((v) => v + 1);
    return () => {
      if (fields.current.get(key) === home) fields.current.delete(key);
      setVersion((v) => v + 1);
    };
  }, []);
  const lookup = useCallback((key: string) => fields.current.get(key), []);
  const homes = useMemo(() => {
    const listed = [...fields.current.values()];
    return new Map(
      problems.map((problem) => {
        const home = homeOf(problem.path, listed);
        return [problem, home && keyOf(home)] as const;
      }),
    );
  }, [problems, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const value = useMemo(() => ({ problems, homes, register, lookup }), [problems, homes, register, lookup]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

function useRegistry(): Registry {
  const registry = useContext(Context);
  if (registry === null) throw new Error('problems are shown out of a ProblemsProvider');
  return registry;
}

// Registers a field as the home of the problems of its path (see homeOf);
// gives the ref of its element, the worst severity of its problems (null:
// none) and their messages
export function useProblemHome(path: Path, label: string, catches?: Path) {
  const { problems, homes, register } = useRegistry();
  const key = keyOf(path);
  const catchKey = catches && keyOf(catches);
  const [element, setElement] = useState<HTMLElement | null>(null);
  useLayoutEffect(
    () => register(key, { path: JSON.parse(key) as Path, label, catches: catchKey ? (JSON.parse(catchKey) as Path) : undefined, element }),
    [register, key, label, catchKey, element],
  );
  const own = bySeverity(problems.filter((problem) => homes.get(problem) === key));
  return { ref: setElement, severity: own[0]?.severity ?? null, messages: own.map((problem) => problem.message) };
}

// A field (or a group of them) that fixes the problems of a path: marked
// by a rule at its side while it has any, which it says on hover
export function ProblemField({ path, label, catches, className, children }: { path: Path; label: string; catches?: Path; className?: string; children: ReactNode }) {
  const home = useProblemHome(path, label, catches);
  return (
    <Tooltip
      label={home.messages.join('\n')}
      disabled={home.messages.length === 0}
      multiline
      maw={360}
      position="top-start"
      openDelay={120}
      classNames={{ tooltip: 'problem-tip' }}
      events={{ hover: true, focus: true, touch: true }}
    >
      <div ref={home.ref} className={['problem-field', className].filter(Boolean).join(' ')} data-severity={home.severity ?? undefined}>
        {children}
      </div>
    </Tooltip>
  );
}

export function ProblemSign({ severity }: { severity: Severity }) {
  return (
    <svg className="problem-sign" data-severity={severity} viewBox="0 0 16 16" aria-label={severity}>
      {severity === 'error' ? (
        <>
          <circle cx="8" cy="8" r="6.5" />
          <path d="M8 4.6v4.2M8 11.1v.3" />
        </>
      ) : (
        <>
          <path d="M8 1.9 14.6 13.6H1.4Z" />
          <path d="M8 6.2v3.6M8 11.6v.3" />
        </>
      )}
    </svg>
  );
}

// The count of the errors and of the warnings, of a fixed width; a click
// lists them all, each leading to its field
export function ProblemCounter() {
  const { problems, homes, lookup } = useRegistry();
  const [open, setOpen] = useState(false);
  const errors = problems.filter((problem) => problem.severity === 'error').length;
  const warnings = problems.length - errors;
  const show = (problem: Problem) => {
    setOpen(false);
    const key = homes.get(problem);
    const element = key ? lookup(key)?.element : null;
    if (!element) return;
    element.scrollIntoView({ block: 'center', behavior: 'smooth' });
    const control = element.querySelector<HTMLElement>('input, button, [tabindex]');
    (control ?? element).focus({ preventScroll: true });
  };
  return (
    <Popover opened={open} onChange={setOpen} position="bottom-end" offset={6} shadow="md" width={420}>
      <Popover.Target>
        <button
          type="button"
          className="problem-counter"
          disabled={problems.length === 0}
          aria-label={`${errors} errors, ${warnings} warnings`}
          onClick={() => setOpen((was) => !was)}
        >
          <span data-severity="error" data-none={errors === 0 || undefined}>
            <ProblemSign severity="error" />
            {errors}
          </span>
          <span data-severity="warning" data-none={warnings === 0 || undefined}>
            <ProblemSign severity="warning" />
            {warnings}
          </span>
        </button>
      </Popover.Target>
      <Popover.Dropdown className="problem-list">
        {bySeverity(problems).map((problem, i) => {
          const key = homes.get(problem);
          return (
            <button key={i} type="button" className="problem-list-item" data-severity={problem.severity} onClick={() => show(problem)}>
              <ProblemSign severity={problem.severity} />
              <span className="problem-list-text">{problem.message}</span>
              <span className="problem-list-label">{(key && lookup(key)?.label) ?? ''}</span>
            </button>
          );
        })}
      </Popover.Dropdown>
    </Popover>
  );
}
