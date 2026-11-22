// The display of the problems of a screen (those the library finds in what
// the user gave, and what the screen still needs). A screen puts them in a
// ProblemsProvider; each field that can fix some registers its path
// (useProblemHome), and each problem has one home: the deepest field whose
// path starts its path, else the field that catches a prefix of it (e.g.
// the files of a dataset catch an unreadable file). A home is marked by a
// rule at its side in the colour of its worst problem, and says its
// problems in a tooltip: fields keep their height, so nothing moves as
// problems come and go. The counter in the head of the screen counts them
// and lists them, each leading to its field: the field is shown (the folds
// around it opened, brought into view, marked, its tooltip open, focused)
// until the next click or key. A fold (see ui/fold) can say the worst of
// the problems inside it, e.g. on the head of a folded section.
import { Popover, Tooltip } from '@mantine/core';
import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { isWithin, openFolds, useFold, type Fold } from './ui/fold';

export type Severity = 'error' | 'warning';

export interface Problem {
  severity: Severity;
  path: (string | number)[];
  message: string;
}

type Path = Problem['path'];

// A field that fixes problems: its path, its name in the list of the
// counter, the prefix whose otherwise homeless problems it takes (if any),
// its element (to bring into view) and the innermost fold around it
interface Home {
  path: Path;
  label: string;
  catches?: Path;
  element: HTMLElement | null;
  fold: Fold | null;
}

interface Registry {
  problems: readonly Problem[];
  // The home of each problem (the key of its path; null: no field)
  homes: Map<Problem, string | null>;
  register: (key: string, home: Home) => () => void;
  lookup: (key: string) => Home | undefined;
  // The home shown (its key; null: none), and how one is
  shown: string | null;
  show: (key: string) => void;
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
  // (a home is shown at once, its folds opened and it marked, so that it
  // can be brought into view and focused; until the next click or key)
  const [shown, setShown] = useState<string | null>(null);
  const show = useCallback((key: string) => {
    const home = fields.current.get(key);
    if (!home) return;
    flushSync(() => {
      openFolds(home.fold);
      setShown(key);
    });
    home.element?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    if (home.element) firstControl(home.element)?.focus({ preventScroll: true });
  }, []);
  useEffect(() => {
    if (shown === null) return;
    const hide = () => setShown(null);
    document.addEventListener('pointerdown', hide, true);
    document.addEventListener('keydown', hide, true);
    return () => {
      document.removeEventListener('pointerdown', hide, true);
      document.removeEventListener('keydown', hide, true);
    };
  }, [shown]);
  const homes = useMemo(() => {
    const listed = [...fields.current.values()];
    return new Map(
      problems.map((problem) => {
        const home = homeOf(problem.path, listed);
        return [problem, home && keyOf(home)] as const;
      }),
    );
  }, [problems, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const value = useMemo(() => ({ problems, homes, register, lookup, shown, show }), [problems, homes, register, lookup, shown, show]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

function useRegistry(): Registry {
  const registry = useContext(Context);
  if (registry === null) throw new Error('problems are shown out of a ProblemsProvider');
  return registry;
}

// The first control of an element that can take the focus, if any
function firstControl(element: HTMLElement): HTMLElement | null {
  const controls = element.querySelectorAll<HTMLElement>('input:not([type="hidden"]), button, select, textarea, [tabindex]:not([tabindex="-1"])');
  return [...controls].find((control) => !control.matches(':disabled')) ?? null;
}

// Registers a field as the home of the problems of its path (see homeOf);
// gives the ref of its element, its problems, worst first, the worst
// severity of them (null: none), their messages, and whether it is shown
// (see ProblemCounter)
export function useProblemHome(path: Path, label: string, catches?: Path) {
  const { problems, homes, register, shown } = useRegistry();
  const fold = useFold();
  const key = keyOf(path);
  const catchKey = catches && keyOf(catches);
  const [element, setElement] = useState<HTMLElement | null>(null);
  useLayoutEffect(
    () => register(key, { path: JSON.parse(key) as Path, label, catches: catchKey ? (JSON.parse(catchKey) as Path) : undefined, element, fold }),
    [register, key, label, catchKey, element, fold],
  );
  const own = bySeverity(problems.filter((problem) => homes.get(problem) === key));
  return { ref: setElement, problems: own, severity: own[0]?.severity ?? null, messages: own.map((problem) => problem.message), shown: shown === key && own.length > 0 };
}

// The worst severity of the problems whose homes are in a fold (null:
// none, or no problems around), e.g. for the head of a folded section
export function useSeverityWithin(foldId: string): Severity | null {
  const registry = useContext(Context);
  if (registry === null) return null;
  const { problems, homes, lookup } = registry;
  const within = problems.filter((problem) => {
    const key = homes.get(problem);
    return key != null && isWithin(lookup(key)?.fold ?? null, foldId);
  });
  return bySeverity(within)[0]?.severity ?? null;
}

// The problems of the field around (see ProblemField), for what it holds
// to mark its parts (see useProblemsAt): e.g. those of a row, in the
// lines of what it reads
const FieldProblems = createContext<readonly Problem[]>([]);

// The problems of the field around at a path, worst first: those of the
// path itself (exact), or of it and under it (none: no path)
function useFieldProblemsAt(path: Path | null, exact: boolean): Problem[] {
  const problems = useContext(FieldProblems);
  if (path === null) return [];
  return bySeverity(problems.filter((problem) => (exact ? keyOf(problem.path) === keyOf(path) : startsWith(problem.path, path))));
}

// The worst severity of the problems of the field around at a path (null:
// none, or no path; see useFieldProblemsAt); e.g. of one part of what the
// field shows
export function useProblemsAt(path: Path | null, exact = false): Severity | null {
  return useFieldProblemsAt(path, exact)[0]?.severity ?? null;
}

// A field (or a group of them) that fixes the problems of a path: marked
// by a rule at its side while it has any, which it says on hover, or
// while it is shown; what it holds can mark the parts they are of (see
// useProblemsAt)
export function ProblemField({ path, label, catches, className, children }: { path: Path; label: string; catches?: Path; className?: string; children: ReactNode }) {
  const home = useProblemHome(path, label, catches);
  return (
    <Tooltip
      label={home.messages.join('\n')}
      disabled={home.messages.length === 0}
      opened={home.shown || undefined}
      multiline
      maw={360}
      position="top-start"
      openDelay={120}
      classNames={{ tooltip: 'problem-tip' }}
      events={{ hover: true, focus: true, touch: true }}
    >
      <div ref={home.ref} className={['problem-field', className].filter(Boolean).join(' ')} data-severity={home.severity ?? undefined} data-shown={home.shown || undefined}>
        <FieldProblems.Provider value={home.problems}>{children}</FieldProblems.Provider>
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
  const { problems, homes, lookup, show } = useRegistry();
  const [open, setOpen] = useState(false);
  const errors = problems.filter((problem) => problem.severity === 'error').length;
  const warnings = problems.length - errors;
  const lead = (problem: Problem) => {
    setOpen(false);
    const key = homes.get(problem);
    if (key) show(key);
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
            <button key={i} type="button" className="problem-list-item" data-severity={problem.severity} onClick={() => lead(problem)}>
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
