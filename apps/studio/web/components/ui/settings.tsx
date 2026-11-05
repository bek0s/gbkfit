// The settings of the user, in a dialog that a button of a gear opens, at
// the end of the bar; by section, chosen at its left:
//  - Appearance: the colour scheme (that of the system, light or dark;
//    see theme.ts), and the theme, each shown by a sample of the Studio in
//    its colours, in the scheme in use;
//  - Maps: the colour map of the maps of values (also chosen from the
//    colour bar of a map), and whether it is inverted.
// They are preferences of the user (see shell/preferences.ts), kept by the
// server in the folder of the user, which the dialog names (or why they
// could not be kept, e.g. a file edited by hand that is no longer valid
// JSON). The mark of
// the tab of the browser is in the accent of the theme.
import { ActionIcon, useComputedColorScheme, useMantineColorScheme, type MantineColorScheme } from '@mantine/core';
import { useEffect, useState, type CSSProperties, type ReactNode } from 'react';
import { colourMaps, STEPS, THEMES, type Scheme } from '../../palette';
import { usePreferencesStatus } from '../../shell/preferences';
import { setTheme, useTheme } from '../../theme';
import { COLOUR_MAPS, setColourMap, useColourMap, type ColourChoice } from '../colours';
import { CheckIcon, SettingsIcon } from '../icons';
import { NavItem } from './basics';
import { Check, Segments } from './inputs';
import { Dialog } from './overlays';

const SECTIONS = ['Appearance', 'Maps'] as const;
type Section = (typeof SECTIONS)[number];

export function SettingsButton() {
  const [open, setOpen] = useState(false);
  const theme = useTheme();
  const inUse = useComputedColorScheme('dark');
  useEffect(() => markTab(theme[inUse].accent), [theme, inUse]);
  return (
    <>
      <ActionIcon variant="subtle" color="gray" size="md" aria-label="Settings" title="Settings" className="ui-settings-button" onClick={() => setOpen(true)}>
        <SettingsIcon />
      </ActionIcon>
      {open && <SettingsDialog onClose={() => setOpen(false)} />}
    </>
  );
}

function SettingsDialog({ onClose }: { onClose: () => void }) {
  const [section, setSection] = useState<Section>('Appearance');
  const { file, problem } = usePreferencesStatus();
  return (
    <Dialog title="Settings" onClose={onClose} wide>
      <div className="settings">
        <nav className="settings-nav" aria-label="Sections">
          {SECTIONS.map((each) => (
            <NavItem key={each} label={each} active={each === section} onClick={() => setSection(each)} />
          ))}
        </nav>
        <div className="settings-page">{section === 'Appearance' ? <Appearance /> : <Maps />}</div>
      </div>
      {problem ? (
        <p className="settings-file" data-problem>
          Not saved: {problem}
        </p>
      ) : (
        file && (
          <p className="settings-file">
            Saved for you in <code>{file}</code>, whatever folder the server serves.
          </p>
        )
      )}
    </Dialog>
  );
}

// (a setting: its name, what it is, and its control)
function Setting({ name, help, children }: { name: string; help?: string; children: ReactNode }) {
  return (
    <section className="setting">
      <h3>{name}</h3>
      {help && <p>{help}</p>}
      {children}
    </section>
  );
}

function Appearance() {
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const inUse = useComputedColorScheme('dark');
  const theme = useTheme();
  return (
    <>
      <Setting name="Colour scheme" help="System follows the setting of this computer, day or night.">
        <Segments<MantineColorScheme>
          label="Colour scheme"
          value={colorScheme}
          onChange={setColorScheme}
          items={[
            { value: 'auto', label: 'System' },
            { value: 'light', label: 'Light' },
            { value: 'dark', label: 'Dark' },
          ]}
        />
      </Setting>
      <Setting name="Theme" help={`Each has a light and a dark scheme; shown here in the ${inUse} one.`}>
        <div className="theme-choices" role="radiogroup" aria-label="Theme">
          {THEMES.map((each) => (
            <button
              key={each.id}
              type="button"
              role="radio"
              aria-checked={each.id === theme.id}
              className="theme-choice"
              onClick={() => setTheme(each.id)}
            >
              <ThemeSample scheme={each[inUse]} />
              <span className="theme-choice-name">
                {each.label}
                {each.id === theme.id && <CheckIcon />}
              </span>
            </button>
          ))}
        </div>
      </Setting>
    </>
  );
}

function Maps() {
  const chosen = useColourMap();
  const theme = useTheme();
  const inUse = useComputedColorScheme('dark');
  return (
    <>
      <Setting
        name="Colour map"
        help="Of the maps of values, e.g. a flux or a dispersion; also chosen from the colour bar of a map. Maps about a middle, e.g. velocities or residuals, keep their two colours."
      >
        <div className="map-choices" role="radiogroup" aria-label="Colour map">
          {COLOUR_MAPS.map(({ name, label }) => (
            <button key={name} type="button" role="radio" aria-checked={name === chosen.map} className="map-choice" onClick={() => setColourMap({ ...chosen, map: name })}>
              <span className="map-choice-ramp" style={{ background: gradient({ ...chosen, map: name }, theme[inUse]) }} />
              <span>{label}</span>
              {name === chosen.map && <CheckIcon />}
            </button>
          ))}
        </div>
        <Check label="Inverted" checked={chosen.inverted} onChange={(inverted) => setColourMap({ ...chosen, inverted })} />
      </Setting>
    </>
  );
}

// (the colours of a colour map, as chosen, from low to high: that of the
// theme from its scheme, rather than the tokens of the page, which change
// only once the page has been drawn in it)
function gradient({ map, inverted }: ColourChoice, scheme: Scheme): string {
  const colours = map in colourMaps ? [...colourMaps[map].colours] : Array.from({ length: STEPS }, (_, i) => scheme[`seq${i}` as keyof Scheme]);
  if (inverted) colours.reverse();
  return `linear-gradient(to right, ${colours.join(', ')})`;
}

// (a theme in a scheme, as a small Studio: its bar, a panel with the rows
// of a list, one chosen, a map of a galaxy in the colours of its map, and
// a button of its accent)
function ThemeSample({ scheme }: { scheme: Scheme }) {
  const at = (fraction: number) => scheme[`seq${Math.round(fraction * (STEPS - 1))}` as keyof Scheme];
  const colours = {
    '--sample-bg': scheme.bg,
    '--sample-panel': scheme.panel,
    '--sample-line': scheme.line,
    '--sample-fg': scheme.fg2,
    '--sample-muted': scheme.muted,
    '--sample-sel': scheme.sel,
    '--sample-accent-text': scheme['accent-text'],
    '--sample-accent': scheme.accent,
    '--sample-fill': scheme['accent-fill'],
    '--sample-on-fill': scheme['on-accent'],
    '--sample-map': `radial-gradient(ellipse 46% 24% at 50% 50%, ${at(1)}, ${at(0.75)} 35%, ${at(0.45)} 70%, ${at(0)})`,
    '--sample-sky': at(0),
  } as CSSProperties;
  return (
    <span className="theme-sample" style={colours} aria-hidden>
      <i className="theme-sample-bar" />
      <i className="theme-sample-list">
        <b />
        <b data-chosen />
        <b />
        <b />
      </i>
      <i className="theme-sample-stage">
        <i className="theme-sample-map" />
        <i className="theme-sample-button" />
      </i>
    </span>
  );
}

// (the mark of the tab: the mark of the logo, see icons.tsx, in a colour)
function markTab(colour: string) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="${colour}" stroke-width="1.8"><ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-30 12 12)"/><circle cx="12" cy="12" r="2.2"/></svg>`;
  const link = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
  if (link) link.href = `data:image/svg+xml,${encodeURIComponent(svg)}`;
}
