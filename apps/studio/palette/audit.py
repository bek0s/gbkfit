"""
The audit of the themes of the Studio (web/themes.json, however they were
made): what every theme, in either scheme, is held to, measured (see
colours). The floors:

- text legible on every surface it is shown on (the window, the fields,
  the panels, what is raised, a selection): apart from them in lightness,
  the quietest at least 4.5:1 by WCAG, as the accent and the states are as
  text, and the text on filled controls (WCAG's floor for text); and none
  too sharp (light text on dark blurs at the highest contrasts);
- the accent and the states, as text and as fills, told apart at a glance,
  and not alike with red or green colour blindness;
- a selection seen on a panel, and apart from a hovered row;
- the surfaces apart in steps that show, tinted little, neither black nor
  white;
- the colours of things told apart: in the band of lightness in which a
  mark reads, of chroma enough to read as a colour, neighbours apart (with
  colour blindness too), and at least 3:1 on the panel (WCAG's floor for
  marks);
- the ramps of the maps: the sequential one rising (or falling) in
  lightness at every step, over most of the range, its colour changing
  about evenly; the diverging one about a grey middle, each arm away from
  it steadily, the two alike in lightness. (The published colour maps are
  even by the measure of their authors, CAM02-UCS: by that of OKLab, the
  steps of their colour differ by up to about 2.6 times, and the arms of
  the diverging ones by up to about 0.09 in lightness; the floors and
  ceilings below hold them, and catch a ramp made badly.)

And, as information, how the accent sits with the greys: the hues of the
two, and how much more colourful the accent is.
"""

import dataclasses
import math
import re
from typing import Any

from .colours import (
    blind_difference, contrast, difference, lch, lightness, over)


__all__ = [
    'Finding',
    'audit'
]


SURFACES = ('bg', 'field', 'panel', 'panel2', 'raise')

# The accent and the states, as fills and as text; and the floors of their
# pairs, at a glance and with colour blindness: clearly apart at a glance
# (10 or more; the accent and the warning, which were confused at first,
# and the warning and the error, 12 or more), and not alike with colour
# blindness, colour being never the only sign of a state (it shows with a
# sign, ! or a check, and words: WCAG 1.4.1), as in the systems of design
# in use (e.g. Primer, Carbon). (A warm accent, e.g. amber or terracotta,
# and a warning, amber or orange as is usual, are alike to the colour blind
# however light either is: hence the low floor of the two with colour
# blindness.)
ROLES = (
    dict(accent='accent-fill', ok='ok-fill', warn='warn-fill', err='err-fill'),
    dict(accent='accent', ok='ok-fill', warn='warn-fill', err='err-fill'),
    dict(accent='accent-text', ok='ok', warn='warn', err='err'))
PAIRS = (
    ('accent', 'warn', 12, 2.5),
    ('warn', 'err', 12, 4),
    ('ok', 'err', 12, 4),
    ('ok', 'warn', 10, 3),
    ('accent', 'err', 10, 3),
    ('accent', 'ok', 8, 2))


@dataclasses.dataclass
class Finding:
    """A measure of a theme in a scheme, and its floor or ceiling."""
    theme: str
    scheme: str
    check: str
    value: float
    floor: float | None = None
    ceiling: float | None = None

    @property
    def passed(self) -> bool:
        return (self.floor is None or self.value >= self.floor - 1e-9) \
            and (self.ceiling is None or self.value <= self.ceiling + 1e-9)

    def __str__(self) -> str:
        bound = f'at least {self.floor}' if self.floor is not None \
            else f'at most {self.ceiling}'
        return f'{self.theme}, {self.scheme}: {self.check} = ' \
            f'{self.value:.3f} ({bound})'


def audit(theme: dict[str, Any]) -> list[Finding]:
    """Return the findings of a theme, in both its schemes."""
    return [*_audit(theme, 'dark'), *_audit(theme, 'light')]


def _selected(s: dict[str, str]) -> str:
    # (a selection as shown over a panel)
    red, green, blue, opacity = re.fullmatch(
        r'rgba\((\d+), (\d+), (\d+), ([\d.]+)\)', s['sel']).groups()
    colour = f'#{int(red):02x}{int(green):02x}{int(blue):02x}'
    return over(s['panel'], colour, float(opacity))


def _audit(theme: dict[str, Any], kind: str) -> list[Finding]:
    s = theme[kind]
    findings = []

    def at(check, value, floor=None, ceiling=None):
        findings.append(
            Finding(theme['id'], kind, check, value, floor, ceiling))

    selected = _selected(s)
    surfaces = [s[name] for name in SURFACES] + [selected]

    def least(measure):
        return min(measure(surface) for surface in surfaces)

    def apart(colour):
        return least(lambda x: abs(lightness(colour) - lightness(x)))

    # Text
    at('fg: lightness from the surfaces', apart(s['fg']), floor=0.61)
    at('fg2: lightness from the surfaces', apart(s['fg2']), floor=0.51)
    at('muted: lightness from the surfaces', apart(s['muted']), floor=0.43)
    for name in ('muted', 'faint', 'accent-text', 'ok', 'warn', 'err'):
        at(f'{name}: WCAG as text on the surfaces',
           least(lambda x: contrast(s[name], x)), floor=4.5)
    at('fg: WCAG on the window (not too sharp)',
       contrast(s['fg'], s['bg']), ceiling=16)
    # (the fills of the states, and the marks of the accent; not its fills,
    # which their text and shape show, bright by day in some themes)
    for name in ('accent', 'ok-fill', 'warn-fill', 'err-fill'):
        at(f'{name}: WCAG as a fill or mark on the surfaces',
           least(lambda x: contrast(s[name], x)), floor=3)

    # The text on filled controls, at rest and as hovered (the text the
    # same, the fill changed enough to show)
    at('text on the accent (a filled button)',
       contrast(s['on-accent'], s['accent-fill']), floor=4.5)
    at('text on the error (a filled button)',
       contrast(s['on-err'], s['err-fill']), floor=4.5)
    at('text on the accent, hovered',
       contrast(s['on-accent'], s['accent-hover']), floor=4.5)
    at('text on the error, hovered',
       contrast(s['on-err'], s['err-hover']), floor=4.5)
    at('accent, hovered: difference',
       difference(s['accent-hover'], s['accent-fill']), floor=3)
    at('error, hovered: difference',
       difference(s['err-hover'], s['err-fill']), floor=3)
    at('text on the warning (a badge)',
       contrast(s['on-warn'], s['warn-fill']), floor=4.5)

    # The states read as colours (not as greys, nor as white)
    at('states: least chroma', min(lch(s[name])[1] for name in (
        'ok', 'warn', 'err', 'ok-fill', 'warn-fill', 'err-fill')),
        floor=0.08)

    # The accent and the states told apart, as fills and as text
    for role in ROLES:
        for a, b, glance, blind in PAIRS:
            x, y = role[a], role[b]
            at(f'{x} / {y}: difference', difference(s[x], s[y]),
               floor=glance)
            at(f'{x} / {y}: difference, colour blind',
               blind_difference(s[x], s[y]), floor=blind)

    # A selection
    at('selection: difference from the panel',
       difference(selected, s['panel']), floor=6)
    at('selection: difference from a hovered row',
       difference(selected, s['panel2']), floor=3)

    # The surfaces
    def L(name):
        return lightness(s[name])

    at('panel above the window (lightness)',
       abs(L('panel') - L('bg')), floor=0.035)
    at('field apart from the panel (lightness)',
       abs(L('panel') - L('field')), floor=0.012)
    at('raised apart from the panel (lightness)',
       abs(L('raise') - L('panel')), floor=0.012)
    at('line on the panel (WCAG)', contrast(s['line'], s['panel']),
       floor=1.2)
    at('line2 (of fields) on the fields (WCAG)',
       contrast(s['line2'], s['field']), floor=1.5)
    at('tint of the surfaces (chroma)',
       max(lch(s[name])[1] for name in SURFACES),
       ceiling=0.035 if kind == 'dark' else 0.015)
    if kind == 'dark':
        at('window not black (lightness)', L('bg'), floor=0.15)
    else:
        at('panel not white (lightness)', L('panel'), ceiling=0.98)

    # The colours of things told apart (in their order, and round it)
    cats = [s[f'cat{i}'] for i in range(1, 7)]
    low, high = (0.48, 0.67) if kind == 'dark' else (0.43, 0.77)
    at('cats: least lightness', min(lightness(c) for c in cats), floor=low)
    at('cats: most lightness', max(lightness(c) for c in cats),
       ceiling=high)
    at('cats: least chroma', min(lch(c)[1] for c in cats), floor=0.1)
    neighbours = list(zip(cats, cats[1:] + cats[:1]))
    at('cats: neighbours, difference',
       min(difference(a, b) for a, b in neighbours), floor=15)
    at('cats: neighbours, difference, colour blind',
       min(blind_difference(a, b) for a, b in neighbours), floor=6)
    at('cats: WCAG on the panel',
       min(contrast(c, s['panel']) for c in cats), floor=3)

    # The ramps (seq0 to seqN, div0 to divN)
    def steps(prefix):
        return [s[name] for name in s if re.fullmatch(prefix + r'\d+', name)]

    ramp = steps('seq')
    seq = [lightness(c) for c in ramp]
    rises = [(b - a) * (1 if kind == 'dark' else -1)
             for a, b in zip(seq, seq[1:])]
    changes = [difference(a, b) for a, b in zip(ramp, ramp[1:])]
    at('seq: steps of lightness, least', min(rises), floor=0.02)
    at('seq: lightness spanned', abs(seq[-1] - seq[0]), floor=0.6)
    at('seq: steps of colour, most over least',
       max(changes) / min(changes), ceiling=3)
    div = steps('div')
    middle = (len(div) - 1) // 2
    at('div: chroma of the middle', lch(div[middle])[1], ceiling=0.035)
    at('div: arms alike (lightness)', max(
        abs(lightness(div[i]) - lightness(div[-1 - i]))
        for i in range(middle)), ceiling=0.1)
    away = [abs(lightness(c) - lightness(div[middle])) for c in div]
    at('div: away from the middle, steadily', min(
        [away[i] - away[i + 1] for i in range(middle)]
        + [away[i + 1] - away[i] for i in range(middle, len(div) - 1)]),
        floor=0.015)

    # How the accent sits with the greys (information)
    _, accent_chroma, accent_hue = lch(s['accent'])
    _, panel_chroma, panel_hue = lch(s['panel'])
    findings.append(Finding(
        theme['id'], kind, 'accent / greys: hues apart (degrees)',
        abs((accent_hue - panel_hue + 540) % 360 - 180)))
    findings.append(Finding(
        theme['id'], kind, 'accent / greys: chroma, times',
        accent_chroma / panel_chroma if panel_chroma > 0.002 else math.inf))
    return findings
