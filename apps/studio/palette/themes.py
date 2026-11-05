"""
The themes of the Studio, each in two schemes (dark and light), made from a
few choices each (CHOICES: the tint of its greys, the hue of its accent,
the colour map of its maps, the hue its colours of things told apart start
from); and the colour maps that can be chosen for maps (MAPS). Lightness is not chosen but solved, to targets (TARGETS), on every
surface a colour is shown on, so that every theme, in either scheme, reads
alike: the same weight of text, accent and states, as legible by night as
by day. Each scheme is the tokens of web/palette.ts (Scheme), which become
the CSS variables of the page.
"""

import itertools
from collections.abc import Callable
from typing import Any

import cmcrameri.cm  # noqa: F401 (the maps of Crameri, as cmc.*)
import cmocean  # noqa: F401 (the maps of cmocean, as cmo.*)
import matplotlib
import numpy as np
from coloraide import Color

from .colours import (
    blind_difference, contrast, difference, lch, lightness, luminance, mix,
    oklch, over, see_through, solve)


__all__ = [
    'CHOICES',
    'MAPS',
    'TARGETS',
    'build',
    'maps'
]


# The steps of the ramps of the maps (seq0 to seq16, div0 to div16), which
# the Studio draws between: enough that its ramps are those of the maps
STEPS = 17

# The colour maps that can be chosen for maps (of Matplotlib, of Crameri's
# Scientific colour maps, and of cmocean: each perceptually uniform, its
# lightness rising steadily, so that it reads alike with colour blindness
# and in grey), by their names in Matplotlib, and their titles
MAPS = {
    'grey': 'Grey', 'viridis': 'Viridis', 'cividis': 'Cividis',
    'inferno': 'Inferno', 'magma': 'Magma', 'cmo.amp': 'Amp',
    'cmo.turbid': 'Turbid', 'cmc.bilbao': 'Bilbao', 'cmc.acton': 'Acton',
    'cmo.ice': 'Ice',
    'cmc.oslo': 'Oslo',
}

# The diverging colour maps, blue below the middle and red above it as the
# light of what comes nearer and goes away is, in every theme: about a dark
# middle by night (Crameri's berlin), about a light one by day (vik)
DIVERGING = dict(dark='cmc.berlin', light='cmc.vik')

# The choices of the themes, the first the default; for each: its tint of
# the greys in each scheme (a hue and a chroma: a tint shows more on light
# greys than on dark ones); how much lighter its dark surfaces are than
# those of every theme (lift: e.g. a charcoal rather than a night); its
# accent, of a hue and chroma, its lightness solved unless given (e.g.
# white, or black), and by day its fills bright, with dark text on them
# (fill='bright', e.g. amber, which is dark by day as text or a mark), or
# deep, with white text; its states (ok, warn, err: a hue and a chroma), and its
# targets, if not those of every theme (e.g. to keep its accent apart from
# its states); the colour map of its maps, or one for each scheme (of
# MAPS: by night from dark to light, by day from light to dark, the low
# values fading into the page, by day from a near white); and the hue its
# colours of things told apart start from. (The accent opposite the tint of the greys
# stands out most, e.g. amber on blue greys; near it, it is calmer, e.g.
# cyan on sea greys.)
CHOICES: list[dict[str, Any]] = [
    # (the first look of the Studio: blue greys, amber; by day its fills
    # the amber of the night, with dark text, its marks a deeper amber and
    # its text an ochre; its warning towards red, deeper by day, its error
    # crimson; inferno, by day amp)
    dict(
        id='amber', label='Amber',
        tint=dict(dark=(255, 0.022), light=(255, 0.009)),
        accent=dict(hue=(79, 0.135), fill='bright'),
        warn=(40, 0.16), err=(350, 0.18),
        targets={
            'ok': dict(dark=5.5, light=6), 'warn': dict(dark=8.5),
            'accent-text': dict(light=8.5), 'err': dict(dark=4.6, light=11.5),
            'warn-fill': dict(light=6.5)},
        map=dict(dark='inferno', light='cmo.amp'), hue=80),
    # (warm greys: a charcoal by night, ivory by day; terracotta, its
    # warning a yellow green gold and its error crimson, away from it; by
    # night its fills standing out from its lighter greys; turbid, a map
    # of sediment, by day bilbao)
    dict(
        id='clay', label='Clay',
        tint=dict(dark=(95, 0.006), light=(95, 0.012)), lift=0.07,
        accent=dict(hue=(40, 0.135), fill='bright'),
        warn=(112, 0.14), err=(345, 0.17),
        targets={
            'accent': dict(dark=5), 'accent-text': dict(light=9.5),
            'ok': dict(dark=5.5, light=7), 'warn': dict(dark=7.5, light=5.2),
            'err': dict(dark=4.6, light=13), 'ok-fill': dict(dark=5.5),
            'warn-fill': dict(dark=5.5, light=5.5), 'err-fill': dict(dark=5.5)},
        map=dict(dark='cmo.turbid', light='cmc.bilbao'), hue=40),
    # (violet greys, lavender; magma, by day acton)
    dict(
        id='nebula', label='Nebula',
        tint=dict(dark=(290, 0.03), light=(290, 0.012)),
        accent=dict(hue=(288, 0.16)),
        map=dict(dark='magma', light='cmc.acton'), hue=290),
    # (sea greys, cyan; by day success darker, away from the cyan; ice)
    dict(
        id='glacier', label='Glacier',
        tint=dict(dark=(225, 0.026), light=(225, 0.0105)),
        accent=dict(hue=(220, 0.12)),
        targets={'ok': dict(light=7), 'err': dict(light=11.5)},
        map='cmo.ice', hue=215),
    # (plain greys, white or black: colour only for states and data; white
    # by night, the warning less light; black by day, the error less dark;
    # Oslo, blue greys)
    dict(
        id='graphite', label='Graphite',
        tint=dict(dark=(260, 0.004), light=(260, 0.002)),
        accent=dict(hue=(262, 0.006), lightness=dict(dark=0.93, light=0.22)),
        targets={
            'warn': dict(dark=8), 'err': dict(light=7.8),
            'ok': dict(light=5)},
        map='cmc.oslo', hue=250),
]

# The lightness of the surfaces, the same in every theme (but for the lift
# of its dark ones): the window darkest, the fields sunk below the panels,
# which stand clear of it, and what is raised over them lighter still; no
# black by night (light text on it blurs), no white by day (it glares)
SURFACES = dict(
    dark=dict(
        bg=0.165, field=0.19, panel=0.222, panel2=0.255, raise_=0.29,
        line=0.322, line2=0.395),
    light=dict(
        bg=0.932, field=0.958, panel=0.972, panel2=0.95, raise_=0.988,
        line=0.875, line2=0.81))

# The targets the lightness of the colours is solved to, on the surface
# least in contrast with them (of the window, the fields, the panels, what
# is raised, and a selection): text by its difference in lightness from
# them (OKLab: the same steps by night and by day), the quietest by WCAG
# (at least 4.5:1, its floor for text), as the accent and the states are as
# text; the accent as a fill or a mark by WCAG (by night on the surfaces,
# as its text; by day, the lightest that white text reads on, which stands
# out from the surfaces by more than 3:1, WCAG's floor for marks: a fill
# any lighter carries neither white nor dark text at 4.5:1; or, if its
# fills are bright, its fills the least light that dark text reads on
# well, accent-fill, which its text and shape show rather than its
# contrast with the surfaces, and its marks apart from the surfaces by
# 3:1 and a little, accent-mark); the fills of
# the states, by night the least light that dark text reads on, by day the
# lightest that white text reads on; and a selection, by its difference
# from the panel (and as hovered). The accent and states are set apart in
# lightness too, not by hue alone, so that they stay apart with colour
# blindness: by night the error the least light and the warning the most,
# by day the other way round.
TARGETS: dict[str, Any] = {
    'fg': 0.62, 'fg2': 0.52, 'muted': 0.44, 'faint': 4.6,
    'accent': dict(dark=6, light=4.6),
    'accent-fill': dict(light=7),
    'accent-mark': dict(light=3.3),
    'accent-text': dict(dark=6, light=5.6),
    'ok': dict(dark=7.2, light=6.2),
    'warn': dict(dark=9.5, light=4.6),
    'err': dict(dark=4.8, light=9.8),
    'ok-fill': dict(dark=4.8, light=4.6),
    'warn-fill': dict(dark=4.8, light=4.6),
    'err-fill': dict(dark=4.8, light=4.6),
    'sel': 7, 'sel-hover': 10,
}

# The hues of the states, the same in every theme (unless a theme has its
# own; success towards teal, the further from the error and the warning
# with colour blindness)
STATES = dict(ok=(160, 0.14), warn=(80, 0.15), err=(22, 0.18))

# The text on a fill: dark on a light one, white on a dark one, past about
# the luminance at which white and the dark text are alike in contrast on
# it, so that the text put on a fill is the more legible
DARK_ON = 0.21

Meets = Callable[[str, str], bool]


def _text_on(fill: str, dark: str) -> str:
    return dark if luminance(fill) > DARK_ON else '#ffffff'


def _hovered(fill: str) -> str:
    # (a fill as hovered: a step further from the text on it, lighter under
    # dark text, darker under white, so that its text stands out the more)
    light, chroma, hue = lch(fill)
    step = 0.05 if luminance(fill) > DARK_ON else -0.05
    return oklch(min(0.985, max(0.05, light + step)), chroma, hue)


def _fit(kind: str, hue: tuple[float, float], surfaces: list[str],
         meets: Meets) -> str:
    # (the colour of a hue at the lightness at which it first meets a test
    # on every surface: the least light by night, the least dark by day)
    h, c = hue
    test = lambda light: all(meets(oklch(light, c, h), s) for s in surfaces)  # noqa: E731
    light = solve(0.35, 0.99, test) if kind == 'dark' else solve(0.75, 0.08, test)
    return oklch(light, c, h)


def _by_wcag(target: float) -> Meets:
    return lambda colour, surface: contrast(colour, surface) >= target


def _by_lightness(target: float) -> Meets:
    return lambda colour, surface: \
        abs(lightness(colour) - lightness(surface)) >= target


def _sampled(name: str, rising: bool | None = None) -> list[str]:
    # (a colour map of Matplotlib in steps, from its low end to its high
    # one; or, of a sequential map, from dark to light, or the other way
    # round; the grey of the Studio from black to white)
    if name == 'grey':
        colours = [mix('#000000', '#ffffff', k / (STEPS - 1))
                   for k in range(STEPS)]
    else:
        rgba = matplotlib.colormaps[name](np.linspace(0.0, 1.0, STEPS))
        colours = [Color('srgb', [float(v) for v in rgb[:3]]).to_string(hex=True)
                   for rgb in rgba]
    if rising is not None and (lightness(colours[-1]) > lightness(colours[0])) != rising:
        colours.reverse()
    return colours


def maps() -> dict[str, dict[str, Any]]:
    """
    Return the colour maps that can be chosen for maps, by their names:
    their titles, and their colours, from dark to light.
    """
    return {name.split('.')[-1]: dict(title=title, colours=_sampled(name, True))
            for name, title in MAPS.items()}


def _categories(kind: str, hue: float, panel: str, lift: float) -> list[str]:
    # (the colours of things told apart: six hues 60 degrees apart, within
    # the band of lightness in which a mark reads on the panel, at two
    # lightnesses by turns, the first of the hue of the theme, the others in
    # the order that keeps neighbours, in the order and round it, the most
    # apart: the least of their chroma, their contrast with the panel, and
    # their differences from their neighbours, over the floors of the audit)
    high, low, chroma = (0.655, 0.555 + lift, 0.16) if kind == 'dark' \
        else (0.62, 0.48, 0.16)
    hues = [hue + 60 * k for k in range(6)]

    def colours(order):
        return [oklch(low if i % 2 else high, chroma, h)
                for i, h in enumerate(order)]

    def score(cats):
        return min(min(
            lch(c)[1] / 0.1, contrast(c, panel) / 3,
            difference(c, cats[(i + 1) % 6]) / 15,
            blind_difference(c, cats[(i + 1) % 6]) / 8)
            for i, c in enumerate(cats))

    candidates = (colours([hues[0], *order])
                  for order in itertools.permutations(hues[1:]))
    return max(candidates, key=score)


def _scheme(choices: dict[str, Any], kind: str) -> dict[str, str]:
    hue, chroma = choices['tint'][kind]
    lift = choices.get('lift', 0) if kind == 'dark' else 0
    surface = {name.rstrip('_'): oklch(light + lift, chroma, hue)
               for name, light in SURFACES[kind].items()}
    bg, field, panel, panel2, raise_ = (
        surface[name] for name in ('bg', 'field', 'panel', 'panel2', 'raise'))

    def target(name):
        return choices.get('targets', {}).get(name, {}).get(
            kind, TARGETS[name][kind])

    # (the accent, as a mark and as a fill, and a selection of its fill over
    # a panel, as much from it as its target, and as hovered)
    accent_hue = choices['accent']['hue']
    given = choices['accent'].get('lightness')
    dark_text = oklch(0.2, 0.03, accent_hue[0])
    plain = [bg, field, panel, panel2, raise_]
    if given:
        accent = accent_fill = oklch(given[kind], accent_hue[1], accent_hue[0])
    elif kind == 'dark':
        accent = accent_fill = _fit(
            kind, accent_hue, plain, _by_wcag(target('accent')))
    elif choices['accent'].get('fill') == 'bright':
        accent = _fit(kind, accent_hue, plain, _by_wcag(target('accent-mark')))
        accent_fill = _fit(
            'dark', accent_hue, [dark_text], _by_wcag(target('accent-fill')))
    else:
        accent = accent_fill = _fit(
            kind, accent_hue, ['#ffffff'], _by_wcag(target('accent')))

    def opacity(name):
        return solve(0.02, 0.6, lambda a: difference(
            over(panel, accent_fill, a), panel) >= TARGETS[name])

    sel, sel_hover = opacity('sel'), opacity('sel-hover')
    surfaces = [*plain, over(panel, accent_fill, sel)]

    # (the text, of the tint but less, as legible as its target on every
    # surface; the accent as text, the accent itself if it is as legible)
    def text(meets):
        return _fit(kind, (hue, chroma * 0.6), surfaces, meets)

    as_text = _by_wcag(target('accent-text'))
    accent_text = accent if given or all(as_text(accent, s) for s in surfaces) \
        else _fit(kind, accent_hue, surfaces, as_text)

    def state(name):
        return _fit(kind, choices.get(name, STATES[name]), surfaces,
                    _by_wcag(target(name)))

    def fill(name):
        return _fit(kind, choices.get(name, STATES[name]),
                    [dark_text if kind == 'dark' else '#ffffff'],
                    _by_wcag(target(f'{name}-fill')))

    ok_fill, warn_fill, err_fill = fill('ok'), fill('warn'), fill('err')

    # (the ramps: the colour map of the theme, by night from dark to light,
    # by day from light to dark; and the diverging one of the scheme)
    chosen = choices['map']
    seq = _sampled(chosen if isinstance(chosen, str) else chosen[kind],
                   kind == 'dark')
    div = _sampled(DIVERGING[kind])
    cats = _categories(kind, choices['hue'], panel, min(lift, 0.1))
    dark_bg = oklch(SURFACES['dark']['bg'], choices['tint']['dark'][1],
                    choices['tint']['dark'][0])
    return {
        **surface,
        'fg': text(_by_lightness(TARGETS['fg'])),
        'fg2': text(_by_lightness(TARGETS['fg2'])),
        'muted': text(lambda t, s: _by_lightness(TARGETS['muted'])(t, s)
                      and _by_wcag(4.5)(t, s)),
        'faint': text(_by_wcag(TARGETS['faint'])),
        'accent': accent,
        'accent-fill': accent_fill,
        'accent-text': accent_text,
        'on-accent': _text_on(accent_fill, dark_text),
        'ok': state('ok'),
        'warn': state('warn'),
        'err': state('err'),
        'ok-fill': ok_fill,
        'warn-fill': warn_fill,
        'on-warn': _text_on(warn_fill, dark_text),
        'err-fill': err_fill,
        'on-err': _text_on(err_fill, dark_text),
        'accent-hover': _hovered(accent_fill),
        'err-hover': _hovered(err_fill),
        'sel': see_through(accent_fill, round(sel, 3)),
        'sel-hover': see_through(accent_fill, round(sel_hover, 3)),
        'shadow': 'rgba(0, 0, 0, .5)' if kind == 'dark'
        else see_through(oklch(0.25, chroma, hue), 0.2),
        # (the dimming and the hints are dark in both schemes: they lie
        # over maps)
        'dim': see_through(dark_bg, 0.55),
        'hint-bg': see_through(dark_bg, 0.72),
        'hint-fg': '#ffffff',
        **{f'cat{i + 1}': colour for i, colour in enumerate(cats)},
        **{f'seq{i}': colour for i, colour in enumerate(seq)},
        **{f'div{i}': colour for i, colour in enumerate(div)},
    }


def _shades(dark: str, light: str) -> list[str]:
    # (the ten shades of a colour that Mantine needs, from light to dark:
    # that of the dark scheme the fifth, that of the light one the eighth)
    return [
        mix('#ffffff', dark, 0.1), mix('#ffffff', dark, 0.22),
        mix('#ffffff', dark, 0.45), mix('#ffffff', dark, 0.72), dark,
        mix(dark, light, 0.33), mix(dark, light, 0.67), light,
        mix(light, '#000000', 0.2), mix(light, '#000000', 0.38)]


def build(choices: dict[str, Any]) -> dict[str, Any]:
    """
    Return a theme of its choices: its name, its schemes, the greys of its
    light scheme as Mantine's ten, and the shades of its accent and error
    as Mantine's ten (see web/theme.ts).
    """
    dark, light = _scheme(choices, 'dark'), _scheme(choices, 'light')
    hue, chroma = choices['tint']['light']
    steps = SURFACES['light']
    greys = [oklch(value, chroma, hue) for value in (
        steps['panel2'], steps['bg'], 0.905, steps['line'], steps['line2'],
        0.7)]
    return dict(
        id=choices['id'],
        label=choices['label'],
        dark=dark,
        light=light,
        greys=[*greys, light['faint'], light['muted'], light['fg2'],
               light['fg']],
        shades=dict(
            accent=_shades(dark['accent-fill'], light['accent-fill']),
            err=_shades(dark['err-fill'], light['err-fill'])))
