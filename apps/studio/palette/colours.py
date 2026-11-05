"""
The colours of the palette, as #rrggbb, on ColorAide: made from OKLCH (its
lightness from 0 to 1, chroma, and hue in degrees: a space in which equal
steps look equal), brought into sRGB by lessening their chroma as CSS does;
mixed; and measured: their lightness (OKLab), the contrast of two (WCAG
2), their difference (OKLab, times 100: about 2 is the least seen, 15 or
more told apart at a glance), and as those with colour blindness see them
(Machado, Oliveira and Fernandes 2009, at full severity).
"""

import functools
import math
from collections.abc import Callable

from coloraide import Color


__all__ = [
    'blind_difference',
    'contrast',
    'difference',
    'lch',
    'lightness',
    'luminance',
    'mix',
    'oklch',
    'over',
    'see_through',
    'solve'
]


@functools.cache
def oklch(lightness_: float, chroma: float, hue: float) -> str:
    """Return a colour of OKLCH, brought into sRGB by its chroma."""
    colour = Color('oklch', [lightness_, chroma, hue])
    colour.fit('srgb', method='oklch-chroma')
    return colour.convert('srgb').to_string(hex=True)


@functools.cache
def lch(colour: str) -> tuple[float, float, float]:
    """Return the lightness, chroma and hue of a colour (a grey: 0)."""
    lightness_, chroma, hue = Color(colour).convert('oklch').coords()
    return lightness_, chroma, 0.0 if math.isnan(hue) else hue


def lightness(colour: str) -> float:
    """Return the lightness of a colour (OKLab)."""
    return lch(colour)[0]


@functools.cache
def mix(start: str, end: str, fraction: float) -> str:
    """Return a colour a fraction of the way to another (in OKLab)."""
    mixed = Color(start).mix(end, fraction, space='oklab')
    return mixed.convert('srgb').to_string(hex=True)


@functools.cache
def over(base: str, colour: str, opacity: float) -> str:
    """Return a colour laid over another with an opacity, as shown."""
    return Color(base).mix(colour, opacity, space='srgb').to_string(hex=True)


def see_through(colour: str, opacity: float) -> str:
    """Return a colour with an opacity, as CSS (rgba)."""
    red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return f'rgba({red}, {green}, {blue}, {opacity:g})'


@functools.cache
def luminance(colour: str) -> float:
    """Return the luminance of a colour (WCAG 2)."""
    return Color(colour).luminance()


@functools.cache
def contrast(first: str, second: str) -> float:
    """Return the contrast of two colours (WCAG 2: from 1 to 21)."""
    return Color(first).contrast(second)


@functools.cache
def difference(first: str, second: str) -> float:
    """Return the difference of two colours (OKLab, times 100)."""
    return Color(first).delta_e(second, method='ok') * 100


@functools.cache
def _seen(colour: str, blindness: str) -> Color:
    return Color(colour).filter(blindness, method='machado')


@functools.cache
def blind_difference(first: str, second: str) -> float:
    """
    Return the difference of two colours as those with red or green colour
    blindness see them (the least of the two).
    """
    return min(
        _seen(first, kind).delta_e(_seen(second, kind), method='ok') * 100
        for kind in ('protan', 'deutan'))


def solve(start: float, end: float, meets: Callable[[float], bool]) -> float:
    """
    Return the value, from one end of a range towards the other, at which
    a test is first met, which it meets more the further it goes (by
    halving); the far end if it is never met.
    """
    if not meets(end):
        return end
    for _ in range(24):
        middle = (start + end) / 2
        if meets(middle):
            end = middle
        else:
            start = middle
    return end
