"""
The dust of the galaxies of a scene, in the cubes of opacity of its
observation (opacity_cubes): how the cubes hold the optical depth of each
absorber, and the ratios of the optical depths of the cubes at the
wavelengths of the light of each emitter, in the frames of the dust and
the light (each at 1 + z times D(vsys) of its galaxy and its component),
which follow the systemic velocities at each evaluation (see _DustModel).
"""

import collections
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gbkfit.grid import doppler
from gbkfit.params import ParamValues
from .components.base import OpacityCubes
from .components.extinction import ExtinctionLaw
from .components.lines import Band
from .components.roles import BandEmission

if TYPE_CHECKING:
    from .galaxy import Galaxy


__all__ = [
    'opacity_cubes'
]


# The most cubes of opacity that the native kernels read (OPACITY_NUM_MAX
# in constants.hpp)
_MAX_OPACITY_CUBES = 4


class _Frame(NamedTuple):
    """
    The frame of the light or the dust of a component, at 1 + z times
    D(vsys) (see Lines.factors): the redshift of its galaxy, and its
    systemic velocity, a parameter (by the index of its galaxy in the
    scene and its name in the parameters of the galaxy; None for none, at
    rest in its galaxy: dust of no vsys, or the band of an image, whose
    wavelength is that observed).
    """
    redshift: float
    vsys: tuple[int, str] | None

    def factor(self, velocities: dict[tuple[int, str], float]) -> float:
        """Return 1 + its redshift, of the systemic velocities."""
        velocity = 0.0 if self.vsys is None else velocities[self.vsys]
        return (1 + self.redshift) * float(doppler(velocity))


class _Light(NamedTuple):
    """
    Light of a rest wavelength (um; None if not known) emitted in a frame.
    """
    wavelength: float | None
    frame: _Frame

    def observed(
            self, velocities: dict[tuple[int, str], float]
    ) -> float | None:
        """Return its observed wavelength (um), if known."""
        return None if self.wavelength is None \
            else self.wavelength * self.frame.factor(velocities)


class _Dust(NamedTuple):
    """
    Dust of an extinction law (None: grey dust, whose optical depth is
    that at V at every wavelength) in a frame (at rest at 0 for grey
    dust).
    """
    law: ExtinctionLaw | None
    frame: _Frame

    def ratio(
            self, light: _Light, velocities: dict[tuple[int, str], float]
    ) -> float:
        """
        Return the ratio of its optical depth at the wavelength of the
        light in its frame to that at V, of the systemic velocities. Raise
        ValueError if it has a law and the wavelength is not known.
        """
        if self.law is None:
            return 1.0
        if light.wavelength is None:
            raise ValueError(
                "dust of an extinction law dims light at its rest "
                "wavelength: the emission of the components it dims must "
                "name its lines (e.g. lines: [halpha]), and an image needs "
                "its band")
        return self.law.ratio(
            light.observed(velocities) / self.frame.factor(velocities))


# Dust whose optical depth is the same at every wavelength
_GREY = _Dust(None, _Frame(0.0, None))


class _DustModel:
    """
    The dust of a scene, in the cubes of opacity of its observation (see
    opacity_cubes): their layout, fixed when it is planned (the kind of
    dust of each cube, and the cube of each absorber), and the numbers of
    each evaluation, of the systemic velocities of the components (see
    update), in the arrays that it gives the plans and updates in place:
    the factor of the optical depth at V that each absorber deposits, and
    for each emitter, the ratios of the optical depths of the cubes to
    theirs at the wavelength of each of its rows of light.
    """

    def __init__(
            self,
            kinds: tuple[_Dust, ...],
            dust: list[_Dust],
            light: list[list[_Light]],
            grey: bool,
            dtype: np.dtype
    ):
        self._kinds = kinds
        self._dust = dust
        self._light = light
        self._grey = grey
        self.scales = np.ones(len(dust))
        self.ratios = [np.ones((len(rows), len(kinds)), dtype)
                       for rows in light]
        # (at rest until the first evaluation; and here the errors of
        # light of no known wavelength)
        self._fill(collections.defaultdict(float))

    def update(self, params: Sequence[ParamValues]) -> None:
        """
        Update its numbers, in place, for the values of the parameters of
        the galaxies of the scene (by their index): of their systemic
        velocities (the mean of their values at the nodes, if node-wise,
        as the lines take them; see systemic_velocity).
        """
        self._fill({
            (g, name): float(np.mean(params[g][name]))
            for g, name in self._parameters()})

    def _fill(self, velocities: dict[tuple[int, str], float]) -> None:
        """Set its numbers, in place, of the systemic velocities."""
        if self._grey:
            first = next(rows[0] for rows in self._light if rows)
            self.scales[:] = [
                dust.ratio(first, velocities) for dust in self._dust]
            return
        for ratios, rows in zip(self.ratios, self._light):
            for i, row in enumerate(rows):
                ratios[i] = [
                    kind.ratio(row, velocities) for kind in self._kinds]

    def _parameters(self) -> set[tuple[int, str]]:
        """Return the systemic velocities that its frames take."""
        frames = [dust.frame for dust in self._dust] + [
            row.frame for rows in self._light for row in rows]
        return {frame.vsys for frame in frames if frame.vsys is not None}


def opacity_cubes(
        galaxies: Sequence['Galaxy'],
        emitters: Sequence[Sequence[int]],
        lines: tuple[str, ...] | None,
        band: Band | None,
        dtype: np.dtype
) -> list[OpacityCubes]:
    """
    Return how the plan of each of the given galaxies holds, in cubes of
    opacity, the dust of all of them, which dims the light of the given
    emitters of each (by index): of each of their lines of the given
    names (all if None), or in the given band of an image (if known). The
    components with dust are the absorbers. The dust of each is that of
    its extinction law in its frame, at 1 + z times D(vsys) of its galaxy
    and its systemic velocity (see _Frame), which dims light at the
    wavelength of that light in its frame: of a line, emitted at its rest
    wavelength in the frame of its component; of the band of an image, at
    the wavelength of the band. The layout of the cubes is fixed here (see
    _layout); their numbers follow the systemic velocities, a parameter,
    at each evaluation (see _DustModel.update, which every OpacityCubes of
    the galaxies shares as its model).

    Raises
    ------
    ValueError
        If dust of an extinction law dims light of a wavelength that is
        not known, or there are too many kinds of dust (see _layout).
    """
    absorbers = [galaxy.absorbers() for galaxy in galaxies]
    if not any(absorbers):
        return [OpacityCubes(0, {}, dict.fromkeys(indices))
                for indices in emitters]
    # The dust of each absorber, and the light of each line (or band) of
    # each emitter of each galaxy
    dust = []
    light = {}
    for g, (galaxy, galaxy_absorbers, galaxy_emitters) in enumerate(zip(
            galaxies, absorbers, emitters)):
        components = galaxy.components()
        redshift = galaxy.redshift()

        def frame(i: int, moves: bool = True) -> _Frame:
            names = galaxy.component_param_names(i)
            vsys = names.get('vsys') if moves else None
            return _Frame(redshift, None if vsys is None else (g, vsys))
        for i in galaxy_absorbers:
            law = components[i].opacity().law()
            dust.append(_GREY if law is None else _Dust(law, frame(i)))
        for i in galaxy_emitters:
            emission = components[i].emission()
            # (the band of an image: its wavelength, as observed)
            at_band = band is not None and isinstance(emission, BandEmission)
            light[g, i] = [
                _Light(wavelength, frame(i, not at_band))
                for wavelength in emission.wavelengths(lines, band, redshift)]
    kinds, cubes, grey = _layout(dust, [
        row for rows in light.values() for row in rows])
    model = _DustModel(kinds, dust, list(light.values()), grey, dtype)
    ratios = dict(zip(light, model.ratios))
    deposits = iter(enumerate(cubes))
    return [
        OpacityCubes(
            len(kinds),
            {i: (cube, model.scales[a:a + 1])
             for i, (a, cube) in zip(galaxy_absorbers, deposits)},
            {i: ratios[g, i] for i in galaxy_emitters}, model)
        for g, (galaxy_absorbers, galaxy_emitters) in enumerate(
            zip(absorbers, emitters))]


def _layout(
        dust: Sequence[_Dust], light: Sequence[_Light]
) -> tuple[tuple[_Dust, ...], tuple[int, ...], bool]:
    """
    Return how the optical depth of absorbers of the given kinds of dust
    is held for an observation of light of the given rows (see _Light):
    the kind of dust of each cube of opacity, the cube of each absorber,
    and whether the cube is one of grey dust at the one wavelength of the
    light. With one kind, one cube of the optical depth at V; with several
    kinds and light of one observed wavelength at every evaluation (see
    _same_light), one cube of the optical depth at that wavelength (of
    grey dust); with several kinds and wavelengths, one cube of the
    optical depth at V of each kind. A kind is a law in a frame: a law
    counts once for each redshift and systemic velocity (a parameter) of
    the components that have it.

    Raises
    ------
    ValueError
        If with several wavelengths there are more kinds of dust than the
        kernels take.
    """
    distinct = tuple(dict.fromkeys(dust))
    if len(distinct) == 1:
        return distinct, (0,) * len(dust), False
    if light and all(_same_light(row, light[0]) for row in light):
        return (_GREY,), (0,) * len(dust), True
    if len(distinct) > _MAX_OPACITY_CUBES:
        raise ValueError(
            f"the dust that an observation of several wavelengths sees can "
            f"have at most {_MAX_OPACITY_CUBES} kinds; it has "
            f"{len(distinct)}: an extinction law counts once for each "
            f"redshift and each systemic velocity (vsys, a parameter) of "
            f"the components that have it, which can tie their vsys")
    return (
        distinct, tuple(distinct.index(kind) for kind in dust), False)


def _same_light(a: _Light, b: _Light) -> bool:
    """
    Check whether two rows of light are observed at one wavelength at
    every evaluation: in frames of one systemic velocity (or none), at one
    wavelength (to the precision of their arithmetic: the rest wavelength
    of a band, times 1 + the redshift, is its wavelength only to
    rounding), or both not known.
    """
    if a.frame.vsys != b.frame.vsys:
        return False
    at_rest = collections.defaultdict(float)
    wa, wb = a.observed(at_rest), b.observed(at_rest)
    if wa is None or wb is None:
        return wa is wb
    return bool(np.isclose(wa, wb, rtol=1e-9, atol=0))
