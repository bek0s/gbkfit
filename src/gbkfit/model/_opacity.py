"""
The dust of the galaxies of a scene, in the cubes of opacity of its
observation (opacity_cubes): how the cubes hold the optical depth of each
absorber, and the ratios of the optical depths of the cubes at the
wavelengths of the light of each emitter, and of the light from behind
the scene (e.g. that of the planes behind a lens plane), in the frames of
the dust and the light (each at 1 + z times D(vsys) of its galaxy and its
component; the light from behind as observed), which follow the systemic
velocities at each evaluation (see _DustModel).
"""

import collections
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import astropy.units
import numpy as np

from gbkfit.grid import doppler
from gbkfit.params import ParamValues
from .components.base import OpacityCubes
from .components.extinction import ExtinctionLaw
from .components.roles import View

if TYPE_CHECKING:
    from .galaxy import Galaxy


__all__ = [
    'check_behind',
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
    Light of rows of rest wavelengths (um; NaN where not known) emitted
    in one frame: e.g. the lines of a component, or the light from behind
    a scene (in the frame of the observer, see _OBSERVED).
    """
    wavelengths: np.ndarray
    frame: _Frame

    @classmethod
    def of(
            cls, wavelengths: Sequence[float | None], frame: _Frame
    ) -> '_Light':
        """Return the light of rest wavelengths (None if not known)."""
        return cls(np.array(
            [np.nan if w is None else w for w in wavelengths], dtype=float),
            frame)

    def observed(
            self, velocities: dict[tuple[int, str], float]
    ) -> np.ndarray:
        """
        Return the observed wavelength of each row (um; NaN where not
        known), of the systemic velocities.
        """
        return self.wavelengths * self.frame.factor(velocities)


class _Dust(NamedTuple):
    """
    Dust of an extinction law (None: grey dust, whose optical depth is
    that at V at every wavelength) in a frame (at rest at 0 for grey
    dust).
    """
    law: ExtinctionLaw | None
    frame: _Frame

    def ratios(
            self, light: _Light, velocities: dict[tuple[int, str], float]
    ) -> np.ndarray:
        """
        Return the ratio of its optical depth at the wavelength of each
        row of the light in its frame to that at V, of the systemic
        velocities: of all of them in one call of its law. Raise
        ValueError if it has a law and a wavelength is not known.
        """
        if self.law is None:
            return np.ones(len(light.wavelengths))
        observed = light.observed(velocities)
        if np.isnan(observed).any():
            raise ValueError(
                "dust of an extinction law dims light at its rest "
                "wavelength: the emission of the components it dims must "
                "name its lines (e.g. lines: [halpha]), an image needs its "
                "band, and the light from behind a scene in a spectral cube "
                "needs a spectral axis with a rest")
        return self.law.ratio(observed / self.frame.factor(velocities))


# Dust whose optical depth is the same at every wavelength
_GREY = _Dust(None, _Frame(0.0, None))

# The frame of the light from behind a scene, of its observed wavelength
_OBSERVED = _Frame(0.0, None)


class _DustModel:
    """
    The dust of a scene, in the cubes of opacity of its observation (see
    opacity_cubes): their layout, fixed when it is planned (the kind of
    dust of each cube, and the cube of each absorber), and the numbers of
    each evaluation, of the systemic velocities of the components (see
    update), in the arrays that it gives the plans and updates in place:
    the factor of the optical depth at V that each absorber deposits, for
    each emitter, the ratios of the optical depths of the cubes to theirs
    at the wavelength of each of its rows of light, and the same for the
    rows of the light from behind the scene (behind; None without it).
    The light from behind is the last of the rows of light.
    """

    def __init__(
            self,
            kinds: tuple[_Dust, ...],
            dust: list[_Dust],
            light: list[_Light],
            grey: bool,
            dtype: np.dtype,
            behind: bool
    ):
        self._kinds = kinds
        self._dust = dust
        self._light = light
        self._grey = grey
        self.scales = np.ones(len(dust))
        self.ratios = [np.ones((len(rows.wavelengths), len(kinds)), dtype)
                       for rows in light]
        self.behind = self.ratios[-1] if behind else None
        # (at rest until the first evaluation; and here the errors of
        # light of no known wavelength)
        self._velocities = collections.defaultdict(float)
        self._fill(self._velocities)

    def update(self, params: Sequence[ParamValues]) -> None:
        """
        Update its numbers, in place, for the values of the parameters of
        the galaxies of the scene (by their index): of their systemic
        velocities (the mean of their values at the nodes, if node-wise,
        as the lines take them; see systemic_velocity). Its numbers are
        those of the velocities alone, so the same velocities leave them
        as they are (the light from behind can have many rows: e.g. a
        sub-channel of each channel of a cube).
        """
        velocities = {
            (g, name): float(np.mean(params[g][name]))
            for g, name in self._parameters()}
        if velocities == self._velocities:
            return
        self._velocities = velocities
        self._fill(velocities)

    def _fill(self, velocities: dict[tuple[int, str], float]) -> None:
        """Set its numbers, in place, of the systemic velocities."""
        if self._grey:
            first = next(
                rows for rows in self._light if len(rows.wavelengths))
            first = first._replace(wavelengths=first.wavelengths[:1])
            self.scales[:] = [
                dust.ratios(first, velocities)[0] for dust in self._dust]
            return
        # (each kind, at all the rows of light of a frame at once)
        for ratios, rows in zip(self.ratios, self._light):
            for k, kind in enumerate(self._kinds):
                ratios[:, k] = kind.ratios(rows, velocities)

    def _parameters(self) -> set[tuple[int, str]]:
        """Return the systemic velocities that its frames take."""
        frames = [dust.frame for dust in self._dust] + [
            rows.frame for rows in self._light]
        return {frame.vsys for frame in frames if frame.vsys is not None}


def opacity_cubes(
        galaxies: Sequence['Galaxy'],
        emitters: Sequence[Sequence[int]],
        view: View,
        dtype: np.dtype,
        behind: Sequence[float | None] | None = None
) -> list[OpacityCubes]:
    """
    Return how the plan of each of the given galaxies holds, in cubes of
    opacity, the dust of all of them, which dims the light of the given
    emitters of each (by index): of each of their lines of the given
    names (all if None), or in the given band of an image (if known); and
    the light from behind them, if any, of the given observed wavelengths
    (behind: um, in vacuum; None where not known), one for each row of it
    (e.g. each channel of a cube, or each sub-channel of the march of
    radiative transfer, or the band of an image; see
    TransferScenePlan.transmission). The components with dust are the
    absorbers. The dust of each is that of its extinction law in its
    frame, at 1 + z times D(vsys) of its galaxy and its systemic velocity
    (see _Frame), which dims light at the wavelength of that light in its
    frame: of a line, emitted at its rest wavelength in the frame of its
    component; of the band of an image, at the wavelength of the band; of
    the light from behind, at its observed wavelength. The layout of the
    cubes is fixed here (see _layout); their numbers follow the systemic
    velocities, a parameter, at each evaluation (see _DustModel.update,
    which every OpacityCubes of the galaxies shares as its model). The
    ratios of the cubes at the rows of the light from behind are those of
    the model (its array behind, of shape (nrows, ncubes)).

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
    # each emitter of each galaxy, then the light from behind
    dust = _scene_dust(galaxies, absorbers)
    # (the band of an image: its wavelength, as observed)
    at_band = not view.spectral and view.band is not None
    light = {}
    for g, (galaxy, galaxy_emitters) in enumerate(zip(galaxies, emitters)):
        for i in galaxy_emitters:
            emission = galaxy.components()[i].emission()
            frame = _component_frame(galaxy, g, i, not at_band)
            light[g, i] = _Light.of(emission.wavelengths(
                view.lines, view.band, galaxy.redshift()), frame)
    rows = list(light.values())
    if behind is not None:
        rows.append(_Light.of(behind, _OBSERVED))
    kinds, cubes, grey = _layout(dust, rows)
    model = _DustModel(kinds, dust, rows, grey, dtype, behind is not None)
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


def check_behind(galaxies: Sequence['Galaxy'], view: View) -> None:
    """
    Check, without the grid of an observation, that the dust of galaxies
    can dim the light from behind them, as seen in a view: of a known
    wavelength, for dust of an extinction law (an image needs its band; a
    spectral axis gives the wavelengths of its channels, if it has a
    rest, which only its grid tells), and in at most _MAX_OPACITY_CUBES
    kinds in a spectral cube, whose light from behind is of many
    wavelengths (see _layout).

    Raises
    ------
    ValueError
        If it cannot.
    """
    absorbers = [galaxy.absorbers() for galaxy in galaxies]
    if not any(absorbers):
        return
    dust = _scene_dust(galaxies, absorbers)
    if not view.spectral:
        band = None if view.band is None \
            else view.band.wavelength().to_value(astropy.units.um)
        for kind in dust:
            kind.ratios(_Light.of([band], _OBSERVED), collections.defaultdict(
                float))
        return
    _check_kinds(tuple(dict.fromkeys(dust)))


def _scene_dust(
        galaxies: Sequence['Galaxy'], absorbers: Sequence[Sequence[int]]
) -> list[_Dust]:
    """
    Return the dust of each absorber (by index) of the galaxies of a
    scene, in their order: its extinction law in its frame, or grey dust.
    """
    dust = []
    for g, (galaxy, galaxy_absorbers) in enumerate(zip(galaxies, absorbers)):
        for i in galaxy_absorbers:
            law = galaxy.components()[i].opacity().law()
            dust.append(_GREY if law is None else _Dust(
                law, _component_frame(galaxy, g, i, True)))
    return dust


def _component_frame(
        galaxy: 'Galaxy', g: int, i: int, moves: bool
) -> _Frame:
    """
    Return the frame of the component of index i of a galaxy, of index g
    in its scene (see _Frame): at its systemic velocity if it moves and
    has one, else at rest in the galaxy.
    """
    vsys = galaxy.component_param_names(i).get('vsys') if moves else None
    return _Frame(galaxy.redshift(), None if vsys is None else (g, vsys))


def _check_kinds(kinds: tuple[_Dust, ...]) -> None:
    """
    Raise ValueError if there are more kinds of dust than the kernels
    take in cubes of opacity of one kind each (see _layout).
    """
    if len(kinds) > _MAX_OPACITY_CUBES:
        raise ValueError(
            f"the dust that an observation of several wavelengths sees can "
            f"have at most {_MAX_OPACITY_CUBES} kinds; it has "
            f"{len(kinds)}: an extinction law counts once for each "
            f"redshift and each systemic velocity (vsys, a parameter) of "
            f"the components that have it, which can tie their vsys")


def _layout(
        dust: Sequence[_Dust], light: Sequence[_Light]
) -> tuple[tuple[_Dust, ...], tuple[int, ...], bool]:
    """
    Return how the optical depth of absorbers of the given kinds of dust
    is held for an observation of the given light (see _Light):
    the kind of dust of each cube of opacity, the cube of each absorber,
    and whether the cube is one of grey dust at the one wavelength of the
    light. With one kind, one cube of the optical depth at V; with several
    kinds and light of one observed wavelength at every evaluation (see
    _one_wavelength), one cube of the optical depth at that wavelength (of
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
    if _one_wavelength(light):
        return (_GREY,), (0,) * len(dust), True
    _check_kinds(distinct)
    return (
        distinct, tuple(distinct.index(kind) for kind in dust), False)


def _one_wavelength(light: Sequence[_Light]) -> bool:
    """
    Check whether all the rows of light are observed at one wavelength at
    every evaluation: in frames of one systemic velocity (or none), at one
    wavelength (to the precision of their arithmetic: the rest wavelength
    of a band, times 1 + the redshift, is its wavelength only to
    rounding), or none known; not without rows.
    """
    groups = [rows for rows in light if len(rows.wavelengths)]
    if not groups:
        return False
    at_rest = collections.defaultdict(float)
    first = groups[0].observed(at_rest)[0]
    for rows in groups:
        if rows.frame.vsys != groups[0].frame.vsys:
            return False
        observed = rows.observed(at_rest)
        same = np.isnan(observed) if np.isnan(first) \
            else np.isclose(observed, first, rtol=1e-9, atol=0)
        if not same.all():
            return False
    return True
