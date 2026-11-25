"""
The rules of version 24 of the format (2026-11-26 on), which rewrite
configurations of version 23: an observation says what its data trace of
the light of the components (its tracer), the emission of a component is
its light (of lines or a continuum), GBKFIT knows no bands, and points
move with kinematics.
"""

from collections.abc import Iterator, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedMap

from .rules import Change, PiecesView, Rule, components, flow, items, rename
from .v10 import _insert_after


__all__ = [
    'RULES',
    'BandsDeclared',
    'ComponentLight',
    'ObservationTracer',
    'PointKinematics'
]


# The kinds of observations of a spectral axis, which saw lines (and with
# continuum, the velocities of the continuum)
_SPECTRAL = (
    'pixel_moments', 'pixel_spectra', 'region_moments', 'region_spectra',
    'slit_spectra', 'slitless_spectra', 'visibilities')

# The bands that GBKFIT knew (of SDSS, 2MASS and the IRAC channels 1 and 2
# of Spitzer), by name: their effective wavelength
_KNOWN_BANDS = {
    'sdss_u': '3551 Angstrom',
    'sdss_g': '4686 Angstrom',
    'sdss_r': '6166 Angstrom',
    'sdss_i': '7480 Angstrom',
    'sdss_z': '8932 Angstrom',
    'twomass_j': '1.235 um',
    'twomass_h': '1.662 um',
    'twomass_ks': '2.159 um',
    'irac1': '3.550 um',
    'irac2': '4.493 um'}


def _galaxy_components(
        config: MutableMapping[str, Any]
) -> Iterator[tuple[list[str | int], MutableMapping[str, Any]]]:
    """
    Yield the components that are dictionaries of the galaxies of a
    configuration that are dictionaries, with their paths.
    """
    for path, galaxy in items(config, 'galaxies'):
        for cpath, component in components(galaxy):
            yield path + cpath, component


class ObservationTracer(Rule):
    """
    An observation says what its data trace of the light (2026-11-26): its
    option tracer, of the type lines (named lines, or all), continuum (in
    a band) or losvd (the velocities of the stars, weighted by their light
    in a band), in place of its options lines, continuum and band.

    An image of a band has the tracer continuum of the band; a spectral
    observation of named lines, the tracer lines of them; one of the
    continuum (continuum: true), the tracer losvd: if it saw lines too
    (it did unless lines: []), they are to see by hand, in an observation
    of their own, as one observation traces one thing.
    """

    @staticmethod
    def version() -> int:
        return 24

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            changes += _rewrite_observation(path, observation)
        return changes


def _rewrite_observation(
        path: list[str | int], observation: MutableMapping[str, Any]
) -> list[Change]:
    """
    Rewrite the options lines, continuum and band of an observation as
    its tracer, in place (see ObservationTracer), and return the changes.
    """
    if not any(k in observation for k in ('lines', 'continuum', 'band')):
        return []
    first = next(
        k for k in observation if k in ('lines', 'continuum', 'band'))
    lines = observation.get('lines')
    continuum = observation.get('continuum', False)
    band = observation.get('band')
    changes = []
    if observation.get('type') in _SPECTRAL and continuum:
        tracer = flow(dict(type='losvd'))
        if lines != []:
            seen = 'all the lines' if lines is None else f"the lines {lines}"
            changes.append(Change(path + ['tracer'], (
                f"by hand: it saw {seen} too, besides the velocities of "
                f"the stars (now the tracer losvd): see them in an "
                f"observation of their own (tracer: {{type: lines}})")))
    elif observation.get('type') in _SPECTRAL and lines is not None:
        tracer = flow(dict(type='lines', lines=lines)) if lines else None
        if not lines:
            changes.append(Change(path + ['lines'], (
                "removed (lines: [] saw no lines)")))
    elif band is not None:
        tracer = flow(dict(type='continuum', band=band))
    else:
        tracer = None
    if tracer is not None:
        _insert_after(observation, first, 'tracer', tracer)
    for key in ('lines', 'continuum', 'band'):
        observation.pop(key, None)
    if tracer is not None:
        changes.insert(0, Change(path + ['tracer'], (
            f"from its options lines, continuum and band: "
            f"{dict(tracer)}")))
    return changes


class ComponentLight(Rule):
    """
    The emission of a component is its light (2026-11-26): the option
    emission is light, and its type band is continuum (a continuum, in
    bands). A component with light cannot be dust (one density was both):
    its dust is to make a component of its own, by hand.
    """

    @staticmethod
    def version() -> int:
        return 24

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, component in _galaxy_components(config):
            if 'emission' not in component:
                continue
            rename(component, 'emission', 'light')
            light = component['light']
            band = isinstance(light, MutableMapping) \
                and light.get('type') == 'band'
            if band:
                light['type'] = 'continuum'
            changes.append(Change(path + ['light'], (
                "emission -> light" + (", type band -> continuum" if band
                                       else ""))))
            opacity = component.get('opacity')
            if isinstance(opacity, MutableMapping) \
                    and opacity.get('type') == 'dust':
                changes.append(Change(path + ['opacity'], (
                    "by hand: a component with light cannot be dust (its "
                    "density was both its light and the optical depth of "
                    "the dust): make the dust a component of its own, of "
                    "the same geometry (its parameters tied to these)")))
        return changes


class PointKinematics(Rule):
    """
    A point moves with kinematics (2026-11-26): its option kinematics
    (kinematics: {type: gauss}: its systemic velocity vsys and dispersion
    disp), which a point of emission lines had without it.
    """

    @staticmethod
    def version() -> int:
        return 24

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, component in _galaxy_components(config):
            light = component.get('light')
            if component.get('type') != 'point' \
                    or not isinstance(light, MutableMapping) \
                    or light.get('type') != 'lines' \
                    or 'kinematics' in component:
                continue
            _insert_after(
                component, 'light', 'kinematics', flow(dict(type='gauss')))
            changes.append(Change(path + ['kinematics'], (
                "added: a point of lines moves (its vsys and disp)")))
        return changes


class BandsDeclared(Rule):
    """
    GBKFIT knows no bands (2026-11-26): the bands of SDSS, 2MASS and the
    IRAC channels 1 and 2 of Spitzer, which it knew, are declared in the
    section bands of the configurations that name them, with their
    effective wavelengths. Pieces of a project that name them leave it to
    the setups, by hand.
    """

    @staticmethod
    def version() -> int:
        return 24

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        declared = config.get('bands')
        declared = declared if isinstance(declared, MutableMapping) else {}
        used = {}
        for path, component in _galaxy_components(config):
            light = component.get('light')
            bands = light.get('bands') if isinstance(light, MutableMapping) \
                else None
            for name in [bands] if isinstance(bands, str) else bands or ():
                if isinstance(name, str):
                    used.setdefault(name, path + ['light', 'bands'])
        for path, observation in items(config, 'observations'):
            tracer = observation.get('tracer')
            band = tracer.get('band') \
                if isinstance(tracer, MutableMapping) else None
            if isinstance(band, str):
                used.setdefault(band, path + ['tracer', 'band'])
        missing = {
            name: path for name, path in used.items()
            if name in _KNOWN_BANDS and name not in declared}
        if not missing:
            return []
        if isinstance(config, PiecesView):
            return [Change(path, (
                f"by hand: GBKFIT knows no bands: declare {name} in the "
                f"section bands of the setups that name the piece ({name}: "
                f"{_KNOWN_BANDS[name]})")) for name, path in missing.items()]
        if 'bands' not in config:
            _insert_after(
                config, 'lines' if 'lines' in config else 'version',
                'bands', CommentedMap())
        for name in missing:
            config['bands'][name] = _KNOWN_BANDS[name]
        return [Change(['bands'], (
            f"declared {sorted(missing)}, which GBKFIT knew (it knows no "
            f"bands)"))]


RULES: tuple[Rule, ...] = (
    ObservationTracer(),
    ComponentLight(),
    PointKinematics(),
    BandsDeclared(),
)
