"""
The rules of version 2 of the format (2026-10-24 to 2026-10-25), which
rewrite configurations of version 1.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

from .rules import (
    Change, Rule, components, insert_first, items, rename)


__all__ = [
    'RULES',
    'DiskTypes',
    'FluxScales',
    'ObservationRenames',
    'ObservationTypes',
    'OwnGeometries'
]


class ObservationTypes(Rule):
    """
    Observations give their kind (2026-10-24): those that do not are on
    the sky (type: sky), as all observations were before.
    """

    @staticmethod
    def version() -> int:
        return 2

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            if 'type' not in item:
                insert_first(item, 'type', 'sky')
                changes.append(Change(path, "type: sky added"))
        return changes


class FluxScales(Rule):
    """
    Flux scales replaced the calibration of observations and the
    amplitudes of spectra of their likelihoods (2026-10-25): calibration
    is flux_scale 'one', whose parameter is '{name}_flux_scale' (was
    '{name}_calibration'), and amplitudes are a solved flux scale
    'per_spectrum' ('per_row' of slitless spectra).
    """

    @staticmethod
    def version() -> int:
        return 2

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            if 'calibration' in item:
                changes += self._calibration(config, path, item)
            likelihood = item.get('likelihood')
            if isinstance(likelihood, MutableMapping) and (
                    {'amplitudes', 'amplitude_sigma'} & set(likelihood)):
                changes += self._amplitudes(config, path, item, likelihood)
        return changes

    @staticmethod
    def _calibration(
            config: MutableMapping[str, Any],
            path: list[str | int],
            item: MutableMapping[str, Any]
    ) -> list[Change]:
        """Rewrite the calibration of an observation as its flux scale."""
        if not item['calibration']:
            del item['calibration']
            return [Change(path, "calibration: false removed")]
        rename(item, 'calibration', 'flux_scale')
        item['flux_scale'] = 'one'
        changes = [Change(path, "calibration -> flux_scale: one")]
        if (name := item.get('name')) is None:
            return changes + [Change(
                path, "by hand: name the observation (its flux scale is "
                      "the parameter '{name}_flux_scale')")]
        old, new = f'{name}_calibration', f'{name}_flux_scale'
        for params_path, params in items(config, 'params'):
            properties = params.get('properties')
            if isinstance(properties, MutableMapping) and old in properties:
                rename(properties, old, new)
                changes.append(Change(
                    params_path + ['properties'], f"{old} -> {new}"))
        if len(changes) == 1:
            changes.append(Change(
                path, f"by hand: rename the parameter {old} to {new}"))
        return changes

    @staticmethod
    def _amplitudes(
            config: MutableMapping[str, Any],
            path: list[str | int],
            item: MutableMapping[str, Any],
            likelihood: MutableMapping[str, Any]
    ) -> list[Change]:
        """
        Rewrite the amplitudes of the likelihood of an observation as its
        solved flux scale.
        """
        amplitudes = likelihood.pop('amplitudes', None)
        sigma = likelihood.pop('amplitude_sigma', None)
        if amplitudes is None:
            return [Change(
                path + ['likelihood'], "amplitude_sigma removed (no "
                "amplitudes)")]
        observable = item.get('observable')
        slitless = isinstance(observable, Mapping) and \
            observable.get('type') == 'slitless_spectra'
        item['flux_scale'] = 'per_row' if slitless else 'per_spectrum'
        changes = [Change(
            path, f"likelihood amplitudes -> flux_scale: "
                  f"{item['flux_scale']}")]
        name = item.get('name')
        param = f'{name}_flux_scale'
        params = list(items(config, 'params'))
        if sigma is None or name is None or len(params) != 1:
            changes.append(Change(
                path, f"by hand: add to the parameters {param}: {{solve: "
                      f"{{type: gaussian, mean: 1, sigma: {sigma}, "
                      f"min: 0}}}}"))
            return changes
        params_path, params = params[0]
        params.setdefault('properties', {})[param] = dict(solve=dict(
            type='gaussian', mean=1, sigma=sigma, min=0))
        changes.append(Change(
            params_path + ['properties'], f"{param} added (solved)"))
        return changes


class DiskTypes(Rule):
    """
    One disk replaced the smooth disk and the disk of clouds, and the cone
    the smooth cone (2026-10-25): the component types 'smdisk' and
    'smcone' are 'disk' and 'cone', and 'mcdisk' is 'disk' with its cflux
    and seed in its option clouds.
    """

    _NEW = dict(smdisk='disk', mcdisk='disk', smcone='cone')

    @staticmethod
    def version() -> int:
        return 2

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, component in components(galaxy):
                if (old := component.get('type')) not in self._NEW:
                    continue
                component['type'] = self._NEW[old]
                message = f"type {old} -> {self._NEW[old]}"
                if old == 'mcdisk' and 'cflux' in component:
                    clouds = dict(cflux=component['cflux'])
                    if 'seed' in component:
                        clouds['seed'] = component.pop('seed')
                    rename(component, 'cflux', 'clouds')
                    component['clouds'] = clouds
                    message += ", cflux and seed -> clouds"
                changes.append(Change(galaxy_path + path, message))
        return changes


class OwnGeometries(Rule):
    """
    The components of a galaxy share its geometry by default (2026-10-25),
    whose parameters are named plainly (e.g. posa); they had their own
    (e.g. disk_posa). The components that would change get the option
    geometry 'own', which keeps the model and the names of its
    parameters: those of galaxies of several components, those with
    names (which prefix their parameters) and the warped ones (loose or
    tilted, their own warps).
    """

    @staticmethod
    def version() -> int:
        return 2

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            found = list(components(galaxy))
            for path, component in found:
                if 'geometry' in component:
                    continue
                if len(found) > 1 or 'name' in component \
                        or component.get('loose') or component.get('tilted'):
                    component['geometry'] = 'own'
                    changes.append(Change(
                        galaxy_path + path, "geometry: own (its own, as "
                        "before; the default is now that of its galaxy)"))
        return changes


class ObservationRenames(Rule):
    """
    Two options of observations were renamed (2026-10-25): scale is
    oversampling (an observation has a flux scale too), and weights is
    pixel_weights (its likelihood has weights too).
    """

    _NEW = dict(scale='oversampling', weights='pixel_weights')

    @staticmethod
    def version() -> int:
        return 2

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            for old, new in self._NEW.items():
                if old in item:
                    rename(item, old, new)
                    changes.append(Change(path, f"{old} -> {new}"))
        return changes


# The rules, in the order of the changes
RULES: tuple[Rule, ...] = (
    ObservationTypes(),
    FluxScales(),
    DiskTypes(),
    OwnGeometries(),
    ObservationRenames(),
)
