"""
The rules of version 9 of the format (2026-11-12 on), which rewrite
configurations of version 8.
"""

import numbers
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any

from .rules import Change, Rule, items, rename, renamed_params


__all__ = [
    'RULES',
    'FittedNoiseOption',
    'GaussFitMethod',
    'GaussTypes',
    'LensParameterNames',
    'SlitMasks',
    'SlitPositionAngles'
]


def _likelihoods(
        config: Mapping[str, Any]
) -> Iterator[tuple[list[str | int], MutableMapping[str, Any]]]:
    """
    Yield the likelihoods of a configuration with their paths: those of
    its observations, and those of its mock data (one for all the
    observations, or one for each).
    """
    for path, observation in items(config, 'observations'):
        likelihood = observation.get('likelihood')
        if isinstance(likelihood, MutableMapping):
            yield path + ['likelihood'], likelihood
    for mock_path, mock in items(config, 'mock'):
        for path, likelihood in items(mock, 'likelihood'):
            yield mock_path + path, likelihood


class GaussTypes(Rule):
    """
    Gaussians are of the type gauss everywhere (2026-11-12), as were those
    of the PSFs, the LSFs, the priors and the noise of mock data: the
    likelihood gaussian (of observations and of mock data), the bootstrap
    scheme gaussian, the solve gaussian (of the properties of params),
    and the surface density gaussian_ring of mass disks (gauss_ring).
    """

    @staticmethod
    def version() -> int:
        return 9

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        found = list(_likelihoods(config))
        for bootstrap_path, bootstrap in items(config, 'bootstrap'):
            found += [
                (bootstrap_path + path, scheme)
                for path, scheme in items(bootstrap, 'scheme')]
        for params_path, params in items(config, 'params'):
            found += _solves(params, params_path)
        changes = [
            Change(path, "type gaussian -> gauss")
            for path, options in found if _retyped(options, 'gauss')]
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, mass in items(galaxy, 'mass'):
                surface = mass.get('surface')
                if mass.get('type') == 'disk' \
                        and isinstance(surface, MutableMapping) \
                        and _retyped(surface, 'gauss_ring', 'gaussian_ring'):
                    changes.append(Change(
                        galaxy_path + path + ['surface'],
                        "type gaussian_ring -> gauss_ring"))
        return changes


def _retyped(
        options: MutableMapping[str, Any], new: str, old: str = 'gaussian'
) -> bool:
    """
    Give options of an old type the new type, in place, and return whether
    they were of it.
    """
    if options.get('type') != old:
        return False
    options['type'] = new
    return True


def _solves(
        params: Mapping[str, Any], path: list[str | int]
) -> list[tuple[list[str | int], MutableMapping[str, Any]]]:
    """
    Return the solves of the properties of a params section with their
    paths: of the elements that it solves ({solve: ...}), of a key or of
    each in a list of them.
    """
    properties = params.get('properties')
    if not isinstance(properties, Mapping):
        return []
    found = []
    for key, value in properties.items():
        values = enumerate(value) if isinstance(value, list) \
            else [(None, value)]
        for i, item in values:
            solve = item.get('solve') if isinstance(item, Mapping) else None
            if isinstance(solve, MutableMapping):
                found.append((
                    path + ['properties', key]
                    + ([] if i is None else [i]) + ['solve'], solve))
    return found


class GaussFitMethod(Rule):
    """
    The moment maps of a Gaussian fitted to each spectrum are those of the
    method gauss_fit (2026-11-12; was gaussian_fit), as those of a
    Gauss-Hermite series are of gauss_hermite: of the observables of pixel
    and region moments.
    """

    _MOMENTS = ('pixel_moments', 'region_moments')

    @staticmethod
    def version() -> int:
        return 9

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            observable = observation.get('observable')
            if isinstance(observable, MutableMapping) \
                    and observable.get('type') in self._MOMENTS \
                    and observable.get('method') == 'gaussian_fit':
                observable['method'] = 'gauss_fit'
                changes.append(Change(
                    path + ['observable'],
                    "method gaussian_fit -> gauss_fit"))
        return changes


class FittedNoiseOption(Rule):
    """
    The noise that a Gaussian likelihood fits is its option fitted_noise
    (2026-11-12; was noise, a name it shared with the noise of mock data):
    of the likelihoods of observations and of mock data. Its parameters
    keep their names ('{name}_noise_scale_{item}' and
    '{name}_noise_jitter_{item}').
    """

    @staticmethod
    def version() -> int:
        return 9

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        # (of version 9: this rule comes after GaussTypes)
        for path, likelihood in _likelihoods(config):
            if likelihood.get('type') == 'gauss' and 'noise' in likelihood:
                rename(likelihood, 'noise', 'fitted_noise')
                changes.append(Change(path, "noise -> fitted_noise"))
        return changes


class SlitMasks(Rule):
    """
    The mask in front of the disperser of slitless spectra (e.g. the
    shutters of JWST NIRSpec) is their option slit_mask (2026-11-12; was
    aperture, a word of the regions on the sky).
    """

    @staticmethod
    def version() -> int:
        return 9

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            observable = observation.get('observable')
            if isinstance(observable, MutableMapping) \
                    and observable.get('type') == 'slitless_spectra' \
                    and 'aperture' in observable:
                rename(observable, 'aperture', 'slit_mask')
                changes.append(Change(
                    path + ['observable'], "aperture -> slit_mask"))
        return changes


class SlitPositionAngles(Rule):
    """
    Slit spectra are oriented by the position angle of their slit
    (2026-11-12), slit_pa: that of their +x axis, along the slit, as the
    PA of a slit is quoted. Their rota was the rotation of their grid, the
    position angle of +y, across the slit, which is 90 degrees more: rota
    r is slit_pa r - 90, of the observables of slit spectra and of their
    data (in place, or pieces of projects). Without rota a slit stays as
    it was: the default slit_pa of an observable is -90, as its rota was
    0, and data still take that of the x axis of their files.
    """

    @staticmethod
    def version() -> int:
        return 9

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            for key in ('observable', 'data'):
                options = observation.get(key)
                if isinstance(options, MutableMapping) \
                        and options.get('type') == 'slit_spectra':
                    changes += _slit_pa(options, path + [key])
        # (the data pieces of a project, of their type)
        for path, data in items(config, 'data'):
            if data.get('type') == 'slit_spectra':
                changes += _slit_pa(data, path)
        return changes


def _slit_pa(
        options: MutableMapping[str, Any], path: list[str | int]
) -> list[Change]:
    """
    Rewrite the rota of the options of slit spectra (of an observable or
    of data) as their slit_pa, in place, and return the change.
    """
    if 'rota' not in options:
        return []
    rota = options['rota']
    # (null: that of the files of data)
    if rota is None:
        rename(options, 'rota', 'slit_pa')
        return [Change(path, "rota -> slit_pa")]
    if isinstance(rota, bool) or not isinstance(rota, numbers.Real):
        return [Change(path, (
            f"by hand: rota {rota!r} (the position angle of +y, across the "
            f"slit) -> slit_pa, 90 degrees less (that of +x, along the "
            f"slit)"))]
    rename(options, 'rota', 'slit_pa')
    options['slit_pa'] = rota - 90
    return [Change(path, (
        f"rota {rota} -> slit_pa {options['slit_pa']} (the position angle "
        f"of the slit, 90 degrees less)"))]


# The types of the masses of lenses (of version 8) placed by a centre, and
# those oriented by a position angle
_CENTRED_MASSES = ('sie', 'epl', 'nfw', 'multipole')
_ORIENTED_MASSES = _CENTRED_MASSES + ('shear',)


class LensParameterNames(Rule):
    """
    The centre and the position angle of the masses of a lens are named
    as those of disks (2026-11-12): xpos, ypos and posa (were x, y and pa;
    e.g. hst_lens_main_xpos, was hst_lens_main_x). In the params sections
    of the configuration, they are renamed in their properties and modes
    and in expressions.
    """

    @staticmethod
    def version() -> int:
        return 9

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        renames: dict[str, str] = {}
        for _, observation in items(config, 'observations'):
            renames |= _lens_renames(observation)
        if not renames:
            return []
        changes = []
        for params_path, params in items(config, 'params'):
            changes += renamed_params(
                params, params_path, renames, 'lens masses')
        return changes


def _lens_renames(observation: Mapping[str, Any]) -> dict[str, str]:
    """
    Return the old and the new names of the parameters of the centres and
    the position angles of the masses of the lens of an observation, if
    any: '{observation}_lens_{mass}_x' and so on, of the prefix of each
    mass (see Lens.pdescs: its name and '_' if all have names, or else
    'mass' and its index and '_', none for the first).
    """
    name = observation.get('name')
    foreground = observation.get('foreground')
    lens = foreground.get('lens') if isinstance(foreground, Mapping) \
        else None
    # (an observation of a lens of parameters has a name)
    if name is None or not isinstance(lens, Mapping):
        return {}
    masses = lens.get('masses')
    masses = masses if isinstance(masses, list) else [masses]
    names = [m.get('name') if isinstance(m, Mapping) else None
             for m in masses]
    prefixes = [f'{n}_' for n in names] if all(names) else [
        f'mass{i}_' if i else '' for i in range(len(masses))]
    renames = {}
    for prefix, mass in zip(prefixes, masses):
        kind = mass.get('type') if isinstance(mass, Mapping) else None
        base = f'{name}_lens_{prefix}'
        if kind in _CENTRED_MASSES:
            renames |= {f'{base}x': f'{base}xpos', f'{base}y': f'{base}ypos'}
        if kind in _ORIENTED_MASSES:
            renames[f'{base}pa'] = f'{base}posa'
    return renames


# The rules, in the order of the changes: FittedNoiseOption reads the
# types that GaussTypes renames
RULES: tuple[Rule, ...] = (
    GaussTypes(),
    GaussFitMethod(),
    FittedNoiseOption(),
    SlitMasks(),
    SlitPositionAngles(),
    LensParameterNames(),
)
