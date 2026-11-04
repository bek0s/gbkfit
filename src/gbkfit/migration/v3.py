"""
The rules of version 3 of the format (2026-10-28 to 2026-11-01), which
rewrite configurations of version 2.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

from .rules import (
    Change, Rule, components, flow, insert_first, items, like, rename)


__all__ = [
    'RULES',
    'CoverageFiles',
    'DataTypes',
    'HermiteItems',
    'ImageTraitFiles',
    'NoiseScaleMin',
    'QuantityRadii',
    'RangeNames',
    'RingsFromZero',
    'SpectralAxes',
    'SpectralTables',
    'TypedKinematics',
    'VisibilityFields',
    'Workers'
]


class TypedKinematics(Rule):
    """
    The kinematics of every component are typed (2026-10-28): those of
    disks and cones, by traits, are of the type 'traits', as those of
    spheroids were, and the vertical equilibrium of a disk is one typed
    option of them: equilibrium: {type: <equilibrium>, mass:
    <equilibrium_density>, direction: <equilibrium_direction>} (mass was
    density until 2026-11-01).
    """

    _OPTIONS = dict(
        equilibrium_density='mass', equilibrium_direction='direction')

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, component in components(galaxy):
                kinematics = component.get('kinematics')
                if not isinstance(kinematics, MutableMapping) \
                        or 'type' in kinematics:
                    continue
                insert_first(kinematics, 'type', 'traits')
                message = "type: traits"
                if 'equilibrium' in kinematics:
                    kinematics['equilibrium'] = self._equilibrium(kinematics)
                    message += ", equilibrium options -> equilibrium: {type}"
                changes.append(Change(
                    galaxy_path + path + ['kinematics'], message))
        return changes

    def _equilibrium(
            self, kinematics: MutableMapping[str, Any]
    ) -> dict[str, Any] | None:
        """
        Remove the options of the vertical equilibrium of kinematics, and
        return it in its new form.
        """
        options = {
            new: kinematics.pop(old) for old, new in self._OPTIONS.items()
            if old in kinematics}
        mode = kinematics['equilibrium']
        return None if mode is None else dict(type=mode) | options


class SpectralAxes(Rule):
    """
    The spectral axis of an observable and its data is one option,
    spectral (2026-10-28): the options spec_* of moments, spectra in
    regions and slitless spectra; the last axis of the grid (size, step,
    rpix, rval), and rest, medium and convention, of spectral cubes and
    slit spectra, and of their data. Its convention no longer needs a
    rest, and one is required without data: an axis without one gets the
    convention it had, 'relativistic'. Options that have spectral are
    left as they are: a rule runs again on what it rewrote when the
    migration of a project stopped before its end.
    """

    # The observables of spec_* options, and those of a spectral axis that
    # is the last axis of their grid, with the number of their axes on the
    # sky
    _SPEC = ('pixel_moments', 'region_moments', 'region_spectra',
             'slitless_spectra')
    _CUBES = dict(pixel_spectra=2, slit_spectra=1)
    _OPTIONS = ('size', 'step', 'rpix', 'rval', 'convention', 'rest',
                'medium')

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            observable = item.get('observable')
            if not isinstance(observable, MutableMapping):
                continue
            kind = observable.get('type')
            data = item.get('data')
            if kind in self._SPEC and self._spec(observable):
                changes.append(Change(
                    path + ['observable'], "spec_* -> spectral"))
            elif kind in self._CUBES and self._cube(
                    observable, self._CUBES[kind], True):
                changes.append(Change(
                    path + ['observable'],
                    "the spectral axis of the grid, rest, medium and "
                    "convention -> spectral"))
            if not isinstance(data, MutableMapping):
                continue
            if kind == 'region_spectra':
                moved = self._cube(data, 0, False)
            elif kind in self._CUBES:
                moved = self._cube(data, self._CUBES[kind], False)
            else:
                moved = False
            if moved:
                changes.append(Change(
                    path + ['data'], "the world coordinates of the spectral "
                    "axis, rest and medium -> spectral"))
        return changes

    def _spec(self, observable: MutableMapping[str, Any]) -> bool:
        """
        Move the options spec_* of an observable into its option spectral;
        return whether it had any.
        """
        if 'spectral' in observable:
            return False
        spectral = {
            key: observable.pop(f'spec_{key}') for key in self._OPTIONS
            if f'spec_{key}' in observable}
        if not spectral:
            return False
        self._with_convention(spectral)
        observable['spectral'] = flow(spectral)
        return True

    def _cube(
            self, options: MutableMapping[str, Any], nspatial: int,
            model: bool
    ) -> bool:
        """
        Move the spectral axis of the grid of an observable (model) or of
        its data, the axis after its nspatial axes on the sky, and its
        rest, medium and convention, into its option spectral; return
        whether it had any.
        """
        if 'spectral' in options:
            return False
        spectral = {}
        for key in ('size', 'step', 'rpix', 'rval'):
            if key not in options:
                continue
            value = options[key]
            if isinstance(value, list) and len(value) == nspatial + 1:
                spectral[key] = value[-1]
                options[key] = like(value, value[:-1])
            elif not isinstance(value, list):
                spectral[key] = value
            if not nspatial or options[key] == []:
                del options[key]
        for key in ('convention', 'rest', 'medium'):
            if key in options:
                spectral[key] = options.pop(key)
        if model and 'size' in spectral:
            self._with_convention(spectral)
        if not spectral:
            return False
        options['spectral'] = flow(spectral)
        return True

    @staticmethod
    def _with_convention(spectral: dict[str, Any]) -> None:
        """
        Give a spectral axis without a rest or a convention the
        convention its velocities had: relativistic.
        """
        if spectral.get('rest') is None \
                and spectral.get('convention') is None:
            spectral['convention'] = 'relativistic'


class VisibilityFields(Rule):
    """
    The field of visibilities is an angle (2026-10-28): fov (arcsec) is
    the size in pixels times the step; its default step is no longer 1 /
    (2 u_max) but 1 / (5 u_max), whose pixels sample the model better.
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            observable = item.get('observable')
            if not isinstance(observable, MutableMapping) \
                    or observable.get('type') != 'visibilities' \
                    or 'size' not in observable:
                continue
            size = observable['size']
            step = observable.get('step')
            path = path + ['observable']
            if step is None:
                del observable['size']
                changes.append(Change(path, (
                    "by hand: give fov, the field (arcsec), that of size "
                    "times the default step of the pixels, which was 1 / (2 "
                    "u_max) of the longest baseline")))
                continue
            steps = [step] * 2 if not isinstance(step, list) else step
            rename(observable, 'size', 'fov')
            observable['fov'] = like(
                size, [n * s for n, s in zip(size, steps)])
            changes.append(Change(
                path, "size -> fov (arcsec): size times step"))
        return changes


class SpectralTables(Rule):
    """
    The tables of the options of PSFs and LSFs that vary along the
    spectral axis, and image PSFs and LSFs, have the options of their
    constructors (2026-10-28): the file of a table is file, not table;
    their points are points, {value, unit}, not wavelength, frequency or
    velocity, {values, unit} (velocities without a unit are in km/s);
    values with a unit are values and unit.
    """

    _AXES = ('wavelength', 'frequency', 'velocity')

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            changes += self._walk(item, path)
        return changes

    def _walk(self, value: Any, path: list[str | int]) -> list[Change]:
        """Rewrite the tables in a value and its items, in place."""
        changes = []
        if isinstance(value, MutableMapping):
            if self._table(value):
                changes.append(Change(path, (
                    "a table (or images): file, points {value, unit}, "
                    "values and unit")))
            for key, item in value.items():
                changes += self._walk(item, path + [key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                changes += self._walk(item, path + [i])
        return changes

    def _table(self, mapping: MutableMapping[str, Any]) -> bool:
        """
        Rewrite a dictionary in place if it is a table or image PSF or
        LSF, and return whether it is.
        """
        changed = False
        if 'table' in mapping and 'column' in mapping:
            rename(mapping, 'table', 'file')
            changed = True
        axes = [axis for axis in self._AXES if axis in mapping]
        if len(axes) == 1 and (
                'values' in mapping or mapping.get('type') == 'images'):
            axis = axes[0]
            points = mapping[axis]
            if isinstance(points, MutableMapping) and 'values' in points:
                rename(points, 'values', 'value')
            elif axis == 'velocity':
                mapping[axis] = flow(dict(value=points, unit='km/s'))
            rename(mapping, axis, 'points')
            changed = True
        values = mapping.get('values')
        if 'points' in mapping and isinstance(values, MutableMapping) \
                and 'values' in values:
            mapping['values'] = values['values']
            mapping['unit'] = values.get('unit')
            changed = True
        return changed


class QuantityRadii(Rule):
    """
    The radii of inline mass tables and of profile observations are a
    Quantity (2026-10-28): radii {value, unit}, not radii and
    radius_unit.
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, galaxy in items(config, 'galaxies'):
            # The mass of a galaxy is a component, or a list of them
            for mass_path, item in items(galaxy, 'mass'):
                if item.get('type') == 'table' \
                        and 'radius_unit' in item and 'radii' in item:
                    item['radii'] = flow(dict(
                        value=item['radii'], unit=item.pop('radius_unit')))
                    changes.append(Change(
                        path + mass_path,
                        "radii and radius_unit -> radii {value, unit}"))
        for path, item in items(config, 'observations'):
            if item.get('type') == 'profile' and 'radius_unit' in item \
                    and 'radii' in item:
                item['radii'] = flow(dict(
                    value=item['radii'], unit=item.pop('radius_unit')))
                changes.append(Change(
                    path, "radii and radius_unit -> radii {value, unit}"))
        return changes


class ImageTraitFiles(Rule):
    """
    The file of an image trait is a FITS file (2026-10-28): file {file,
    hdu}, not file and hdu, as those of the image PSFs and LSFs.
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, galaxy in items(config, 'galaxies'):
            changes += self._walk(galaxy, path)
        return changes

    def _walk(self, value: Any, path: list[str | int]) -> list[Change]:
        """Rewrite the image traits in a value and its items, in place."""
        changes = []
        if isinstance(value, MutableMapping):
            if value.get('type') == 'image' and 'size' in value \
                    and 'hdu' in value:
                hdu = value.pop('hdu')
                if hdu != 0:
                    value['file'] = flow(dict(file=value['file'], hdu=hdu))
                changes.append(
                    Change(path, "file and hdu -> file {file, hdu}"))
            for key, item in value.items():
                changes += self._walk(item, path + [key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                changes += self._walk(item, path + [i])
        return changes


class NoiseScaleMin(Rule):
    """
    The smallest scale of a fitted noise is scale_min (2026-10-28), as
    the ranges of other options (min and max).
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            likelihood = item.get('likelihood')
            noise = likelihood.get('noise') \
                if isinstance(likelihood, MutableMapping) else None
            if not isinstance(noise, MutableMapping):
                continue
            for key, value in noise.items():
                if isinstance(value, MutableMapping) \
                        and 'scale_minimum' in value:
                    value['scale_min'] = value.pop('scale_minimum')
                    changes.append(Change(
                        path + ['likelihood', 'noise', key],
                        "scale_minimum -> scale_min"))
        return changes


class RangeNames(Rule):
    """
    The ranges of parameters and of solves are min and max (2026-10-28),
    as those of priors: the options minimum and maximum of the
    descriptions of parameters (section pdescs), and minimum of the solves
    of parameters (the properties of section params).
    """

    _NEW = dict(minimum='min', maximum='max')

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, pdescs in items(config, 'pdescs'):
            for name, pdesc in pdescs.items():
                changes += self._renamed(pdesc, path + [name])
        for path, params in items(config, 'params'):
            properties = params.get('properties')
            if not isinstance(properties, MutableMapping):
                continue
            for name, value in properties.items():
                if isinstance(value, MutableMapping):
                    changes += self._renamed(
                        value.get('solve'),
                        path + ['properties', name, 'solve'])
        return changes

    def _renamed(self, info: Any, path: list[str | int]) -> list[Change]:
        """Rename minimum and maximum in a dictionary, in place."""
        if not isinstance(info, MutableMapping):
            return []
        changes = []
        for old, new in self._NEW.items():
            if old in info:
                info[new] = info.pop(old)
                changes.append(Change(path, f"{old} -> {new}"))
        return changes


class CoverageFiles(Rule):
    """
    The coverage of a visibilities observable is given by its file as
    {file} (2026-10-28), as the other objects read from files.
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            observable = item.get('observable')
            if isinstance(observable, MutableMapping) \
                    and observable.get('type') == 'visibilities' \
                    and isinstance(observable.get('coverage'), str):
                observable['coverage'] = flow(
                    dict(file=observable['coverage']))
                changes.append(Change(
                    path + ['observable'], "coverage -> coverage {file}"))
        return changes


class DataTypes(Rule):
    """
    The data of a sky observation give their type (2026-10-28), as the
    members of the other families do: that of its observable (the types
    of the datasets are those of their observables).
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            data, observable = item.get('data'), item.get('observable')
            if item.get('type') == 'sky' \
                    and isinstance(data, MutableMapping) \
                    and 'type' not in data \
                    and isinstance(observable, MutableMapping) \
                    and 'type' in observable:
                insert_first(data, 'type', observable['type'])
                changes.append(Change(
                    path + ['data'], f"type: {observable['type']} added"))
        return changes


class Workers(Rule):
    """
    The evaluations of several parameters at once are on workers of a
    kind (2026-10-29): the section 'parallel' ({drivers}) is the section
    'workers' of threads.
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        parallel = config.get('parallel')
        if not isinstance(parallel, MutableMapping) or 'workers' in config:
            return []
        insert_first(parallel, 'type', 'threads')
        rename(config, 'parallel', 'workers')
        return [Change(['parallel'], "parallel -> workers {type: threads}")]


class RingsFromZero(Rule):
    """
    Rings and shells start at the centre (2026-10-31): the option rnmin
    is gone. That of 0 goes; another is a hole (a sharp inner edge) that
    the nodes must now give by hand, as 0 density at the inner nodes.
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, galaxy in items(config, 'galaxies'):
            changes += self._walk(galaxy, path)
        return changes

    def _walk(self, value: Any, path: list[str | int]) -> list[Change]:
        """Rewrite rnmin in a value and its items, in place."""
        changes = []
        if isinstance(value, MutableMapping):
            if 'rnmin' in value:
                if value['rnmin'] == 0:
                    del value['rnmin']
                    changes.append(Change(path, "rnmin: 0 removed"))
                else:
                    changes.append(Change(path, (
                        "by hand: the rings start at 0 (rnmin is gone); "
                        "give rnodes from 0, with 0 density at the nodes "
                        "inside the old rnmin")))
            for key, item in value.items():
                changes += self._walk(item, path + [key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                changes += self._walk(item, path + [i])
        return changes


class HermiteItems(Rule):
    """
    The coefficients of a Gauss-Hermite series are the data items h3 and
    h4 (2026-11-01), not moment3 and moment4, which are central moments:
    those of observations of the method gauss_hermite are renamed.
    """

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            observable = observation.get('observable')
            data = observation.get('data')
            if not (isinstance(observable, Mapping)
                    and observable.get('method') == 'gauss_hermite'
                    and isinstance(data, MutableMapping)):
                continue
            for order in (3, 4):
                if f'moment{order}' in data:
                    data[f'h{order}'] = data.pop(f'moment{order}')
                    changes.append(Change(
                        path + ['data'], f"moment{order} -> h{order}"))
        return changes


# The rules, in the order of the changes
RULES: tuple[Rule, ...] = (
    TypedKinematics(),
    SpectralAxes(),
    VisibilityFields(),
    SpectralTables(),
    QuantityRadii(),
    ImageTraitFiles(),
    NoiseScaleMin(),
    RangeNames(),
    CoverageFiles(),
    DataTypes(),
    Workers(),
    RingsFromZero(),
    HermiteItems(),
)
