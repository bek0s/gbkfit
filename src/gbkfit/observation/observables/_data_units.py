"""
The units of the light of the data of the observables (see
Observable.data_unit): of the pixels of grids, and of fluxes, sums of
the light of pixels (e.g. of regions, or visibilities).
"""

import astropy.units

import gbkfit.units
from gbkfit.dataset import Dataset
from gbkfit.model import DataUnit, LightUnit


def data_unit(
        dataset: Dataset | None,
        key: str | None,
        pixel: float,
        integrated: bool,
        light: LightUnit | None = None
) -> DataUnit | None:
    """
    Return the unit of the light of a dataset (see Observable.data_unit):
    that of its item of the given key (None: it has none), with its beam,
    whose pixels have the given solid angle (arcsec^2), and whose light
    is integrated over the lines or not; without data, that of the given
    light, if any; None without either, or without an item of light.
    """
    if key is None:
        return None
    if dataset is None:
        return None if light is None else DataUnit(
            light.unit, pixel, light.beam, light.beam_spectral, integrated)
    data = dataset[key]
    beam = data.beam()
    return DataUnit(
        data.unit(), pixel,
        None if beam is None else beam.area(),
        None if beam is None else beam.spectral, integrated)


def flux_data_unit(
        dataset: Dataset | None,
        key: str | None,
        integrated: bool,
        light: LightUnit | None = None
) -> DataUnit | None:
    """
    Return the unit of the light of a dataset of fluxes (see data_unit;
    e.g. of regions, or visibilities): the sums of the surface brightness
    of the model over pixels, times the area of each (see flux_weights),
    so the model takes the unit per arcsec^2 (see flux_per_area).
    """
    result = data_unit(dataset, key, 1.0, integrated, light)
    if result is None or result.unit is None:
        return result
    return result._replace(
        unit=flux_per_area(result.unit), beam=None, beam_spectral=None)


def flux_per_area(unit: astropy.units.UnitBase) -> astropy.units.UnitBase:
    """
    Return the unit of light per arcsec^2 whose sums over the area of
    regions are data of the given unit, a flux (e.g. Jy, 1e-20
    erg/s/cm2/Angstrom, Jy km/s). Raise ValueError if the unit is per
    beam or per solid angle (the light of a pixel, not of a region).
    """
    if gbkfit.units.spatial_unit(unit) != astropy.units.dimensionless_unscaled:
        raise ValueError(
            f"the data of regions are fluxes, sums over their pixels: their "
            f"unit is one of flux (e.g. 'Jy', '1e-20 erg/s/cm2/Angstrom', "
            f"'Jy km/s'); {gbkfit.units.format_unit(unit)!r} is not")
    return unit / astropy.units.arcsec ** 2
