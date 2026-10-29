"""
The catalogue of GBKFIT: its families, the base classes whose
configurations choose one of their members by its type (the option
'type'; e.g. the PSFs), and their members.

A member that needs modules GBKFIT does not always have (e.g. the
nautilus fitter, or the cuda driver; see Serializable.requires) is in the
catalogue whether they are installed or not, with what to install.
"""

import dataclasses
from collections.abc import Mapping

from gbkfit import (
    dataset, driver, fitting, instrument, model, observation, parallel,
    params, region)
from gbkfit.model.components import spheroids, traits
from gbkfit.observation.observables import apertures, traces
from gbkfit.utils import parseutils


__all__ = [
    'Family',
    'Member',
    'catalogue',
    'family_of'
]


# The families, by name
_PARSERS: dict[str, parseutils.TypedParser] = dict(
    aperture=region.aperture_parser,
    bootstrap_scheme=fitting.bootstrap_scheme_parser,
    component=model.component_parser,
    cone_kinematics=model.cone_kinematics_parser,
    dataset=dataset.dataset_parser,
    dht=traits.dht_parser,
    disk_kinematics=model.disk_kinematics_parser,
    disk_surface=model.disk_surface_parser,
    disk_vertical=model.disk_vertical_parser,
    distance=model.distance_parser,
    dpt=traits.dpt_parser,
    driver=driver.driver_parser,
    emission=model.emission_parser,
    equilibrium=model.equilibrium_parser,
    extinction_law=model.extinction_law_parser,
    fitter=fitting.fitter_parser,
    gravity=model.gravity_parser,
    halo_response=model.halo_response_parser,
    joint_prior=params.joint_prior_parser,
    lens_mass=observation.lens_mass_parser,
    light_profile=spheroids.light_profile_parser,
    likelihood=observation.likelihood_parser,
    line_of_sight=observation.line_of_sight_parser,
    lsf=instrument.lsf_parser,
    mass_component=model.mass_component_parser,
    noise=observation.noise_parser,
    observable=observation.observable_parser,
    observation=observation.observation_parser,
    opacity=model.opacity_parser,
    param_mode=params.param_mode_parser,
    pdesc=params.pdesc_parser,
    primary_beam=instrument.primary_beam_parser,
    prior=params.prior_parser,
    psf=instrument.psf_parser,
    regions=region.regions_parser,
    rht=traits.rht_parser,
    rpt=traits.rpt_parser,
    slitless_aperture=apertures.aperture_parser,
    solve=params.solve_parser,
    spheroid_kinematics=spheroids.spheroid_kinematics_parser,
    spheroid_profile=model.spheroid_profile_parser,
    spheroid_shape=model.spheroid_shape_parser,
    spt=traits.spt_parser,
    trace=traces.trace_parser,
    vht=traits.vht_parser,
    vpt=traits.vpt_parser,
    weight_trait=observation.weight_trait_parser,
    workers=parallel.workers_parser,
    zpt=traits.zpt_parser)


@dataclasses.dataclass(frozen=True)
class Member:
    """
    A member of a family.

    Parameters
    ----------
    type : str
        Its type (the option 'type' of its configurations).
    cls : type
        Its class.
    missing : Mapping
        The modules it needs that are not installed, each with how to
        install it (see Serializable.requires); none if it is available.
    """

    type: str
    cls: type[parseutils.TypedSerializable]
    missing: Mapping[str, str]

    def available(self) -> bool:
        """Check whether it can be made: the modules it needs are installed."""
        return not self.missing


@dataclasses.dataclass(frozen=True)
class Family:
    """
    A family: a base class whose configurations choose one of its members
    by its type.

    Parameters
    ----------
    name : str
        Its name (e.g. 'psf').
    parser : TypedParser
        The parser of its configurations.
    members : tuple of Member
        Its members, by type.
    """

    name: str
    parser: parseutils.TypedParser
    members: tuple[Member, ...]

    def base(self) -> type[parseutils.TypedSerializable]:
        """Return its base class."""
        return self.parser.cls()


def catalogue() -> dict[str, Family]:
    """
    Return the families of GBKFIT and their members.

    Returns
    -------
    dict
        The families, by name.
    """
    return {name: _family(name, parser) for name, parser in _PARSERS.items()}


def family_of(base: type) -> Family | None:
    """
    Return the family of a base class (e.g. that of PSF, for an option of
    the type PSF), or None if it is the base of none.

    Parameters
    ----------
    base : type
        The class.

    Returns
    -------
    Family or None
        The family.
    """
    for name, parser in _PARSERS.items():
        if parser.cls() is base:
            return _family(name, parser)
    return None


def _family(name: str, parser: parseutils.TypedParser) -> Family:
    """Return a family of the catalogue, of its name and parser."""
    members = tuple(
        Member(type_, cls, parseutils.missing_modules(cls))
        for type_, cls in sorted(parser.registered_classes().items()))
    return Family(name, parser, members)
