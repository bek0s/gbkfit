"""
Tests that the constructors that configurations call declare their
options: each parameter is annotated, and checked first, by the
constructor named by its class (see parseutils.check_options).
"""

import ast
import inspect
import textwrap

import pytest

import gbkfit.dataset
import gbkfit.fitting
import gbkfit.instrument
import gbkfit.model
import gbkfit.model.components.disks
import gbkfit.model.components.parts
import gbkfit.model.components.roles
import gbkfit.model.components.spheroids
import gbkfit.model.components.traits
import gbkfit.model.mass
import gbkfit.params
import gbkfit.region
from gbkfit.observation.observables import traces
from gbkfit.utils import funcutils, typeutils


# The families whose options are declared, by their parsers or classes
PARSERS = [
    gbkfit.dataset.dataset_parser,
    gbkfit.fitting.bootstrap_scheme_parser,
    gbkfit.fitting.fitter_parser,
    gbkfit.instrument.lsf_parser,
    gbkfit.instrument.primary_beam_parser,
    gbkfit.instrument.psf_parser,
    gbkfit.model.components.traits.dht_parser,
    gbkfit.model.components.traits.dpt_parser,
    gbkfit.model.components.traits.rht_parser,
    gbkfit.model.components.traits.rpt_parser,
    gbkfit.model.components.traits.spt_parser,
    gbkfit.model.components.traits.vht_parser,
    gbkfit.model.components.traits.vpt_parser,
    gbkfit.model.components.traits.zpt_parser,
    gbkfit.model.components.roles.emission_parser,
    gbkfit.model.components.roles.opacity_parser,
    gbkfit.model.components.parts.cone_kinematics_parser,
    gbkfit.model.components.spheroids.light_profile_parser,
    gbkfit.model.components.spheroids.spheroid_kinematics_parser,
    gbkfit.model.components.parts.disk_kinematics_parser,
    gbkfit.model.components.parts.equilibrium_parser,
    gbkfit.model.component_parser,
    gbkfit.model.distance_parser,
    gbkfit.model.extinction_law_parser,
    gbkfit.model.gravity_parser,
    gbkfit.model.mass.halo_response_parser,
    gbkfit.model.mass.disk_surface_parser,
    gbkfit.model.mass.disk_vertical_parser,
    gbkfit.model.mass.mass_component_parser,
    gbkfit.model.mass.spheroid_profile_parser,
    gbkfit.model.mass.spheroid_shape_parser,
    gbkfit.params.joint_prior_parser,
    gbkfit.params.prior_parser,
    gbkfit.region.aperture_parser,
    gbkfit.region.regions_parser,
    traces.trace_parser]
CLASSES = [
    gbkfit.fitting.BootstrapOptions,
    gbkfit.fitting.UncertaintyOptions,
    gbkfit.instrument.Instrument,
    gbkfit.instrument.SpectralTable,
    gbkfit.model.Band,
    gbkfit.model.Galaxy,
    gbkfit.model.components.disks.Clouds,
    gbkfit.model.components.parts.Density,
    gbkfit.model.Geometry,
    gbkfit.model.Line,
    gbkfit.model.mass.DarkFraction,
    gbkfit.model.mass.MassModel]


def constructors():
    """The constructors that configurations call, of each class."""
    classes = [*CLASSES, *(
        cls for parser in PARSERS
        for cls in parser.registered_classes().values())]
    for cls in classes:
        for name in cls.config_constructors:
            func = cls.__init__ if name == '__init__' else getattr(cls, name)
            # (a class without options of its own has that of object)
            if func is not object.__init__:
                yield pytest.param(func, id=f'{cls.__qualname__}.{name}')


@pytest.mark.parametrize('func', list(constructors()))
def test_constructors_declare_their_options(func):
    params = funcutils.parameter_names(func).all
    assert set(params) <= set(typeutils.option_types(func))
    body = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant):
        body = body[1:]
    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    callee = first.value.func
    assert getattr(callee, 'attr', getattr(callee, 'id', None)) == (
        'check_options')
    assert ast.unparse(first.value.args[0]) == func.__qualname__
