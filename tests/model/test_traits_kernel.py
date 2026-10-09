"""
Every parameter of every trait reaches the native kernels: changing it
changes the model. This catches a trait whose parameters in Python
(params_sm, params_rnw) do not match what the kernel reads.
"""

import inspect

import numpy as np
import pytest

from gbkfit.model.gmodels import traits
from modelutils import observation_group


RNODES = list(range(0, 14, 2))

# A value for each parameter name, chosen so that no parameter can leave
# the model unchanged (e.g. an elongated blob at an angle)
VALUES = dict(
    a=1.0, s=3.0, b=2.0, r=4.0, t=40.0, q=0.6, p=30.0, g=1.5,
    rt=2.0, vt=150.0, vr=20.0, vv=10.0, vl=15.0, z0=0.5, rmin=1.0,
    rmax=8.0)

# The traits of the disk that are not being tested, with their values
DEFAULT_TRAITS = dict(
    bptraits=(dict(type='exponential'), dict(bpt_a=1, bpt_s=4)),
    bhtraits=(dict(type='sech2'), dict(bht_s=1)),
    vptraits=(dict(type='tan_arctan'), dict(vpt_rt=2, vpt_vt=150)),
    dptraits=(dict(type='uniform'), dict(dpt_a=20)))

# The trait kinds tested in the spectral disk, and their parsers
SPECTRAL_KINDS = dict(
    bptraits=traits.bpt_parser, bhtraits=traits.bht_parser,
    vptraits=traits.vpt_parser, vhtraits=traits.vht_parser,
    dptraits=traits.dpt_parser, dhtraits=traits.dht_parser,
    zptraits=traits.zpt_parser, sptraits=traits.spt_parser)


def trait_configs(parser):
    """
    A configuration of every type of trait of a parser: with order 2 and
    one blob where they apply, and height traits with and without node-
    wise parameters (rnodes).
    """
    configs = []
    for type_, cls in parser._parsers.items():
        options = inspect.signature(cls.__init__).parameters
        config = dict(type=type_)
        if 'order' in options:
            config['order'] = 2
        if 'nblobs' in options:
            config['nblobs'] = 1
        if 'rnodes' in options:
            configs += [
                config | dict(rnodes=False), config | dict(rnodes=True)]
        else:
            configs.append(config)
    return configs


def cases():
    kinds = SPECTRAL_KINDS | dict(
        optraits=traits.opt_parser, ohtraits=traits.oht_parser)
    for key, parser in kinds.items():
        for config in trait_configs(parser):
            yield pytest.param(key, config, id=f"{key}-{config}")


def model_and_properties(driver, key, config):
    """
    A kinematics_3d model with one smooth disk whose traits of the given
    kind (key) are the given trait, and the parameter properties of the
    model. Opacity traits go in an opacity component. Return the model,
    the properties, and the names of the parameters of the trait.
    """
    geometry = dict(xpos=0, ypos=0, posa=20, incl=60, vsys=0)
    disk = dict(type='smdisk', loose=False, tilted=False, rnodes=RNODES)
    properties = dict(geometry)
    if key in ('optraits', 'ohtraits'):
        component = disk | {
            name: trait for name, (trait, _) in DEFAULT_TRAITS.items()}
        for _, values in DEFAULT_TRAITS.values():
            properties |= values
        opacity = dict(disk, optraits=dict(type='exponential'),
                       ohtraits=dict(type='sech2'))
        opacity[key] = config
        properties |= dict(
            ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=20, ocmp_incl=60,
            ocmp_opt_a=0.05, ocmp_opt_s=4, ocmp_oht_s=1)
        prefix = 'ocmp_' + key[:3] + '_'
        gmodel = dict(
            type='kinematics_3d', components=[component],
            opacity_components=[opacity])
    else:
        component = dict(disk)
        for name, (trait, values) in DEFAULT_TRAITS.items():
            if name != key:
                component[name] = trait
                properties |= values
        component[key] = config
        prefix = key[:3] + '_'
        gmodel = dict(type='kinematics_3d', components=[component])
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='scube', size=[32, 32, 40], step=[1, 1, 10]),
        gmodel=gmodel)
    return model, properties, prefix


@pytest.mark.parametrize('key, config', list(cases()))
def test_every_trait_parameter_changes_the_model(
        driver, evaluate_models, key, config):
    model, properties, prefix = model_and_properties(driver, key, config)
    pdescs = observation_group([model]).pdescs()
    names = [name for name in pdescs if name.startswith(prefix)]
    for name in names:
        size = pdescs[name].size()
        value = VALUES[name[len(prefix):]]
        properties[name] = value if pdescs[name].type() == 'scalar' \
            else [value] * size

    def evaluate(props):
        data, _ = evaluate_models([model], props)
        return data[0]['scube']['d'].copy()

    base = evaluate(properties)
    assert np.isfinite(base).all()
    unchanged = []
    for name in names:
        changed = dict(properties)
        changed[name] = np.multiply(properties[name], 1.25) + 0.1
        data = evaluate(changed)
        assert np.isfinite(data).all(), \
            f"changing {name} gives non-finite values"
        if not np.abs(data - base).max() > 1e-6 * np.abs(base).max():
            unchanged.append(name)
    assert not unchanged, \
        f"these parameters do not change the model: {unchanged}"
