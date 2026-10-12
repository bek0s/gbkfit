"""
Every parameter of every trait reaches the native kernels: changing it
changes the model. This catches a trait whose parameters in Python
(params_sm, params_rnw) do not match what the kernel reads.
"""

import inspect

import numpy as np
import pytest

from gbkfit.model.components.disks import traits
from modelutils import observation_group


RNODES = list(range(0, 14, 2))

# A value for each parameter name, chosen so that no parameter can leave
# the model unchanged (e.g. an elongated blob at an angle)
VALUES = dict(
    a=1.0, s=3.0, b=2.0, r=4.0, t=40.0, q=0.6, p=30.0, g=1.5,
    rt=2.0, vt=150.0, vr=20.0, vv=10.0, vl=15.0, z0=0.5, rmin=1.0,
    rmax=8.0, n=1.5)

# Each kind of trait of a disk, named after the prefix of its parameters:
# the part of the disk it belongs to (None: the disk itself), its option
# and its parser
KINDS = dict(
    pt=('density', 'ptraits', traits.rpt_parser),
    ht=('density', 'htraits', traits.rht_parser),
    vpt=('kinematics', 'vptraits', traits.vpt_parser),
    vht=('kinematics', 'vhtraits', traits.vht_parser),
    dpt=('kinematics', 'dptraits', traits.dpt_parser),
    dht=('kinematics', 'dhtraits', traits.dht_parser),
    zpt=(None, 'zptraits', traits.zpt_parser),
    spt=(None, 'sptraits', traits.spt_parser))

# The traits of the disks that are not being tested, by kind, with their
# values: those of the disk that emits lines, and of the dusty disk
LINE_DISK_TRAITS = dict(
    pt=(dict(type='exponential'), dict(pt_a=1, pt_s=4)),
    ht=(dict(type='sech2'), dict(ht_s=1)),
    vpt=(dict(type='tan_arctan'), dict(vpt_rt=2, vpt_vt=150)),
    dpt=(dict(type='uniform'), dict(dpt_a=20)))
DUST_DISK_TRAITS = dict(
    pt=(dict(type='exponential'), dict(pt_a=0.05, pt_s=4)),
    ht=(dict(type='sech2'), dict(ht_s=1)))

GEOMETRY = dict(xpos=0, ypos=0, posa=20, incl=60)


def trait_configs(parser):
    """
    A configuration of every type of trait of a parser: with order 2 and
    one blob where they apply, and height traits with and without node-
    wise parameters (rnodes).
    """
    configs = []
    for type_, cls in parser.registered_classes().items():
        # (the trait mass has no parameters: its values come from the
        # mass of its galaxy; see test_mass)
        if cls is traits.VPTraitMass:
            continue
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
    for kind, (_, _, parser) in KINDS.items():
        for config in trait_configs(parser):
            yield pytest.param(kind, False, config, id=f"{kind}-{config}")
    for kind in DUST_DISK_TRAITS:
        for config in trait_configs(KINDS[kind][2]):
            yield pytest.param(
                kind, True, config, id=f"dust_{kind}-{config}")


def thick_disk(name, traits_, **roles):
    """
    The configuration of a smooth thick disk of the given name, traits
    (by kind) and roles.
    """
    disk = dict(
        type='smdisk', name=name, loose=False, tilted=False, rnodes=RNODES,
        **roles)
    for kind, trait in traits_.items():
        part, option, _ = KINDS[kind]
        if part is None:
            disk[option] = trait
        else:
            disk.setdefault(part, {})[option] = trait
    return disk


def traits_and_values(default_traits, tested_kind, tested_trait):
    """
    The traits of a disk, by kind, and the values of their parameters:
    the default traits, but that of the tested kind (if any), which is the
    tested trait (without values).
    """
    traits_ = {}
    values = {}
    for kind, (trait, trait_values) in default_traits.items():
        if kind != tested_kind:
            traits_[kind] = trait
            values |= trait_values
    if tested_kind is not None:
        traits_[tested_kind] = tested_trait
    return traits_, values


def prefixed(prefix, values):
    return {f'{prefix}_{name}': value for name, value in values.items()}


def case_and_properties(driver, kind, dust, config):
    """
    A galaxy seen as a spectral cube on a grid: a smooth thick disk that
    emits lines ('disk'), and with dust a dusty one ('dust'). The traits
    of the given kind of the dusty disk (with dust), or else of the disk
    that emits lines, are the given trait. Return the case, the parameter
    properties, and the prefix of the parameters of the trait.
    """
    disk_traits, disk_values = traits_and_values(
        LINE_DISK_TRAITS, None if dust else kind, config)
    components = [
        thick_disk('disk', disk_traits, emission=dict(type='lines'))]
    properties = prefixed('disk', GEOMETRY | dict(vsys=0) | disk_values)
    if dust:
        dust_traits, dust_values = traits_and_values(
            DUST_DISK_TRAITS, kind, config)
        components.append(
            thick_disk('dust', dust_traits, opacity=dict(type='dust')))
        properties |= prefixed('dust', GEOMETRY | dust_values)
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[32, 32, 40], step=[1, 1, 10]),
        line_of_sight=dict(type='grid'),
        galaxy=dict(components=components))
    tested = 'dust' if dust else 'disk'
    return case, properties, f'{tested}_{kind}_'


@pytest.mark.parametrize('kind, dust, config', list(cases()))
def test_every_trait_parameter_changes_the_model(
        driver, evaluate_cases, kind, dust, config):
    case, properties, prefix = case_and_properties(
        driver, kind, dust, config)
    pdescs = observation_group([case]).pdescs()
    names = [name for name in pdescs if name.startswith(prefix)]
    for name in names:
        size = pdescs[name].size()
        value = VALUES[name[len(prefix):]]
        properties[name] = value if pdescs[name].type() == 'scalar' \
            else [value] * size

    def evaluate(props):
        data, _ = evaluate_cases([case], props)
        return data[0]['spectra']['d'].copy()

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
