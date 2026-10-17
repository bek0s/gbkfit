"""
Tests for the validation of trait options.
"""

import pytest

from gbkfit.model.components import traits
from gbkfit.utils.parseutils import ConfigError


def kernel_consts_cases():
    """
    Each type of trait, options for it, and the constants the kernels
    read from it (traits.hpp): the truncation and the node-wise switch of
    the density heights, the node-wise switch of the other
    heights, the largest displacement of the vertical distortions (and
    the order of a harmonic one), the number of blobs of mixtures, the
    order of harmonics, and the grid of a map. The other traits have
    none.
    """
    parsers = dict(
        rpt=traits.rpt_parser, rht=traits.rht_parser,
        vpt=traits.vpt_parser, vht=traits.vht_parser,
        dpt=traits.dpt_parser, dht=traits.dht_parser,
        zpt=traits.zpt_parser, spt=traits.spt_parser)
    for kind, parser in parsers.items():
        for type_, cls in parser.registered_classes().items():
            if kind == 'rht':
                options, consts = dict(zmax=5.0, rnodes=True), (5.0, True)
            elif kind in ('vht', 'dht'):
                options, consts = dict(), (False,)
            elif kind == 'zpt' and type_.endswith('harmonic'):
                options, consts = dict(zmax=5.0, order=2), (5.0, 2)
            elif kind == 'zpt':
                options, consts = dict(zmax=5.0), (5.0,)
            elif type_.startswith('mixture'):
                options, consts = dict(nblobs=3), (3,)
            elif type_.endswith('harmonic'):
                options, consts = dict(order=2), (2,)
            elif type_ == 'map':
                options, consts = dict(size=[3, 2], step=2.0), (3, 2, 2.0)
            else:
                options, consts = dict(), ()
            yield pytest.param(cls, options, consts, id=f"{kind}-{type_}")


@pytest.mark.parametrize('cls, options, consts', list(kernel_consts_cases()))
def test_traits_give_the_constants_the_kernels_read(cls, options, consts):
    assert cls(**options).consts() == consts


def test_harmonic_orders_are_integers():
    # (a configuration is checked against the annotation; the Python API
    # by the trait)
    with pytest.raises(ConfigError, match="order must be an integer"):
        traits.RPTraitNWHarmonic(order=1.5)
