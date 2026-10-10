"""
Tests for the validation of trait options.
"""

import pytest

from gbkfit.model.components.disks import traits


@pytest.mark.parametrize('angle', [-10, 190])
def test_axis_range_rejects_angles_outside_0_to_180(angle):
    with pytest.raises(RuntimeError, match="angle"):
        traits.WPTraitAxisRange(axis=0, angle=angle, weight=1)


def test_axis_range_accepts_angles_from_0_to_180():
    for angle in (0, 90, 180):
        traits.WPTraitAxisRange(axis=0, angle=angle, weight=1)


def kernel_consts_cases():
    """
    Each type of trait, options for it, and the constants the kernels
    read from it (traits.hpp): the truncation and the node-wise switch of
    the brightness and opacity heights, the node-wise switch of the other
    heights, the number of blobs of mixtures, the order of harmonics, and
    the options of axis_range. The other traits have none.
    """
    parsers = dict(
        bpt=traits.bpt_parser, bht=traits.bht_parser,
        vpt=traits.vpt_parser, vht=traits.vht_parser,
        dpt=traits.dpt_parser, dht=traits.dht_parser,
        zpt=traits.zpt_parser, spt=traits.spt_parser,
        wpt=traits.wpt_parser, opt=traits.opt_parser,
        oht=traits.oht_parser)
    for kind, parser in parsers.items():
        for type_, cls in parser.registered_classes().items():
            if kind in ('bht', 'oht'):
                options, consts = dict(trunc=0, rnodes=True), (0, True)
            elif kind in ('vht', 'dht'):
                options, consts = dict(), (False,)
            elif type_.startswith('mixture'):
                options, consts = dict(nblobs=3), (3,)
            elif type_.endswith('harmonic'):
                options, consts = dict(order=2), (2,)
            elif type_ == 'axis_range':
                options, consts = dict(axis=1, angle=30, weight=2), (1, 30, 2)
            else:
                options, consts = dict(), ()
            yield pytest.param(cls, options, consts, id=f"{kind}-{type_}")


@pytest.mark.parametrize('cls, options, consts', list(kernel_consts_cases()))
def test_traits_give_the_constants_the_kernels_read(cls, options, consts):
    assert cls(**options).consts() == consts
