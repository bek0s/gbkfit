"""
Tests for the geometry and velocity conventions of the disk models.
"""

import numpy as np


def kinematics_2d_model(driver, **component):
    """An scube model with one thin smooth disk component."""
    return dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='scube', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=dict(type='kinematics_2d', components=[dict(
            type='smdisk',
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type='tan_uniform'),
            dptraits=dict(type='uniform'),
            **component)]))


def test_loose_disk_systemic_velocity(driver, evaluate_models):
    # A loose disk without rotation: the line-of-sight velocity of the
    # disk must be the systemic velocity everywhere, regardless of the
    # (also node-wise) centre position.
    model = kinematics_2d_model(driver, loose=True, tilted=False)
    properties = dict(
        vsys=50, xpos=3, ypos=-2, posa=0, incl=45,
        bpt_a=1, bpt_s=4, vpt_vt=0, dpt_a=10)
    _, extra = evaluate_models([model], properties)
    velocity = extra['model0_gmodel_component0_vdata']
    on_disk = np.isfinite(velocity)
    assert on_disk.sum() > 100
    np.testing.assert_allclose(velocity[on_disk], 50)
