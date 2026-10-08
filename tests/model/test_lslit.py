"""
Tests for the long-slit data model (lslit): a position-velocity image
along a slit, which must match a row of a spectral cube.
"""

import gbkfit.model
import gbkfit.params
import numpy as np
import pytest


GMODEL = dict(type='kinematics_2d', components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'),
    vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))])

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)


def evaluate(driver, dmodel):
    model_group = gbkfit.model.ModelGroup(gbkfit.model.model_parser.load([
        dict(driver=dict(type=driver.type()), dmodel=dmodel, gmodel=GMODEL)]))
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), PROPERTIES)
    data = model_group.model_h(params.evaluate())[0]
    return next(iter(data.values()))['d'].copy()


@pytest.mark.parametrize('rota', [0, 30])
@pytest.mark.parametrize('psf', [None, dict(type='gauss', sigma=1.5)])
def test_lslit_matches_the_central_row_of_an_scube(driver, psf, rota):
    # The scube has an odd number of rows, so its central row is
    # centred on the slit, which passes through the reference position
    common = dict(rota=rota, psf=psf, lsf=dict(type='gauss', sigma=15))
    scube = evaluate(driver, dict(
        type='scube', size=[32, 41, 51], step=[1, 1, 10], **common))
    lslit = evaluate(driver, dict(
        type='lslit', size=[32, 51], step=[1, 10], **common))
    assert lslit.shape == (51, 32)
    np.testing.assert_allclose(
        lslit, scube[:, 20, :], rtol=1e-4, atol=1e-5 * scube.max())


def test_lslit_objective_residual(driver):
    # Data equal to the model plus 1, with errors of 2: every residual
    # (model - data) / error is -0.5
    from gbkfit.dataset import Data
    from gbkfit.dataset.datasets import DatasetLSlit
    from gbkfit.objective import Objective
    dmodel = dict(type='lslit', size=[32, 51], step=[1, 10], rota=30)
    model = evaluate(driver, dmodel)
    dataset = DatasetLSlit(Data(
        model + 1, error=np.full_like(model, 2), step=(1, 10),
        spectral_axis=1))
    model_group = gbkfit.model.ModelGroup(gbkfit.model.model_parser.load([
        dict(driver=dict(type=driver.type()), dmodel=dmodel, gmodel=GMODEL)],
        dataset=[dataset]))
    objective = Objective([dataset], model_group)
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), PROPERTIES)
    residual = objective.residual_nddata_h(params.evaluate(), False)
    np.testing.assert_allclose(residual[0]['lslit'], -0.5, rtol=1e-5)
