
import json
import logging
import time

import ruamel.yaml

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.fitting.fitter
import gbkfit.fitting.objective
import gbkfit.model
import gbkfit.params
import gbkfit.params.descs
import gbkfit.params.utils
from . import _detail

import numpy as np
import astropy.wcs
import astropy.io.fits as fits

log = logging.getLogger(__name__)

# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping ordered dicts
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


class Foo:
    def __call__(self, objective, params):
        print('foo:', params)


callback = Foo()


def foobar(filename, data, cdelt=None, crpix=None, crval=None, crota=None, overwrite=False):
    crota = np.radians(crota)
    wcs = astropy.wcs.WCS(naxis=data.ndim, relax=False)
    if cdelt is not None:
        wcs.wcs.cdelt = cdelt
    if crpix is not None:
        wcs.wcs.crpix = crpix
    if crval is not None:
        wcs.wcs.crval = crval
    if crota is not None:
        wcs.wcs.pc = np.identity(data.ndim)
        wcs.wcs.pc[0][0] = +np.cos(crota)
        wcs.wcs.pc[0][1] = -np.sin(crota)
        wcs.wcs.pc[1][0] = +np.sin(crota)
        wcs.wcs.pc[1][1] = +np.cos(crota)
    fits.writeto(
        filename, data, header=wcs.to_header(),
        output_verify='exception', overwrite=overwrite, checksum=True)

    eleos = fits.getheader(filename)
    wcs1 = astropy.wcs.WCS(eleos)
    print(wcs1)



def fit(config):

    #data = np.ones((51, 51, 51))

    #foobar('foo.fits', data, [10, 10, 5], crpix=[10, 10, 10], crval=[0, 0, 0], rota=45.0, overwrite=True)

    #exit()

    #
    # Read configuration file and
    # perform all necessary validation/patching/preparation
    #

    log.info(f"reading configuration file: '{config}'...")
    cfg = _detail.prepare_config(
        yaml.load(open(config)),
        ('datasets', 'drivers', 'dmodels', 'gmodels', 'params', 'fitter'),
        ('pdescs',))

    #
    # Setup all the components described in the configuration
    #

    log.info("setting up datasets...")
    datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    log.info("setting up drivers...")
    drivers = gbkfit.driver.driver.parser.load(cfg['drivers'])

    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    log.info("setting up model...")
    model = gbkfit.model.Model(dmodels, gmodels, drivers)

    log.info("setting up fitter...")
    fitter = gbkfit.fitting.fitter.parser.load(cfg['fitter'])

    log.info("setting up objective...")
    objective = gbkfit.fitting.objective.Objective(datasets, model)

    pdescs = None
    if 'pdescs' in cfg:
        log.info("setting up pdescs...")
        pdescs = gbkfit.params.descs.load_descriptions(cfg['pdescs'])
    pdescs = gbkfit.params.descs.merge_descriptions(model.pdescs(), pdescs)

    log.info("setting up params...")
    params = fitter.load_params(cfg['params'], pdescs)

    #
    # Perform fit
    #

    log.info("model-fitting started")
    t1 = time.time_ns()
    result = fitter.fit(objective, params)
    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1000000
    log.info(f"model-fitting completed (elapsed time: {t_ms} ms)")

    from gbkfit.fitting.result import dump_result

    dump_result('out', result)

    #print(result)
