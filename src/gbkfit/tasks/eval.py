
import json
import logging
import time

import astropy.io.fits as fits

import gbkfit
import gbkfit.broker
import gbkfit.dataset
import gbkfit.dmodel
import gbkfit.driver
import gbkfit.gmodel
import gbkfit.model
import gbkfit.params
from . import _detail

log = logging.getLogger(__name__)


def _prepare_config(config):

    _detail.require_config_sections(
        config, ['dmodels', 'gmodels', 'params'])

    _detail.listify_config_sections(
        config, ['brokers', 'drivers', 'datasets', 'dmodels', 'gmodels'])

    _detail.check_config_sections_length(
        config, ['datasets', 'dmodels', 'gmodels'])


def eval_(config):

    log.info("Initializing gbkfit...")
    gbkfit.init()

    log.info(f"Reading configuration file: '{config}'...")
    config = json.load(open(config))
    _prepare_config(config)

    brokers = None
    if config.get('brokers'):
        log.info("Setting up brokers...")
        brokers = gbkfit.broker.parser.load_many(config['brokers'])

    drivers = None
    if config.get('drivers'):
        log.info("Setting up drivers...")
        drivers = gbkfit.driver.parser.load_many(config['drivers'])

    datasets = None
    if config.get('datasets'):
        log.info("Setting up datasets...")
        datasets = gbkfit.dataset.parser.load_many(config['datasets'])

    log.info("Setting up dmodels...")
    dmodels = gbkfit.dmodel.parser.load_many(config['dmodels'], datasets)

    log.info("Setting up gmodels...")
    gmodels = gbkfit.gmodel.parser.load_many(config['gmodels'])

    log.info("Setting up model...")
    model = gbkfit.model.Model(dmodels, gmodels, drivers, brokers)

    log.info("Setting up params...")
    params = gbkfit.params.convert_params_free_to_fixed(config['params'])
    model.set_param_exprs(params)

    for i in range(10):
        log.info("Model evaluation started.")
        t1 = time.time_ns()
        out_dextra = []
        out_gextra = []
        out_params = {}
        out_eparams = {}
        out_eparams_free = {}
        out_eparams_fixed = {}
        out_models = model.evaluate(
            {}, True, True, out_dextra, out_gextra,
            out_params, out_eparams, out_eparams_free, out_eparams_fixed)
        t2 = time.time_ns()
        t_ms = (t2 - t1) // 1000000
        log.info("Model evaluation completed.")
        log.info(f"Elapsed time: {t_ms} ms.")

    def save_output(filename, data):
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)

    log.info("Writing output to the filesystem...")
    for i in range(len(out_models)):
        prefix = 'model' + f'_{i}' * i
        for key, value in out_models[i].items():
            save_output(f'{prefix}_{key}.fits', value)
        for key, value in out_dextra[i].items():
            save_output(f'{prefix}_extra_dmodel_{key}.fits', value)
        for key, value in out_gextra[i].items():
            save_output(f'{prefix}_extra_gmodel_{key}.fits', value)



"""
gbkfit.driver.drivers.DriverHost()
gbkfit.driver.drivers.DriverCUDA()
gbkfit.dmodel.dmodels.DModelSCube()
gbkfit.gmodel.gmodels.GModelKinematics3D()
gbkfit.psflsf.psflsfs.PSFGauss()
gbkfit.psflsf.psflsfs.LSFGauss()
gbkfit.fitter.fitters.FitterDynestyDynamicNS()
gbkfit.fitter.fitters.FitterDynestyStaticNS()


gbkfit.params.params
gbkfit.model.Model()
"""