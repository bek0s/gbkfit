
import json
import logging
import time

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml
import scipy.stats as stats

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.params
import gbkfit.params.params
import gbkfit.params.descs
import gbkfit.params.utils
from . import _detail


log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping ordered dicts
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _patch_parameters(info, descs):
    keys, values = gbkfit.params.utils.parse_param_keys(info, descs)[:2]
    info = dict(zip(keys, values))
    recovery_failed = []
    recovery_succeed = []
    for key, val in info.items():
        if isinstance(val, dict):
            if 'val' in val:
                info[key] = val['val']
                recovery_succeed.append(key)
            else:
                recovery_failed.append(key)
    if recovery_succeed:
        log.info(
            f"successfully recovered values "
            f"for the following parameter keys: {recovery_succeed}")
    if recovery_failed:
        raise RuntimeError(
            f"failed to recover values "
            f"for the following parameter keys: {recovery_failed}")
    return info


def eval_(config, perf=None):

    #
    # Read configuration file and
    # perform all necessary validation/patching/preparation
    #

    log.info(f"reading configuration file: '{config}'...")
    cfg = _detail.prepare_config(
        yaml.load(open(config)),
        ('drivers', 'dmodels', 'gmodels', 'params'),
        ('datasets', 'pdescs'))

    #
    # Setup all the components described in the configuration
    #

    datasets = None
    if 'datasets' in cfg:
        log.info("setting up datasets...")
        datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    log.info("setting up drivers...")
    drivers = gbkfit.driver.parser.load(cfg['drivers'])

    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    log.info("setting up models...")
    models = gbkfit.model.make_model_group(dmodels, gmodels, drivers)

    pdescs = None
    if 'pdescs' in cfg:
        log.info("setting up pdescs...")
        pdescs = gbkfit.params.descs.load_descs(cfg['pdescs'])
    pdescs = gbkfit.params.descs.merge_descs(models.pdescs(), pdescs)

    log.info("setting up params...")
    cfg['params']['parameters'] = _patch_parameters(
        cfg['params']['parameters'], pdescs)
    params = gbkfit.params.params.EvalParams.load(cfg['params'], pdescs)

    #
    # Calculate model parameters
    #

    log.info("calculating model parameters...")

    eparams = {ename: None for ename in params.expressions().enames()}
    params = params.expressions().evaluate({}, eparams)

    params_info = _detail.nativify(dict(
        params=params,
        eparams=eparams))
    filename = 'gbkfit_result_params'
    json.dump(params_info, open(f'{filename}.json', 'w+'), indent=2)
    yaml.dump(params_info, open(f'{filename}.yaml', 'w+'))

    #
    # Evaluate models
    #

    log.info("evaluating model...")

    extras = []
    output = models.evaluate_h(params, extras)

    def save_model(file, data):
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(file, overwrite=True)

    log.info("writing model to the filesystem...")
    for i in range(len(output)):
        prefix = 'model' + f'_{i}' * bool(i)
        for key, value in output[i].items():
            save_model(f'{prefix}_{key}.fits', value)
        for key, value in extras[i].items():
            save_model(f'{prefix}_extra_{key}.fits', value)

    """
    import scipy.signal
    from gbkfit.psflsf.psfs import PSFGauss
    from gbkfit.psflsf.lsfs import LSFGauss

    gauss1d = LSFGauss(2).asarray(1)
    gauss2d = PSFGauss(2).asarray((1, 1))
    gauss3d = gauss2d * gauss1d[:, None, None]
    rcube = extras[0]['gmodel_component0_rdata']
    rcube_sm = scipy.signal.fftconvolve(rcube, gauss3d, mode='full')

    fits.writeto('warp_psf_3d.fits', gauss3d, overwrite=True)
    fits.writeto('warp_rcube.fits', rcube, overwrite=True)
    fits.writeto('warp_rcube_sm.fits', rcube_sm, overwrite=True)

    rcube_sm = np.swapaxes(rcube_sm, 1, 2)
    import pyvista as pv
    p = pv.Plotter(
        off_screen=False, window_size=(1024, 768), multi_samples=8,
        line_smoothing=True, point_smoothing=True, polygon_smoothing=True)
    p.enable_anti_aliasing()
    p.enable_depth_peeling(number_of_peels=0, occlusion_ratio=0)
    p.disable_parallel_projection()
    p.add_axes()
    p.set_background('black')
    p.add_volume(
        rcube_sm, cmap='twilight_shifted', n_colors=512, opacity='linear',
        opacity_unit_distance=1, mapper='gpu')
    p.camera_position = [0, 1, 0.2]
    p.show()
    print("BYE")
    exit()
    """

    #
    # Run performance tests
    #

    if perf > 0:
        log.info("running performance test...")
        times = []
        for i in range(perf):
            t1 = time.time_ns()
            models.evaluate_h(params)
            t2 = time.time_ns()
            t_ms = (t2 - t1) / 1000000
            times.append(t_ms)
            log.info(f"evaluation {i}: {t_ms} ms")
        log.info("calculating performance test statistics...")
        time_stats = dict(_detail.nativify(dict(
            min=np.round(np.min(times), 2),
            max=np.round(np.max(times), 2),
            mean=np.round(np.mean(times), 2),
            median=np.round(np.median(times), 2),
            stddev=np.round(np.std(times), 2),
            mad=np.round(stats.median_absolute_deviation(times), 2))))
        log.info(', '.join(f'{k}: {v} ms' for k, v in time_stats.items()))
        time_stats = dict(unit='milliseconds', **time_stats)
        filename = 'gbkfit_result_time'
        json.dump(time_stats, open(f'{filename}.json', 'w+'), indent=2)
        yaml.dump(time_stats, open(f'{filename}.yaml', 'w+'))
