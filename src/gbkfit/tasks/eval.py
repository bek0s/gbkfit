
import json
import logging
import numbers
import time

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml as yaml
import scipy.stats as stats

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.params
import gbkfit.params.descs
import gbkfit.params.interpreter
import gbkfit.params.utils
import gbkfit.utils.miscutils
from . import _detail


log = logging.getLogger(__name__)


# This is needed for dumping dicts in the correct order
yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_pdescs(gmodels, extra_pdescs):
    pdescs, mappings = gbkfit.utils.miscutils.merge_dicts_and_make_mappings(
        [gmodel.params() for gmodel in gmodels], 'model')
    if extra_pdescs:
        duplicates = set(pdescs).intersection(extra_pdescs)
        if duplicates:
            raise RuntimeError(
                f"the following parameter descriptions are present "
                f"in both the model and pdescs: {str(duplicates)}")
        pdescs.update(extra_pdescs)
    return pdescs, mappings


def _prepare_params(info, descs):
    keys, values = gbkfit.params.utils.parse_param_keys(info, descs)[:2]
    info = {}
    recovery_failed = []
    recovery_succeed = {}
    for key, value in zip(keys, values):
        if isinstance(value, dict):
            value = value.get('init')
            if value is None:
                recovery_failed.append(key)
                continue
            recovery_succeed[key] = value
        info[key] = value
    if recovery_succeed:
        log.info(
            f"successfully recovered values "
            f"for the following keys: {recovery_succeed}")
    if recovery_failed:
        raise RuntimeError(
            f"failed to recover values "
            f"for the following keys: {recovery_failed}")
    values, exprs = gbkfit.params.utils.parse_param_values(
        info, descs, lambda x: isinstance(x, numbers.Number))[4:]
    return values, exprs


def eval_(config, perf=None):
    """
    import astropy.wcs
    import matplotlib.pyplot as plt
    import gbkfit.math.math

    size = (21, 36)
    data1x = np.zeros(size[::-1])
    data1y = np.zeros(size[::-1])
    data2x = np.zeros(size[::-1])
    data2y = np.zeros(size[::-1])
    data3x = np.zeros(size[::-1])
    data3y = np.zeros(size[::-1])
    data4x = np.zeros(size[::-1])
    data4y = np.zeros(size[::-1])

    step = (1.8, 2.1)
    rpix = (12.0, 17.0)
    rval = (10.0, 20.0)
    rota = np.radians(65.0)

    header = dict(
        CDELT1=step[0],
        CDELT2=step[1],
        CRPIX1=rpix[0],
        CRPIX2=rpix[1],
        CRVAL1=rval[0],
        CRVAL2=rval[1],
        PC1_1=np.cos(rota),
        PC1_2=np.sin(rota),
        PC2_1=-np.sin(rota),
        PC2_2=np.cos(rota))

    wcs = astropy.wcs.WCS(header)
    for y in range(size[1]):
        for x in range(size[0]):
                data1x[y, x] = wcs.all_pix2world([[x, y]], 1)[0][0]
                data1y[y, x] = wcs.all_pix2world([[x, y]], 1)[0][1]
                xn = (x - rpix[0])
                yn = (y - rpix[1])
                xn, yn = gbkfit.math.math.transform_lh_rotate_z(xn, yn, rota)
                xn = xn * step[0] + rval[0]
                yn = yn * step[1] + rval[1]
                data2x[y, x] = xn
                data2y[y, x] = yn

    datarx = data1x - data2x
    datary = data1y - data2y

    zero1 = [data1x[0, 0], data1y[0, 0]]
    zero2 = [data2x[0, 0], data2y[0, 0]]
    print(zero1)
    print(zero2)

    rpix = (0, 0)
    rval1 = zero1
    rval2 = zero2

    header = dict(
        CDELT1=step[0],
        CDELT2=step[1],
        CRPIX1=rpix[0],
        CRPIX2=rpix[1],
        CRVAL1=rval1[0],
        CRVAL2=rval1[1],
        PC1_1=np.cos(rota),
        PC1_2=np.sin(rota),
        PC2_1=-np.sin(rota),
        PC2_2=np.cos(rota))

    wcs = astropy.wcs.WCS(header)
    for y in range(size[1]):
        for x in range(size[0]):
            data3x[y, x] = wcs.all_pix2world([[x, y]], 1)[0][0]
            data3y[y, x] = wcs.all_pix2world([[x, y]], 1)[0][1]
            xn = (x - rpix[0])
            yn = (y - rpix[1])
            xn, yn = gbkfit.math.math.transform_lh_rotate_z(xn, yn, rota)
            xn = xn * step[0] + rval2[0]
            yn = yn * step[1] + rval2[1]
            data4x[y, x] = xn
            data4y[y, x] = yn

    zero3 = [data3x[0, 0], data3y[9, 0]]
    zero4 = [data4x[0, 0], data4y[9, 0]]
    print(zero3)
    print(zero4)

    fig = plt.figure(figsize=(10, 10))
    ax1x = fig.add_subplot(2, 3, 1)
    ax2x = fig.add_subplot(2, 3, 2)
    axrx = fig.add_subplot(2, 3, 3)
    ax1y = fig.add_subplot(2, 3, 4)
    ax2y = fig.add_subplot(2, 3, 5)
    axry = fig.add_subplot(2, 3, 6)
    ax1x.imshow(data1x, interpolation='nearest', origin='bottom left')
    ax2x.imshow(data2x, interpolation='nearest', origin='bottom left')
    axrx.imshow(datarx, interpolation='nearest', origin='bottom left')
    ax1y.imshow(data1y, interpolation='nearest', origin='bottom left')
    ax2y.imshow(data2y, interpolation='nearest', origin='bottom left')
    axry.imshow(datary, interpolation='nearest', origin='bottom left')

    plt.show()

    exit()
    """
    """
    import abc

    class TypeSupport(abc.ABC):
        @staticmethod
        @abc.abstractmethod
        def type():
            pass

    class DescSupport(abc.ABC):
        @staticmethod
        @abc.abstractmethod
        def desc():
            pass

    class Model(TypeSupport, DescSupport):
        @staticmethod
        def type():
            return 'model'

        @staticmethod
        def desc():
            return 'model description'

    model = Model()

    exit()
    """

    #exit()


    #
    # Read configuration file and
    # perform all necessary validation/patching/preparation
    #

    log.info(f"reading configuration file: '{config}'...")
    config = _detail.prepare_config(
        yaml.YAML().load(open(config)),
        ('drivers', 'dmodels', 'gmodels', 'params'),
        ('datasets', 'pdescs'))

    #
    # Setup all the components described in the configuration
    #

    datasets = None
    if config.get('datasets'):
        log.info("setting up datasets...")
        datasets = gbkfit.dataset.dataset_parser.load_many(config['datasets'])

    drivers = None
    if config.get('drivers'):
        log.info("setting up drivers...")
        drivers = gbkfit.driver.driver.parser.load_many(config['drivers'])

    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load_many(config['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load_many(config['gmodels'])

    pdescs = None
    if config.get('pdescs'):
        log.info("setting up pdescs...")
        pdesc_keys = config['pdescs'].keys()
        pdesc_vals = config['pdescs'].values()
        pdesc_list = gbkfit.params.descs.parser.load_many(pdesc_vals)
        pdescs = dict(zip(pdesc_keys, pdesc_list))
    pdescs_all, pdescs_mappings = _prepare_pdescs(gmodels, pdescs)

    log.info("setting up params...")
    values, exprs = _prepare_params(config['params'], pdescs_all)

    #
    # Calculate model parameters
    #

    interpreter = gbkfit.params.interpreter.ParamInterpreter(pdescs_all, exprs)
    params_all = interpreter.evaluate(values)
    params_list = [{param: params_all[mapping[param]] for param in mapping}
                   for mapping in pdescs_mappings]

    filename = 'gbkfit_result_params'
    params_all = _detail.nativify(params_all)
    json.dump(params_all, open(f'{filename}.json', 'w+'))
    yaml.dump(params_all, open(f'{filename}.yaml', 'w+'))

    #
    # Evaluate models
    #

    log.info("evaluating model...")
    models = []
    extras = []
    for driver, dmodel, gmodel, params in zip(
            drivers, dmodels, gmodels, params_list):
        extra = {}
        model_d = dmodel.evaluate(driver, gmodel, params, extra)
        model_h = {k: driver.mem_copy_d2h(v) for k, v in model_d.items()}
        models.append(model_h)
        extras.append(extra)

    def save_model(file, data):
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(file, overwrite=True)

    log.info("writing model to the filesystem...")
    for i in range(len(models)):
        prefix = 'model' + f'_{i}' * bool(i)
        for key, value in models[i].items():
            save_model(f'{prefix}_{key}.fits', value)
            print(np.sum(value))
        for key, value in extras[i].items():
            save_model(f'{prefix}_extra_{key}.fits', value)

    #
    # Run performance tests
    #

    if perf > 0:
        log.info("running performance test...")
        times = []
        for i in range(perf):
            t1 = time.time_ns()
            for driver, dmodel, gmodel, params in zip(
                    drivers, dmodels, gmodels, params_list):
                model_d = dmodel.evaluate(driver, gmodel, params)
                _ = {k: driver.mem_copy_d2h(v) for k, v in model_d.items()}
            t2 = time.time_ns()
            t_ms = (t2 - t1) // 1000000
            times.append(t_ms)
            log.info(f"evaluation {i}: {t_ms} ms")
        log.info("calculating performance test statistics...")
        time_stats = _detail.nativify(dict(
            min=np.round(np.min(times), 1),
            max=np.round(np.max(times), 1),
            mean=np.round(np.mean(times), 1),
            stddev=np.round(np.std(times), 1),
            median=np.round(np.median(times), 1),
            mad=np.round(stats.median_absolute_deviation(times), 1)))
        log.info(', '.join(f'{k}: {v} ms' for k, v in time_stats.items()))
        filename = 'gbkfit_result_time'
        json.dump(time_stats, open(f'{filename}.json', 'w+'))
        yaml.dump(time_stats, open(f'{filename}.yaml', 'w+'))
