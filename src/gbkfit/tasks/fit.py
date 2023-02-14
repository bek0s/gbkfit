import copy
import logging
import os
import time

import numpy as np
import ruamel.yaml

import gbkfit.dataset
import gbkfit.driver
import gbkfit.fitting
import gbkfit.model
import gbkfit.objective
import gbkfit.params
from gbkfit.utils import miscutils
from . import _detail


_log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping dicts with correct order
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_params(info, pdescs):

    # Prepare the supplied parameters. This will:
    # - Ensure the parameter keys are valid
    # - Explode parameter values that are dicts and can be exploded
    parameters = gbkfit.params.prepare_param_info(
        info.get('parameters'), pdescs)

    # Update parameter info and return it
    return info | dict(parameters=parameters)


def fit(config):

    # class Foo:
    #
    #     def __init__(self):
    #         self._array1 = np.ones((2, 2))
    #         self._array2 = self._array1.reshape((4,))
    #
    #     def check_arrays(self):
    #         print("array1 address: ", self._array1.data)
    #         print("array2 address: ", self._array2.data)
    #         print("share memory:", np.shares_memory(self._array1, self._array2))
    #
    # a = Foo()
    # a.check_arrays()
    # b = copy.deepcopy(a)
    # b.check_arrays()
    #
    #
    # exit()

    #
    # Read configuration file and
    # perform all necessary validation/preparation
    #

    config = os.path.abspath(config)
    _log.info(f"reading configuration from file: {config}...")

    try:
        cfg = yaml.load(open(config))
    except Exception as e:
        raise RuntimeError(
            f"error while reading configuration file {config}; "
            f"see reason below:\n{e}") from e

    _log.info("preparing configuration...")
    # This is not a full-fledged validation. It just tries to catch
    # and inform the user about the really obvious mistakes.
    # todo: investigate the potential use of jsonschema for validation
    required_sections = (
        'datasets', 'drivers', 'dmodels', 'gmodels', 'params', 'fitter')
    optional_sections = ('pdescs', 'objective')
    cfg = _detail.prepare_config(cfg, required_sections, optional_sections)

    #
    # Setup all the components described in the configuration.
    # We assume that the datasets, drivers, dmodels, and gmodels
    # configurations are lists, while the objective, pdescs, and params
    # configurations are dicts. This assumption is based on the fact
    # that the _detail.prepare_config() called above should do that
    #

    _log.info("setting up datasets...")
    datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    _log.info("setting up drivers...")
    drivers = gbkfit.driver.driver_parser.load(cfg['drivers'])

    _log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    _log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    _log.info("setting up model...")
    model = gbkfit.model.Model(drivers, dmodels, gmodels)

    _log.info("setting up objective...")
    objective = gbkfit.objective.objective_parser.load(
        cfg.get('objective', {}), datasets, model)

    _log.info("setting up fitter...")
    fitter = gbkfit.fitting.fitter_parser.load(cfg['fitter'])

    pdescs = None
    if 'pdescs' in cfg:
        _log.info("setting up pdescs...")
        pdescs = gbkfit.params.pdescs.load_pdescs_dict(cfg['pdescs'])
    pdescs = _detail.merge_pdescs(objective.pdescs(), pdescs)

    _log.info("setting up params...")
    cfg['params'] = _prepare_params(cfg['params'], pdescs)

    print(cfg['params'])

    params = fitter.load_params(cfg['params'], pdescs)

    #
    # Perform fit
    #

    _log.info("model-fitting started")
    t1 = time.time_ns()
    result = fitter.fit(objective, params)
    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1_000_000
    _log.info(f"model-fitting completed (elapsed time: {t_ms} ms)")

    output_dir = os.path.abspath(miscutils.make_unique_path('out'))
    _log.info(f"saving result under '{output_dir}'...")
    gbkfit.fitting.result.dump_result(output_dir, result)
    print(result.summary())
