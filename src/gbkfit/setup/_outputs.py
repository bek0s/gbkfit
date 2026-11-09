"""
The outputs of evaluations of setups, as tasks write them as products of
runs (see gbkfit.outputs): the models and residuals of the observations
(evaluation_outputs), and their data (data_outputs).
"""

import copy
from collections.abc import Mapping
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import Driver
from gbkfit.model import Extra, Extras
from gbkfit.objective import Objective
from gbkfit.observation import GroupPlan, Observation
from gbkfit.outputs import Output, product_extension
from gbkfit.params import ParamValues


__all__ = [
    'data_outputs',
    'evaluation_outputs'
]


# The folders of the extra outputs by their ids, in this order (see
# _extra_output)
_EXTRA_FOLDERS = dict(
    observation='observations', galaxy='galaxies', group='groups',
    component='components')


def data_outputs(group_plan: GroupPlan) -> dict[str, Output]:
    """
    Return the data of the observations of a plan of a group as outputs
    (the images on the drivers of the observations), by path, in the form
    the observations give them (see Observation.output): their values
    (kind 'data', NaN where masked) and errors ('error'), in their units,
    of each data item, and the images of their values and of their PSF
    that the observations make ('data_image'; e.g. the dirty image and
    beam of visibilities, see Observation.images and psf_images), in
    observations/<observation>/<kind>/<item or image>.
    """
    group = group_plan.group()
    outputs = []
    for i, (observation, driver) in enumerate(zip(
            group.observations(), group_plan.observation_drivers())):
        if (dataset := observation.data()) is None:
            continue
        name = group.observation_label(i)
        for key in observation.keys():
            item = dataset[key]
            unit = item.unit()
            outputs.append(_observation_output(
                name, 'data', key, observation.output(item.data(), unit)))
            if item.error() is not None:
                outputs.append(_observation_output(
                    name, 'error', key,
                    observation.output(item.error(), unit)))
        weights = _weights(dataset)
        images = observation.images(
            driver, {key: dataset[key].data() for key in observation.keys()},
            weights, _units(dataset)) | observation.psf_images(driver, weights)
        outputs += [
            _observation_output(name, 'data_image', image, x, 'image')
            for image, x in images.items()]
    return {output.path: output for output in outputs}


def _units(dataset: Any) -> dict[str, Any] | None:
    """Return the units of the data items of a dataset; None without data."""
    return None if dataset is None \
        else {key: x.unit() for key, x in dataset.items()}


def _weights(dataset: Any) -> dict[str, np.ndarray] | None:
    """
    Return the weights of the values of the data items of a dataset, 1 /
    error^2 (0 where they have none); None without data or errors.
    """
    if dataset is None or any(x.error() is None for x in dataset.values()):
        return None
    with np.errstate(divide='ignore'):
        return {
            key: np.where(np.isfinite(x.error()), 1 / x.error() ** 2, 0)
            for key, x in dataset.items()}


def _unit(
        observation: Observation, key: str
) -> astropy.units.UnitBase | None:
    """
    Return the unit of a data item of an observation (which its model
    takes too), if it has data with one; None otherwise.
    """
    data = observation.data()
    return None if data is None else data[key].unit()


def evaluation_outputs(
        group_plan: GroupPlan,
        objective: Objective | None,
        values: ParamValues,
        model: bool,
        residuals: bool,
        extra: bool = True
) -> dict[str, Output]:
    """
    Return the outputs of an evaluation of a plan of a group, by path, in
    the form the observations give them (their images made on their
    drivers), in observations/<observation>/<kind>/<item>
    of each data item: the models (kind 'model'; 'model_mask' and
    'model_weight'), and the residuals ((model - data) / error) without
    and with the weights of the data items ('residual' and 'wresidual';
    these need the objective; the values of the parameters must have
    those of the solved elements that it solves, see Setup.evaluate).
    With the models, a flux scale of several values of an observation
    (see ObservationSky) is 'flux_scale'
    (observations/<observation>/flux_scale), of the shape of its spectra
    but their axis. With the models, the images that the observations
    make of them ('model_image', in place of the item: the name of the
    image; see Observation.images), and of the data minus them
    ('residual_image'), with the weights of the data. Unless left out
    (extra), the extra outputs of the models are 'extra', with their
    titles and descriptions (see _extra_output).
    """
    group = group_plan.group()
    outputs = []
    extras = Extras() if extra else None
    if model:
        # (the plan reuses its arrays of the models, so copy them: the next
        # evaluation would overwrite the outputs)
        models = copy.deepcopy(group_plan.model_h(values, extras))
        for i, data_i in enumerate(models):
            observation = group.observations()[i]
            name = group.observation_label(i)
            for key, value in data_i.items():
                for kind, part in (
                        ('model', 'd'), ('model_mask', 'm'),
                        ('model_weight', 'w')):
                    if value.get(part) is None:
                        continue
                    output = observation.output(
                        value[part],
                        _unit(observation, key) if part == 'd' else None)
                    outputs.append(
                        _observation_output(name, kind, key, output))
            outputs += _images(
                observation, group_plan.observation_drivers()[i], name,
                {k: v['d'] for k, v in data_i.items()})
            if (scale := _flux_scale_map(observation, values, data_i)) \
                    is not None:
                outputs.append(Output(
                    f'observations/{name}/flux_scale.fits', 'flux_scale',
                    scale, dict(observation=name)))
    if residuals:
        if objective is None:
            raise ValueError("the residuals need an objective")
        for kind, weighted in (('residual', False), ('wresidual', True)):
            # (the objective reuses its residual arrays, so copy them;
            # without the models, the extra outputs are those of the
            # first residual evaluation, which evaluates the models)
            data = copy.deepcopy(objective.residual_nddata_h(
                values, weighted, None if model or weighted else extras))
            for i, data_i in enumerate(data):
                name = group.observation_label(i)
                observation = group.observations()[i]
                outputs += [
                    _observation_output(
                        name, kind, key, observation.output(value))
                    for key, value in data_i.items()]
    outputs += [_extra_output(x) for x in extras or ()]
    return {output.path: output for output in outputs}


def _flux_scale_map(
        observation: Observation,
        values: ParamValues,
        model: Mapping[str, Mapping[str, np.ndarray | None]]
) -> np.ndarray | None:
    """
    Return the values of the flux scale of an observation of several
    values (see ObservationSky), of the shape of its model but the axis of
    the scale (e.g. an image of the scale of each spectrum of a cube);
    None for an observation without one.
    """
    if (flux_scale := observation.flux_scale()) is None \
            or flux_scale[1].axis is None:
        return None
    param, scale = flux_scale
    shape = list(model[scale.items[0]]['d'].shape)
    del shape[scale.axis]
    return np.reshape(values[param], shape)


def _images(
        observation: Observation,
        driver: Driver,
        name: str,
        model: Mapping[str, np.ndarray]
) -> list[Output]:
    """
    Return the images that an observation (of the given name in the
    outputs) makes of its model ('model_image') and of its data minus the
    model ('residual_image'; see Observation.images), on its driver.
    """
    dataset = observation.data()
    weights = _weights(dataset)
    units = _units(dataset)
    images = [
        _observation_output(name, 'model_image', image, x, 'image')
        for image, x in observation.images(
            driver, model, weights, units).items()]
    # (an observation that makes no images of its model makes none of the
    # residual either)
    if images and dataset is not None:
        residual = {key: dataset[key].data() - model[key] for key in model}
        images += [
            _observation_output(name, 'residual_image', image, x, 'image')
            for image, x in observation.images(
                driver, residual, weights, units).items()]
    return images


def _extra_output(extra: Extra) -> Output:
    """
    Return an extra output of a model (kind 'extra', with its ids and its
    name as the id 'name'), in the folders of its ids (see
    _EXTRA_FOLDERS; e.g. observations/<observation>/galaxies/<galaxy>/
    components/<component>/<name>), or in observations/<observation>/
    extra/<name> if it is of the observation alone (e.g. the model cube;
    its folder holds the other outputs of the observation by kind).
    """
    folders = []
    for key, folder in _EXTRA_FOLDERS.items():
        if key in extra.ids:
            folders += [folder, extra.ids[key]]
    if set(extra.ids) == {'observation'}:
        folders.append('extra')
    return Output(
        '/'.join(folders + [f'{extra.name}{product_extension(extra.value)}']),
        'extra', extra.value, extra.ids | dict(name=extra.name),
        extra.title, extra.description)


def _observation_output(
        observation: str,
        kind: str,
        name: str,
        value: Any,
        id_name: str = 'item'
) -> Output:
    """
    Return an output of a kind of an observation, of a data item (or of
    another id: e.g. 'image'), in observations/<observation>/<kind>/
    <name>.
    """
    return Output(
        f'observations/{observation}/{kind}/{name}{product_extension(value)}',
        kind, value, {'observation': observation, id_name: name})
