import abc

from gbkfit.model.core import GModel
from gbkfit.utils import iterutils, parseutils
from . import _detail


__all__ = [
    'GModel2D',
    'GModel3D'
]


# The prefix of the parameters of the components and the opacity
# components, and whether the parameters of the first one have it (see
# _detail.make_component_params)
_CMP_PREFIX = ('cmp', False)
_OCMP_PREFIX = ('ocmp', True)


class GModel2D(GModel, abc.ABC):
    """
    A gmodel that adds up components on a 2d spatial grid: the x and y
    axes of the data. Its components get a 3d grid with a z axis of size
    1.

    A subclass declares the parser of its components (_cmp_parser), and
    its options (__init__).
    """

    _cmp_parser: parseutils.TypedParser

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        parseutils.load_option_and_update_info(
            cls._cmp_parser, info, 'components',
            required=True, allow_none=False)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def __init__(self, components):
        if not components:
            raise RuntimeError("at least one component must be configured")
        self._components = iterutils.tuplify(components, False)
        self._params, self._mappings = _detail.make_component_params(
            self._components, *_CMP_PREFIX)
        # The spatial grid: the x and y axes of the data
        self._size = None
        self._step = None
        self._zero = None
        self._wdata = None
        self._dtype = None
        self._driver = None
        self._backend = None

    def pdescs(self):
        return self._params

    def has_weights(self):
        return any(cmp.has_weights() for cmp in self._components)

    def constants(self):
        return _detail.make_component_constants(
            self._components, *_CMP_PREFIX)

    def _prepare(self, driver, weights, size, step, zero, dtype):
        self._driver = driver
        self._size = tuple(size[:2])
        self._step = tuple(step[:2])
        self._zero = tuple(zero[:2])
        self._dtype = dtype
        # The spatial weights, if weighting is requested
        self._wdata = None
        if weights is not None:
            self._wdata = driver.mem_alloc_d(self._size[::-1], dtype)
        self._backend = driver.native_class('GModel', dtype)()

    def _evaluate(
            self, driver, params, outputs, spectral_axis, weights,
            size, step, zero, rota, dtype, out_extra):
        """
        Evaluate the gmodel on the spatial grid of the data (size, step and
        zero) and the given spectral axis (size, step and zero). outputs
        has the array the components add to: the 'image' or the 'scube'.
        """
        if (self._driver is not driver
                or self._size != tuple(size[:2])
                or self._step != tuple(step[:2])
                or self._zero != tuple(zero[:2])
                or self._dtype != dtype):
            self._prepare(driver, weights, size, step, zero, dtype)

        spec_size, spec_step, spec_zero = spectral_axis
        grid = dict(
            spat_size=self._size + (1,),
            spat_step=self._step + (0,),
            spat_zero=self._zero + (0,),
            spat_rota=rota,
            spec_size=spec_size,
            spec_step=spec_step,
            spec_zero=spec_zero)

        wdata = self._wdata
        bdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(self._size[::-1], dtype)
            driver.mem_fill(bdata, 0)

        _detail.evaluate_components(
            self._components, self._mappings,
            driver, params, grid, outputs | dict(wdata=wdata, bdata=bdata),
            dtype, out_extra, '')

        # Weight the data with the spatial weights evaluated above
        if weights is not None:
            self._backend.wcube_evaluate(wdata[None], weights)

        if out_extra is not None:
            if wdata is not None:
                out_extra['total_wdata'] = driver.mem_copy_d2h(wdata)
            out_extra['total_bdata'] = driver.mem_copy_d2h(bdata)


class GModel3D(GModel, abc.ABC):
    """
    A gmodel that adds up components and opacity components on a 3d
    spatial grid: the x and y axes of the data, and a z axis that can be
    configured. The opacity components make an opacity cube, which
    absorbs the brightness of the components.

    A subclass declares the parsers of its components (_cmp_parser) and
    opacity components (_ocmp_parser), and its options (__init__).
    """

    _cmp_parser: parseutils.TypedParser
    _ocmp_parser: parseutils.TypedParser

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        parseutils.load_option_and_update_info(
            cls._cmp_parser, info, 'components',
            required=True, allow_none=False)
        parseutils.load_option_and_update_info(
            cls._ocmp_parser, info, 'opacity_components')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def __init__(
            self, components, opacity_components=None,
            size_z=None, step_z=None, zero_z=None):
        if not components:
            raise RuntimeError("at least one component must be configured")
        self._components = iterutils.tuplify(components, False)
        self._ocomponents = iterutils.tuplify(opacity_components, False)
        self._params, self._mappings = _detail.make_component_params(
            self._components, *_CMP_PREFIX)
        oparams, self._omappings = _detail.make_component_params(
            self._ocomponents, *_OCMP_PREFIX)
        self._params |= oparams
        # The spatial grid. The x and y axes are those of the data, and
        # the z axis is either configured or picked by _prepare().
        self._size_z = size_z
        self._step_z = step_z
        self._zero_z = zero_z
        self._size = None
        self._step = None
        self._zero = None
        self._wdata = None
        self._odata = None
        self._dtype = None
        self._driver = None
        self._backend = None

    def pdescs(self):
        return self._params

    def has_weights(self):
        return any(cmp.has_weights() for cmp in self._components)

    def constants(self):
        return (
            _detail.make_component_constants(
                self._components, *_CMP_PREFIX)
            | _detail.make_component_constants(
                self._ocomponents, *_OCMP_PREFIX))

    def _prepare(self, driver, weights, size, step, zero, dtype):
        self._driver = driver
        self._dtype = dtype
        # A z axis that encloses the whole galaxy is hard to calculate.
        # Instead, if it is not configured, we pick the size and step of
        # the longest of the x and y axes, and place zero in the middle.
        longest = int(size[0] < size[1])
        size_z = self._size_z if self._size_z is not None else size[longest]
        step_z = self._step_z if self._step_z is not None else step[longest]
        zero_z = self._zero_z if self._zero_z is not None \
            else -(size_z / 2 - 0.5) * step_z
        self._size = tuple(size[:2]) + (size_z,)
        self._step = tuple(step[:2]) + (step_z,)
        self._zero = tuple(zero[:2]) + (zero_z,)
        # The spatial weights, if weighting is requested, and the opacity,
        # if there are opacity components
        self._wdata = None
        self._odata = None
        if weights is not None:
            self._wdata = driver.mem_alloc_d(self._size[::-1], dtype)
        if self._ocomponents:
            self._odata = driver.mem_alloc_d(self._size[::-1], dtype)
        self._backend = driver.native_class('GModel', dtype)()

    def _evaluate(
            self, driver, params, outputs, spectral_axis, weights,
            size, step, zero, rota, dtype, out_extra):
        """
        Evaluate the gmodel on the spatial grid of the data (size, step and
        zero) and the given spectral axis (size, step and zero). outputs
        has the array the components add to: the 'image' or the 'scube'.
        """
        if (self._driver is not driver
                or self._size[:2] != tuple(size[:2])
                or self._step[:2] != tuple(step[:2])
                or self._zero[:2] != tuple(zero[:2])
                or self._dtype != dtype):
            self._prepare(driver, weights, size, step, zero, dtype)

        spec_size, spec_step, spec_zero = spectral_axis
        grid = dict(
            spat_size=self._size,
            spat_step=self._step,
            spat_zero=self._zero,
            spat_rota=rota,
            spec_size=spec_size,
            spec_step=spec_step,
            spec_zero=spec_zero)

        wdata = self._wdata
        odata = self._odata
        bdata = None
        obdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(self._size[::-1], dtype)
            driver.mem_fill(bdata, 0)
            obdata = driver.mem_alloc_d(self._size[::-1], dtype)
            driver.mem_fill(obdata, 0)

        # The opacity components add to the opacity cube, so clear it
        if odata is not None:
            driver.mem_fill(odata, 0)
            _detail.evaluate_components(
                self._ocomponents, self._omappings,
                driver, params, grid, dict(odata=odata), dtype,
                out_extra, 'opacity_')

        outputs = outputs | dict(
            wdata=wdata, bdata=bdata, odata=odata, obdata=obdata)
        _detail.evaluate_components(
            self._components, self._mappings,
            driver, params, grid, outputs, dtype, out_extra, '')

        # Weight the data with the spatial weights evaluated above
        if weights is not None:
            self._backend.wcube_evaluate(wdata, weights)

        if out_extra is not None:
            totals = dict(wdata=wdata, odata=odata, bdata=bdata, obdata=obdata)
            for name, total in totals.items():
                if total is not None:
                    out_extra[f'total_{name}'] = driver.mem_copy_d2h(total)
