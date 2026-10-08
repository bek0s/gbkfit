
from gbkfit.utils import iterutils, parseutils
from . import _detail


__all__ = [
    'ComponentGModel'
]


class ComponentGModel:
    """
    The base of the gmodels that add up components on a 3d spatial grid
    (and a spectral axis, for spectral cubes).

    A subclass declares the parser of its components (_cmp_parser). A 3d
    gmodel also declares the parser of its opacity components
    (_ocmp_parser), and its z axis can be configured; the z axis of a 2d
    gmodel has size 1. The __init__ of a subclass declares its options.
    """

    _cmp_parser: parseutils.TypedParser
    _ocmp_parser: parseutils.TypedParser | None = None

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        parseutils.load_option_and_update_info(
            cls._cmp_parser, info, 'components',
            required=True, allow_none=False)
        if cls._is_3d():
            parseutils.load_option_and_update_info(
                cls._ocmp_parser, info, 'opacity_components')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        info = dict(type=self.type())
        if self._is_3d():
            info.update(
                size_z=self._size[2],
                step_z=self._step[2],
                zero_z=self._zero[2])
        info.update(components=self._cmp_parser.dump(self._components))
        if self._is_3d():
            info.update(opacity_components=self._ocmp_parser.dump(
                self._ocomponents))
        return info

    def __init__(
            self, components, opacity_components=None,
            size_z=None, step_z=None, zero_z=None):
        if not components:
            raise RuntimeError("at least one component must be configured")
        if not self._is_3d():
            size_z, step_z, zero_z = 1, 0, 0
        self._components = iterutils.tuplify(components, False)
        self._ocomponents = iterutils.tuplify(opacity_components, False)
        # The spatial grid. The x and y axes are those of the data, and
        # the z axis is either configured or picked by _prepare().
        self._size = [None, None, size_z]
        self._step = [None, None, step_z]
        self._zero = [None, None, zero_z]
        self._wdata = None
        self._odata = None
        self._dtype = None
        self._driver = None
        self._backend = None
        (self._params,
         self._mappings,
         self._omappings) = _detail.make_gmodel_params(
            self._components, self._ocomponents)

    def pdescs(self):
        return self._params

    def has_weights(self):
        return any(cmp.has_weights() for cmp in self._components)

    @classmethod
    def _is_3d(cls):
        return cls._ocmp_parser is not None

    def _spatial_shape(self):
        """The shape of the spatial arrays: (z, y, x), or (y, x) if 2d."""
        shape = self._size[::-1]
        return tuple(shape if self._is_3d() else shape[1:])

    def _prepare(self, driver, weights, size, step, zero, dtype):
        self._driver = driver
        self._size[:2] = size[:2]
        self._step[:2] = step[:2]
        self._zero[:2] = zero[:2]
        self._dtype = dtype
        # A z axis that encloses the whole galaxy is hard to calculate.
        # Instead, if it is not configured, we pick the size and step of
        # the longest of the x and y axes, and place zero in the middle.
        longest = int(size[0] < size[1])
        if self._size[2] is None:
            self._size[2] = size[longest]
        if self._step[2] is None:
            self._step[2] = step[longest]
        if self._zero[2] is None:
            self._zero[2] = -(self._size[2] / 2 - 0.5) * self._step[2]
        # The spatial weights, if weighting is requested, and the opacity,
        # if there are opacity components
        self._wdata = None
        self._odata = None
        if weights is not None:
            self._wdata = driver.mem_alloc_d(self._spatial_shape(), dtype)
        if self._ocomponents:
            self._odata = driver.mem_alloc_d(self._spatial_shape(), dtype)
        self._backend = driver.native_class('GModel', dtype)()

    def _evaluate(
            self, driver, params, data, weights, size, step, zero, rota,
            dtype, out_extra):
        """
        Evaluate the gmodel on the grid of the given data: an image or a
        spectral cube ({'image': ...} or {'scube': ...}), with weights or
        not (None).
        """
        if (self._driver is not driver
                or tuple(self._size[:2]) != tuple(size[:2])
                or tuple(self._step[:2]) != tuple(step[:2])
                or tuple(self._zero[:2]) != tuple(zero[:2])
                or self._dtype != dtype):
            self._prepare(driver, weights, size, step, zero, dtype)

        spectral = 'scube' in data
        grid = dict(
            spat_size=tuple(self._size),
            spat_step=tuple(self._step),
            spat_zero=tuple(self._zero),
            spat_rota=rota,
            spec_size=size[2] if spectral else 1,
            spec_step=step[2] if spectral else 0,
            spec_zero=zero[2] if spectral else 0)

        wdata = self._wdata
        odata = self._odata
        bdata = None
        obdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(self._spatial_shape(), dtype)
            driver.mem_fill(bdata, 0)
            if self._is_3d():
                obdata = driver.mem_alloc_d(self._spatial_shape(), dtype)
                driver.mem_fill(obdata, 0)

        # The opacity components add to the opacity cube, so clear it
        if odata is not None:
            driver.mem_fill(odata, 0)
            _detail.evaluate_components(
                self._ocomponents, self._omappings,
                driver, params, grid, dict(odata=odata), dtype,
                out_extra, 'opacity_')

        _detail.evaluate_components(
            self._components, self._mappings,
            driver, params, grid,
            data | dict(wdata=wdata, bdata=bdata, odata=odata, obdata=obdata),
            dtype, out_extra, '')

        # Weight the data with the spatial weights evaluated above
        if weights is not None:
            self._backend.wcube_evaluate(
                wdata.reshape(self._size[::-1]), weights)

        if out_extra is not None:
            totals = dict(wdata=wdata, odata=odata, bdata=bdata, obdata=obdata)
            for name, total in totals.items():
                if total is not None:
                    out_extra[f'total_{name}'] = driver.mem_copy_d2h(total)
