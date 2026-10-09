from gbkfit.model.base import GModelPlan
from gbkfit.utils import gridutils, iterutils, miscutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail


__all__ = [
    'IMAGE_SPECTRAL_AXIS',
    'ComponentSet2D',
    'ComponentSet3D',
    'ComponentSetPlan2D',
    'ComponentSetPlan3D',
    'ComponentSetGModelPlan'
]


# The spectral axis of an image: one channel, without a width
IMAGE_SPECTRAL_AXIS = gridutils.Grid(
    (1,), gridutils.Coords((0,), (0,), (0,), 0), 0)

# The components and the opacity components: their label in messages, the
# prefix of their parameters, and whether the parameters of the first one
# have it (see _detail.component_prefixes)
_CMP_PREFIX = ('components', 'cmp', False)
_OCMP_PREFIX = ('opacity components', 'ocmp', True)


def _with_mass_model_params(params, components, mass_model):
    """
    The parameters of the components of a gmodel, and those of its mass
    model (if any). Raise ConfigError if components take the circular
    velocity of a mass model (see Component.circular_velocity_params) and
    there is none, if there is one and none does, or if the names of the
    parameters repeat.
    """
    moved = any(cmp.circular_velocity_params() for cmp in components)
    if moved and mass_model is None:
        raise ConfigError(
            "the 'mass' velocity traits need the mass_model of their gmodel")
    if mass_model is None:
        return params
    if not moved:
        raise ConfigError(
            "no component uses the mass_model of the gmodel; give the "
            "components that it moves 'mass' velocity traits")
    if repeated := sorted(params.keys() & mass_model.pdescs().keys()):
        raise ConfigError(
            f"the parameters of the mass model have the names of parameters "
            f"of the components: {repeated}; rename the components")
    return params | mass_model.pdescs()


def _circular_velocities(components, mass_model, params):
    """
    The values of the parameters of each component that are the circular
    velocity of the mass model (see Component.circular_velocity_params),
    from the parameters of the mass model in params.
    """
    return [
        {name: mass_model.vcirc(radii, params)
         for name, radii in cmp.circular_velocity_params().items()}
        for cmp in components]


class ComponentSet2D:
    """
    The components of a 2d gmodel: their parameters, and their evaluation
    on a 2d spatial grid, the x and y axes of the data. The components get
    a 3d grid with a z axis of size 1.
    """

    def __init__(self, components, mass_model=None):
        if not components:
            raise RuntimeError("at least one component must be configured")
        self._components = iterutils.tuplify(components, False)
        self._prefixes = _detail.component_prefixes(
            self._components, *_CMP_PREFIX)
        params, self._mappings = miscutils.merge_with_prefixes(
            [cmp.pdescs() for cmp in self._components], self._prefixes)
        self._params = _with_mass_model_params(
            params, self._components, mass_model)
        self._mass_model = mass_model

    def components(self):
        return self._components

    def mass_model(self):
        return self._mass_model

    def mappings(self):
        """The parameter names of each component, by its own names."""
        return self._mappings

    def pdescs(self):
        return self._params

    def has_weights(self):
        return any(cmp.has_weights() for cmp in self._components)

    def constants(self):
        constants, _ = miscutils.merge_with_prefixes(
            [cmp.constants() for cmp in self._components], self._prefixes)
        return constants

    def plan(self, driver, grid, spectral, has_weights, dtype, selection):
        """
        The evaluation of the components and lines of the selection (see
        Selection) on the given driver, grid of the x and y axes and
        spectral axis (gridutils.Grid, the second of one axis) and dtype,
        with spatial weights if has_weights.
        """
        return ComponentSetPlan2D(
            self, driver, grid, spectral, has_weights, dtype, selection)


class ComponentSetPlan2D:
    """
    The evaluation of a ComponentSet2D: the plans of its components and
    the spatial weights, on a grid.
    """

    def __init__(
            self, component_set, driver, grid, spectral, has_weights,
            dtype, selection):
        self._component_set = component_set
        self._driver = driver
        self._grid = grid
        self._dtype = dtype
        size = tuple(grid.size[:2])
        spec_size = spectral.size[0]
        spec_step = spectral.coords.step[0]
        spec_zero = spectral.zero()[0]
        self._native_grid = dict(
            spat_size=size + (1,),
            spat_step=tuple(grid.coords.step[:2]) + (0,),
            spat_zero=tuple(grid.zero()[:2]) + (0,),
            spat_rota=grid.coords.rota,
            spec_size=spec_size,
            spec_step=spec_step,
            spec_zero=spec_zero)
        # The selected components, their plans and their parameters
        selected = _detail.select_components(
            component_set.components(), selection.components)
        self._components = tuple(
            component_set.components()[i] for i in selected)
        self._mappings = tuple(component_set.mappings()[i] for i in selected)
        _detail.check_selected_lines(self._components, selection.lines)
        self._component_plans = tuple(
            cmp.plan(driver, spectral, dtype, selection.lines)
            for cmp in self._components)
        # The spatial weights, if weighting is requested
        self._wdata = None
        if has_weights:
            self._wdata = driver.mem_alloc_d(size[::-1], dtype)
        self._backend = driver.native_class('GModel', dtype)()

    def evaluate(self, params, outputs, weights, out_extra):
        """
        Add the components to the output array, the 'image' or the
        'scube' in outputs, and weight the data weights (if any) with the
        spatial weights.
        """
        driver = self._driver
        grid = self._grid
        dtype = self._dtype
        size = tuple(grid.size[:2])

        wdata = self._wdata
        bdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(size[::-1], dtype)
            driver.mem_fill(bdata, 0)

        # The extra outputs are images on the grid; those of the
        # components are on a grid one voxel thick
        def image(data):
            return gridutils.GridData(data, grid.coords, None)

        _detail.evaluate_components(
            self._components, self._component_plans, self._mappings,
            params, self._native_grid,
            outputs | dict(wdata=wdata, bdata=bdata),
            out_extra, '', lambda data: image(data[0]),
            _circular_velocities(
                self._components, self._component_set.mass_model(), params))

        # Weight the data with the spatial weights evaluated above
        if weights is not None:
            self._backend.wcube_evaluate(wdata[None], weights)

        if out_extra is not None:
            if wdata is not None:
                out_extra['total_wdata'] = image(driver.mem_copy_d2h(wdata))
            out_extra['total_bdata'] = image(driver.mem_copy_d2h(bdata))


class ComponentSet3D:
    """
    The components and opacity components of a 3d gmodel: their
    parameters, and their evaluation on a 3d spatial grid, the x and y
    axes of the data and a z axis that can be configured. The opacity
    components make an opacity cube, which absorbs the brightness of the
    components.
    """

    def __init__(
            self, components, opacity_components=None,
            size_z=None, step_z=None, zero_z=None, mass_model=None):
        if not components:
            raise RuntimeError("at least one component must be configured")
        self._components = iterutils.tuplify(components, False)
        self._ocomponents = iterutils.tuplify(opacity_components, False)
        # The components and the opacity components share their names
        repeated = sorted(
            {cmp.name() for cmp in self._components}
            & {cmp.name() for cmp in self._ocomponents} - {None})
        if repeated:
            raise ConfigError(
                f"the components and the opacity components must have "
                f"different names; repeated: {repeated}")
        self._prefixes = _detail.component_prefixes(
            self._components, *_CMP_PREFIX)
        self._oprefixes = _detail.component_prefixes(
            self._ocomponents, *_OCMP_PREFIX)
        params, mappings = miscutils.merge_with_prefixes(
            [cmp.pdescs() for cmp in self._all_components()],
            self._prefixes + self._oprefixes)
        self._mappings = mappings[:len(self._components)]
        self._omappings = mappings[len(self._components):]
        self._params = _with_mass_model_params(
            params, self._components, mass_model)
        self._mass_model = mass_model
        # The z axis, as configured (or None to pick it per grid)
        self._size_z = size_z
        self._step_z = step_z
        self._zero_z = zero_z

    def components(self):
        return self._components

    def opacity_components(self):
        return self._ocomponents

    def mass_model(self):
        return self._mass_model

    def size_z(self):
        return self._size_z

    def step_z(self):
        return self._step_z

    def zero_z(self):
        return self._zero_z

    def pdescs(self):
        return self._params

    def has_weights(self):
        return any(cmp.has_weights() for cmp in self._components)

    def constants(self):
        constants, _ = miscutils.merge_with_prefixes(
            [cmp.constants() for cmp in self._all_components()],
            self._prefixes + self._oprefixes)
        return constants

    def mappings(self):
        """The parameter names of each component, by its own names."""
        return self._mappings

    def omappings(self):
        """The parameter names of each opacity component."""
        return self._omappings

    def _all_components(self):
        return self._components + self._ocomponents

    def plan(self, driver, grid, spectral, has_weights, dtype, selection):
        """
        The evaluation of the components and lines of the selection (see
        Selection) on the given driver, grid of the x and y axes and
        spectral axis (gridutils.Grid, the second of one axis) and dtype,
        with spatial weights if has_weights.
        """
        return ComponentSetPlan3D(
            self, driver, grid, spectral, has_weights, dtype, selection)


class ComponentSetPlan3D:
    """
    The evaluation of a ComponentSet3D: the plans of its components and
    opacity components, the z axis, the spatial weights and the opacity
    cube, on a grid.
    """

    def __init__(
            self, component_set, driver, grid, spectral, has_weights,
            dtype, selection):
        self._component_set = component_set
        self._driver = driver
        self._grid = grid
        self._dtype = dtype
        # A z axis that encloses the whole galaxy is hard to calculate.
        # Instead, if it is not configured, we pick the size and step of
        # the longest of the x and y axes, and place zero in the middle.
        size = grid.size
        step = grid.coords.step
        zero = grid.zero()
        longest = int(size[0] < size[1])
        size_z = component_set.size_z()
        step_z = component_set.step_z()
        zero_z = component_set.zero_z()
        size_z = size_z if size_z is not None else size[longest]
        step_z = step_z if step_z is not None else step[longest]
        zero_z = zero_z if zero_z is not None \
            else -(size_z / 2 - 0.5) * step_z
        self._size = tuple(size[:2]) + (size_z,)
        self._step = tuple(step[:2]) + (step_z,)
        self._zero = tuple(zero[:2]) + (zero_z,)
        spec_size = spectral.size[0]
        spec_step = spectral.coords.step[0]
        spec_zero = spectral.zero()[0]
        self._native_grid = dict(
            spat_size=self._size,
            spat_step=self._step,
            spat_zero=self._zero,
            spat_rota=grid.coords.rota,
            spec_size=spec_size,
            spec_step=spec_step,
            spec_zero=spec_zero)
        # The extra outputs are cubes on the grid: the x and y axes of the
        # data, and the z axis along the line of sight (from z = 0)
        self._coords = gridutils.Coords(
            self._step,
            grid.coords.rpix + (-self._zero[2] / self._step[2],),
            grid.coords.rval + (0.0,), grid.coords.rota)
        # The selected components, their plans and their parameters
        selected = _detail.select_components(
            component_set.components(), selection.components)
        self._components = tuple(
            component_set.components()[i] for i in selected)
        self._mappings = tuple(component_set.mappings()[i] for i in selected)
        _detail.check_selected_lines(self._components, selection.lines)
        self._component_plans = tuple(
            cmp.plan(driver, spectral, dtype, selection.lines)
            for cmp in self._components)
        self._ocomponent_plans = tuple(
            cmp.plan(driver, spectral, dtype, None)
            for cmp in component_set.opacity_components())
        # The spatial weights, if weighting is requested, and the opacity,
        # if there are opacity components
        self._wdata = None
        self._odata = None
        if has_weights:
            self._wdata = driver.mem_alloc_d(self._size[::-1], dtype)
        if component_set.opacity_components():
            self._odata = driver.mem_alloc_d(self._size[::-1], dtype)
        self._backend = driver.native_class('GModel', dtype)()

    def evaluate(self, params, outputs, weights, out_extra):
        """
        Add the components to the output array, the 'image' or the
        'scube' in outputs, and weight the data weights (if any) with the
        spatial weights.
        """
        driver = self._driver
        dtype = self._dtype
        component_set = self._component_set

        def cube(data):
            return gridutils.GridData(data, self._coords, None)

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
                component_set.opacity_components(), self._ocomponent_plans,
                component_set.omappings(), params, self._native_grid,
                dict(odata=odata), out_extra, 'opacity_', cube)

        outputs = outputs | dict(
            wdata=wdata, bdata=bdata, odata=odata, obdata=obdata)
        _detail.evaluate_components(
            self._components, self._component_plans, self._mappings,
            params, self._native_grid, outputs, out_extra, '', cube,
            _circular_velocities(
                self._components, component_set.mass_model(), params))

        # Weight the data with the spatial weights evaluated above
        if weights is not None:
            self._backend.wcube_evaluate(wdata, weights)

        if out_extra is not None:
            totals = dict(wdata=wdata, odata=odata, bdata=bdata, obdata=obdata)
            for name, total in totals.items():
                if total is not None:
                    out_extra[f'total_{name}'] = cube(
                        driver.mem_copy_d2h(total))


class ComponentSetGModelPlan(GModelPlan):
    """
    The evaluation of a gmodel made of a component set: the plan of the
    set, which adds to the output of the given key ('image' or 'scube').
    """

    def __init__(self, component_set_plan, key):
        self._component_set_plan = component_set_plan
        self._key = key

    def evaluate(self, params, data, weights, out_extra):
        self._component_set_plan.evaluate(
            params, {self._key: data}, weights, out_extra)
