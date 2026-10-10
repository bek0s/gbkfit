from gbkfit.driver.base import driver_parser


# The drivers, imported when first asked for: each needs its native module,
# and cuda needs CuPy
driver_parser.register_optional({
    'host': 'gbkfit.driver.drivers.host.DriverHost',
    'cuda': 'gbkfit.driver.drivers.cuda.DriverCuda',
})
