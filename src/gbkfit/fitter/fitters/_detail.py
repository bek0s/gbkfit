


def parse_args(info, required, optional):
    unknown = list(set(info) - set(required + list(optional.keys())))
    missing = list(set(required) - set(info))
    args = {k: v for k, v in info.items() if k in required + list(optional.keys())}
    return args, unknown, missing


def parse_fitter_args(info, required, optional):
    pass


def parse_params_args(info, required, optional):
    pass



def __init__(self, **kwargs):
    required_args = self.required_args()
    optional_args = self.optional_args()
    self._kwargs, unknown, missing = parse_args(
        kwargs, required_args, optional_args)
    if unknown:
        log.warning(
            f'The following fitter options are not recognised: {unknown}')
    if missing:
        raise RuntimeError(
            f'The following fitter options are required: {missing}')

    print('kwargs: ', self._kwargs)