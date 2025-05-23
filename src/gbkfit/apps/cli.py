
import argparse
import logging.config
from collections.abc import Callable, Sequence
from typing import Any, Literal

_log = logging.getLogger(__name__)


def configure_logging(
        log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
) -> None:
    """
    Configure logging with specified log level.
    """
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console': {
                'format': '{message}',
                'style': '{'
            },
            'file': {
                'format':
                    '{asctime}:{processName}({process}):{threadName}({thread}):'
                    '{levelname}:{name}:{funcName}:{lineno}:{message}',
                'style': '{'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'console'
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'file',
                'filename': 'gbkfit.log'
            }
        },
        'loggers': {
            'gbkfit': {
                'level': f'{log_level}',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    })


def _validate_moment_count(
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Sequence[str],
        option_string: str
) -> None:
    """
    Validate that the number of provided values matches the number of
    specified moment orders in the namespace.

    Raises an error if the lengths do not match.
    """
    if namespace.orders and len(namespace.orders) != len(values):
        parser.error(
            f"argument {option_string}: invalid length; the number of values "
            f"must match the number of specified moment orders; expected "
            f"{len(namespace.orders)}, but got {len(values)}")


def _validate_range_1d(
        parser: argparse.ArgumentParser,
        _namespace: argparse.Namespace,
        values: Sequence[float],
        option_string: str
) -> None:
    """
    Validate that the given range consists of exactly two values,
    where the first value (minimum) must be less than the second
    value (maximum).
    """
    min_, max_ = values
    if min_ >= max_:
        parser.error(
            f"argument {option_string}: invalid range; "
            f"[MIN: {min_}, MAX: {max_}]; "
            f"Maximum (MAX) must be greater than Minimum (MIN)")


def _validate_range_2d(
        parser: argparse.ArgumentParser,
        _namespace: argparse.Namespace,
        values: Sequence[float],
        option_string: str
):
    """
    Validate that the given 2D range consists of four values:
    - Left (L) < Right (R)
    - Bottom (B) < Top (T)
    """
    left, right, bottom, top = values
    if left >= right:
        parser.error(
            f"argument {option_string}: invalid range; "
            f"[L: {left}, R: {right}]; "
            f"Right (R) must be greater than Left (L)")
    if bottom >= top:
        parser.error(
            f"argument {option_string}: invalid range; "
            f"[B: {bottom}, T: {top}]; "
            f"Top (T) must be greater than Bottom (B)")


def _create_validator(
        validators: list[Callable[
            [argparse.ArgumentParser, argparse.Namespace, Sequence[Any], str],
            None
        ]]
) -> type[argparse.Action]:
    """
    Creates a custom validator action for argparse that applies a list of
    validation functions to the argument values.
    """
    class Validator(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            for validator in validators:
                validator(parser, namespace, values, option_string)
            setattr(namespace, self.dest, values)
    return Validator


def _number_range(
        type_: type,
        min_: float | None = None,
        max_: float | None = None
) -> Callable[[str], float]:
    """
    Creates a function that validates a number's type and ensures it
    falls within a specified range.
    """
    if min_ is not None:
        assert isinstance(min_, type_)
    if max_ is not None:
        assert isinstance(max_, type_)

    def _number_range_checker(arg):
        try:
            num = type_(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"must be of type {type_.__name__}")
        if min_ is not None and num < min_:
            raise argparse.ArgumentTypeError(f"must be >= {min_}")
        if max_ is not None and num > max_:
            raise argparse.ArgumentTypeError(f"must be <= {max_}")
        return num

    return _number_range_checker


def main():

    from gbkfit import __version__ as version

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version=version)
    parsers_task = parser.add_subparsers(dest='task')
    parsers_task.required = True

    #
    # Define common arguments across all tasks
    #

    parser_common = argparse.ArgumentParser(add_help=False)
    parser_common.add_argument(
        '--verbose', action='store_true', help="enable verbose logging")

    parser_common_output = argparse.ArgumentParser(add_help=False)
    parser_common_output.add_argument(
        '--output-dir', type=str, default='output',
        metavar='OUTPUT',
        help="the path of the directory to store the task's output")
    parser_common_output.add_argument(
        '--output-dir-mode', choices=['terminate', 'overwrite', 'unique'],
        default='overwrite',
        help="Specifies how to handle existing output directories: "
             "'terminate' (fail if the directory exists), "
             "'overwrite' (overwrite existing contents), or "
             "'unique' (append a number suffix to ensure uniqueness)")

    #
    # Create parser for eval task
    #

    parser_eval = parsers_task.add_parser('eval', parents=[
        parser_common_output, parser_common],
        help="evaluate model")
    parser_eval.add_argument(
        'mode', type=str, choices=['model', 'objective'],
        help="the evaluation mode")
    parser_eval.add_argument(
        'config', type=str,
        help="configuration file path; json and yaml formats are supported")
    parser_eval.add_argument(
        '--profile', type=_number_range(int, 0, None), default=0,
        metavar='ITERS',
        help="perform the evaluation ITERS times "
             "and provide performance evaluation statistics")

    #
    # Create parser for prep task
    #

    _DATA_D_HELP = "input data (measurements)"
    _DATA_E_HELP = "input data (uncertainties)"
    _DATA_M_HELP = "input data (mask)"
    _CLIP_MIN_HELP = """
            minimum clip threshold value; 
            all values below MIN will be set to nan
            """
    _CLIP_MAX_HELP = """
            maximum clip threshold value; 
            all values above MAX will be set to nan
            """
    _SCLIP_SIGMA_HELP = """
            the number of standard deviations to use for sigma-clipping
            """
    _SCLIP_ITERS_HELP = """
            the maximum number of sigma-clipping iterations to perform
            """
    # ...
    parser_prep_common = argparse.ArgumentParser(add_help=False)
    parser_prep_common.add_argument(
        '--dtype', type=str, default='float32', choices=['float32', 'float64'],
        help="data type of the output")
    parser_prep_common.add_argument(
        '--minify', action='store_true',
        help="crop the edges of the input data until valid pixels are found")
    # ...
    parser_prep_nanpad = argparse.ArgumentParser(add_help=False)
    parser_prep_nanpad.add_argument(
        '--nanpad', type=_number_range(int, 0, None),
        metavar='SIZE',
        help="nan-pad the resulting data by SIZE along all dimensions")
    # ...
    parser_prep_orders = argparse.ArgumentParser(add_help=False)
    parser_prep_orders.add_argument(
        'orders', type=_number_range(int, 0, 7), nargs='+',
        help="the orders of the provided moment map data")
    # ...
    parser_prep_input_1 = argparse.ArgumentParser(add_help=False)
    parser_prep_input_1.add_argument(
        '--data-d', type=str, required=True,
        metavar='DATA',
        help=_DATA_D_HELP)
    parser_prep_input_1.add_argument(
        '--data-e', type=str,
        metavar='ERRORS',
        help=_DATA_E_HELP)
    parser_prep_input_1.add_argument(
        '--data-m', type=str,
        metavar='MASK',
        help=_DATA_M_HELP)
    # ...
    parser_prep_input_n = argparse.ArgumentParser(add_help=False)
    parser_prep_input_n.add_argument(
        '--data-d', type=str, nargs='+', required=True,
        action=_create_validator([_validate_moment_count]),
        metavar='DATA',
        help=_DATA_D_HELP)
    parser_prep_input_n.add_argument(
        '--data-e', type=str, nargs='+',
        action=_create_validator([_validate_moment_count]),
        metavar='ERRORS',
        help=_DATA_E_HELP)
    parser_prep_input_n.add_argument(
        '--data-m', type=str, nargs='+',
        action=_create_validator([_validate_moment_count]),
        metavar='MASK',
        help=_DATA_M_HELP)
    # ...
    parser_prep_roi_spat_1d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spat_1d.add_argument(
        '--roi-spat', type=_number_range(int, 0, None), nargs=2,
        action=_create_validator([_validate_range_1d]),
        metavar=('MIN', 'MAX'),
        help="crop input data around a region of interest (spatial)")
    # ...
    parser_prep_roi_spat_2d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spat_2d.add_argument(
        '--roi-spat', type=_number_range(int, 0, None), nargs=4,
        action=_create_validator([_validate_range_2d]),
        metavar=('L', 'R', 'B', 'T'),
        help="crop input data around a region of interest (spatial); "
             "L: Left, R: Right, B: Bottom, T: Top")
    # ...
    parser_prep_roi_spec_1d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spec_1d.add_argument(
        '--roi-spec', type=_number_range(int, 0, None), nargs=2,
        action=_create_validator([_validate_range_1d]),
        metavar=('MIN', 'MAX'),
        help="crop input data around a region of interest (spectral)")
    # ...
    parser_prep_clip_1 = argparse.ArgumentParser(add_help=False)
    parser_prep_clip_1.add_argument(
        '--clip-min', type=float, nargs=1,
        metavar='MIN',
        help=_CLIP_MIN_HELP)
    parser_prep_clip_1.add_argument(
        '--clip-max', type=float, nargs=1,
        metavar='MAX',
        help=_CLIP_MAX_HELP)
    parser_prep_clip_1.add_argument(
        '--sclip-sigma', type=_number_range(float, 0.0, None),
        metavar='SIGMA',
        help=_SCLIP_SIGMA_HELP)
    parser_prep_clip_1.add_argument(
        '--sclip-iters', type=_number_range(int, 1, None), default=5,
        metavar='ITERS',
        help=_SCLIP_ITERS_HELP)
    # ...
    parser_prep_clip_n = argparse.ArgumentParser(add_help=False)
    parser_prep_clip_n.add_argument(
        '--clip-min', type=float, nargs='+',
        action=_create_validator([_validate_moment_count]),
        metavar='MIN',
        help=_CLIP_MIN_HELP)
    parser_prep_clip_n.add_argument(
        '--clip-max', type=float, nargs='+',
        action=_create_validator([_validate_moment_count]),
        metavar='MAX',
        help=_CLIP_MAX_HELP)
    parser_prep_clip_n.add_argument(
        '--sclip-sigma', type=_number_range(float, 0.0, None), nargs='+',
        action=_create_validator([_validate_moment_count]),
        metavar='SIGMA',
        help=_SCLIP_SIGMA_HELP)
    parser_prep_clip_n.add_argument(
        '--sclip-iters', type=_number_range(int, 1, None), nargs='+', default=5,
        action=_create_validator([_validate_moment_count]),
        metavar='ITERS',
        help=_SCLIP_ITERS_HELP)
    # ...
    parser_prep_ccl = argparse.ArgumentParser(add_help=False)
    parser_prep_ccl.add_argument(
        '--ccl-lcount', type=_number_range(int, 1, None),
        metavar='COUNT',
        help="connected component labeling; maximum number of labels")
    parser_prep_ccl.add_argument(
        '--ccl-pcount', type=_number_range(int, 1, None),
        metavar='COUNT',
        help="connected component labeling; minimum area per label")
    parser_prep_ccl.add_argument(
        '--ccl-lratio', type=_number_range(float, 0.0, 1.0),
        metavar='RATIO',
        help="connected component labeling; minimum area ratio per label; "
             "RATIO = (label area) / (largest label area)")
    # ...
    parser_prep = parsers_task.add_parser(
        'prep', parents=[parser_common], help="prepare data for fitting")
    parsers_prep = parser_prep.add_subparsers(
        dest='prep_task', help="the type of data to prepare")
    parsers_prep.required = True
    parsers_prep.add_parser('image', parents=[
        parser_prep_input_1,
        parser_prep_roi_spat_2d,
        parser_prep_clip_1, parser_prep_ccl, parser_prep_nanpad,
        parser_prep_common, parser_common_output, parser_common],
        help="image")
    parsers_prep.add_parser('lslit', parents=[
        parser_prep_input_1,
        parser_prep_roi_spat_1d, parser_prep_roi_spec_1d,
        parser_prep_clip_1, parser_prep_ccl, parser_prep_nanpad,
        parser_prep_common, parser_common_output, parser_common],
        help="long slit")
    parsers_prep.add_parser('mmaps', parents=[
        parser_prep_orders,
        parser_prep_input_n,
        parser_prep_roi_spat_2d,
        parser_prep_clip_n, parser_prep_ccl, parser_prep_nanpad,
        parser_prep_common, parser_common_output, parser_common],
        help="moment maps")
    parsers_prep.add_parser('scube', parents=[
        parser_prep_input_1,
        parser_prep_roi_spat_2d, parser_prep_roi_spec_1d,
        parser_prep_clip_1, parser_prep_ccl, parser_prep_nanpad,
        parser_prep_common, parser_common_output, parser_common],
        help="spectral cube")

    #
    # Create parser for fit task
    #

    parser_fit = parsers_task.add_parser('fit', parents=[
        parser_common_output, parser_common],
        help="fit model to data")
    parser_fit.add_argument(
        'config', type=str,
        help="configuration file path; json and yaml formats are supported")

    #
    # Create parser for plot task
    #

    parser_plot = parsers_task.add_parser('plot', parents=[
        parser_common_output, parser_common],
        help="plot fitting results")
    parser_plot.add_argument(
        'result_dir', type=str,
        metavar='result-dir',
        help="the path of the output directory of a fitting run")
    parser_plot.add_argument(
        '--format', type=str, default='pdf', choices=['pdf', 'png'],
        help="the format of the created figures")
    parser_plot.add_argument(
        '--dpi', type=_number_range(int, 100, None), default=150,
        metavar='DPI',
        help="the dpi of the created figures")
    parser_plot.add_argument(
        '--only-best', action='store_true',
        help="only plot results of the best solution; "
             "only useful for results containing multiple solutions")
    parser_plot.add_argument(
        '--posterior-mode', type=str,
        default='none', choices=['none', 'corner', 'separate'],
        metavar='MODE',
        help="select posterior plotting mode; "
             "MODE=none: "
             "do not create posterior plots, "
             "MODE=corner: "
             "create a single corner plot for all selected parameters, "
             "MODE=separate: "
             "create a separate corner plot for each parameter pair")
    parser_plot.add_argument(
        '--posterior-params', type=str, nargs='+',
        metavar='PARAM',
        help="plot posterior only for the model parameters in the PARAMS list; "
             "if not defined, all model parameters will be used")

    #
    # Parse arguments and run the appropriate task
    #

    args = parser.parse_args()

    # Configure log level before doing anything else
    configure_logging('DEBUG' if args.verbose else 'INFO')

    _log.debug(
        f"cli has been called with the following arguments: {vars(args)}")

    if args.task == 'eval':
        import gbkfit.tasks.eval
        gbkfit.tasks.eval.eval_(
            args.mode, args.config, args.profile,
            args.output_dir, args.output_dir_mode)

    elif args.task == 'prep':
        import gbkfit.tasks.prep
        if args.prep_task == 'image':
            gbkfit.tasks.prep.prep_image(
                args.data_d, args.data_e, args.data_m,
                args.roi_spat, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters,
                args.minify, args.nanpad, args.dtype)
        elif args.prep_task == 'lslit':
            gbkfit.tasks.prep.prep_lslit(
                args.data_d, args.data_e, args.data_m,
                args.roi_spat, args.roi_spec, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters,
                args.minify, args.nanpad, args.dtype)
        elif args.prep_task == 'mmaps':
            gbkfit.tasks.prep.prep_mmaps(
                args.orders, args.data_d, args.data_e, args.data_m,
                args.roi_spat, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters,
                args.minify, args.nanpad, args.dtype)
        elif args.prep_task == 'scube':
            gbkfit.tasks.prep.prep_scube(
                args.data_d, args.data_e, args.data_m,
                args.roi_spat, args.roi_spec, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters,
                args.minify, args.nanpad, args.dtype)

    elif args.task == 'fit':
        import gbkfit.tasks.fit
        gbkfit.tasks.fit.fit(
            args.config, args.output_dir, args.output_dir_mode)

    elif args.task == 'plot':
        import gbkfit.tasks.plot
        gbkfit.tasks.plot.plot(
            args.result_dir, args.output_dir, args.format, args.dpi,
            args.only_best, args.posterior_mode, args.posterior_params)

    _log.info("So long and thanks for all the fish!")


if __name__ == '__main__':
    main()
