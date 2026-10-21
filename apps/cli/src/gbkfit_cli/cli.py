"""
gbkfit-cli: the command-line tool of GBKFIT. It runs the tasks of the
library (see gbkfit.tasks) in-process.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any, Literal

from gbkfit.tasks import log_handler
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError

_log = logging.getLogger(__name__)


def configure_logging(
        log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
) -> None:
    """
    Configure logging with specified log level: GBKFIT and the tool log to
    the console of the tasks (stderr; see gbkfit.tasks.log_handler; the
    tasks also log to their runs).
    """
    handler = log_handler()
    for name in ('gbkfit', 'gbkfit_cli'):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.addHandler(handler)
        logger.propagate = False


# The options of prep pixel_moments and region_moments with one value for
# each moment order
_MOMENT_OPTIONS = (
    'data_d', 'data_e', 'data_m',
    'clip_min', 'clip_max', 'sclip_sigma', 'sclip_iters')


def _check_moment_counts(
        parser: argparse.ArgumentParser,
        args: argparse.Namespace
) -> None:
    """
    Exit with an error unless each option of _MOMENT_OPTIONS given as a
    list has as many values as moment orders. This is checked after the
    parsing: argparse reads the options in the order they come, possibly
    before the orders.
    """
    for dest in _MOMENT_OPTIONS:
        values = getattr(args, dest, None)
        if isinstance(values, list) and len(values) != len(args.orders):
            option = '--' + dest.replace('_', '-')
            parser.error(
                f"argument {option}: invalid length; the number of values "
                f"must match the number of moment orders; expected "
                f"{len(args.orders)}, but got {len(values)}")


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
) -> None:
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
        def __call__(
                self,
                parser: argparse.ArgumentParser,
                namespace: argparse.Namespace,
                values: Any,
                option_string: str | None = None
        ) -> None:
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

    def _number_range_checker(arg: str) -> float:
        try:
            num = type_(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"must be of type {type_.__name__}")
        # (NaN fails both comparisons below)
        if num != num:
            raise argparse.ArgumentTypeError("must be a number")
        if min_ is not None and num < min_:
            raise argparse.ArgumentTypeError(f"must be >= {min_}")
        if max_ is not None and num > max_:
            raise argparse.ArgumentTypeError(f"must be <= {max_}")
        return num

    return _number_range_checker


def _output_parser(
        default_mode: Literal['terminate', 'overwrite', 'unique'],
        default_dir: str | None = 'output',
        default_dir_text: str = 'output'
) -> argparse.ArgumentParser:
    """
    Return the parser of the options of the output directory of a task,
    with its default (and its text in the help), and a default for what
    to do if it exists.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--output-dir', type=str, default=default_dir,
        metavar='OUTPUT',
        help=f"the path of the directory to store the task's output; "
             f"default: {default_dir_text}")
    parser.add_argument(
        '--output-dir-mode', choices=['terminate', 'overwrite', 'unique'],
        default=default_mode,
        help=f"what to do if the output directory exists: 'terminate' "
             f"(fail), 'overwrite' (write into it), or 'unique' (write "
             f"into a new one with a numbered suffix, e.g. output_2); "
             f"default: {default_mode}")
    return parser


def _make_parser() -> argparse.ArgumentParser:
    """Return the parser of the command line."""
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

    # (the results of eval and fit are never overwritten by default)
    parser_common_output = _output_parser('overwrite')
    parser_results_output = _output_parser('unique')

    parser_common_config = argparse.ArgumentParser(add_help=False)
    parser_common_config.add_argument(
        '--strict', action='store_true',
        help="treat unknown configuration options, sections and "
             "parameters as errors instead of warnings")

    #
    # Create parser for eval task
    #

    parser_eval = parsers_task.add_parser('eval', parents=[
        parser_common_config, parser_results_output, parser_common],
        help="evaluate model")
    parser_eval.add_argument(
        'mode', type=str, choices=['model', 'objective', 'mock'],
        help="the evaluation mode: the model, the objective (the "
             "residuals against the data), or mock data of the "
             "observations (the model plus noise; see the section 'mock')")
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
    _CLIP_MIN_HELP = "the values below MIN are set to NaN (invalid)"
    _CLIP_MAX_HELP = "the values above MAX are set to NaN (invalid)"
    _SCLIP_SIGMA_HELP = (
        "the number of standard deviations at which sigma clipping rejects "
        "values")
    _SCLIP_ITERS_HELP = "the most iterations of sigma clipping"
    # ...
    parser_prep_dtype = argparse.ArgumentParser(add_help=False)
    parser_prep_dtype.add_argument(
        '--dtype', type=str, default='float32', choices=['float32', 'float64'],
        help="data type of the output")
    parser_prep_minify = argparse.ArgumentParser(add_help=False)
    parser_prep_minify.add_argument(
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
        metavar='DATA',
        help=_DATA_D_HELP)
    parser_prep_input_n.add_argument(
        '--data-e', type=str, nargs='+',
        metavar='ERRORS',
        help=_DATA_E_HELP)
    parser_prep_input_n.add_argument(
        '--data-m', type=str, nargs='+',
        metavar='MASK',
        help=_DATA_M_HELP)
    # ...
    parser_prep_rest = argparse.ArgumentParser(add_help=False)
    parser_prep_rest.add_argument(
        '--velocity-rest', type=str, metavar='REST',
        help="convert the spectral axis (a linear wavelength or frequency) "
             "to the optical or radio velocities of this rest wavelength or "
             "frequency, e.g. '6562.8 Angstrom' (in air for an AWAV axis)")
    parser_prep_rest.add_argument(
        '--velocity', type=str, choices=['radio', 'optical'],
        help="the convention of the velocities of a spectral axis of the "
             "AIPS convention without VELREF (e.g. VELO-HEL), which is "
             "ambiguous: AIPS reads them as optical, MIRIAD writes radio "
             "ones")

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
        metavar='MIN',
        help=_CLIP_MIN_HELP)
    parser_prep_clip_n.add_argument(
        '--clip-max', type=float, nargs='+',
        metavar='MAX',
        help=_CLIP_MAX_HELP)
    parser_prep_clip_n.add_argument(
        '--sclip-sigma', type=_number_range(float, 0.0, None), nargs='+',
        metavar='SIGMA',
        help=_SCLIP_SIGMA_HELP)
    parser_prep_clip_n.add_argument(
        '--sclip-iters', type=_number_range(int, 1, None), nargs='+',
        default=5,
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
        'prep', help="prepare data for fitting")
    parsers_prep = parser_prep.add_subparsers(
        dest='prep_task', help="the type of data to prepare")
    parsers_prep.required = True
    parsers_prep.add_parser('pixel_brightness', parents=[
        parser_prep_input_1,
        parser_prep_roi_spat_2d,
        parser_prep_clip_1, parser_prep_ccl, parser_prep_nanpad,
        parser_prep_dtype, parser_prep_minify, parser_common_output,
        parser_common],
        help="brightness on a grid of pixels (an image)")
    parsers_prep.add_parser('slit_spectra', parents=[
        parser_prep_input_1, parser_prep_rest,
        parser_prep_roi_spat_1d, parser_prep_roi_spec_1d,
        parser_prep_clip_1, parser_prep_ccl, parser_prep_nanpad,
        parser_prep_dtype, parser_prep_minify, parser_common_output,
        parser_common],
        help="spectra along a long slit")
    parsers_prep.add_parser('pixel_moments', parents=[
        parser_prep_orders,
        parser_prep_input_n,
        parser_prep_roi_spat_2d,
        parser_prep_clip_n, parser_prep_ccl, parser_prep_nanpad,
        parser_prep_dtype, parser_prep_minify, parser_common_output,
        parser_common],
        help="moments on a grid of pixels (moment maps)")
    parser_prep_region_moments = parsers_prep.add_parser(
        'region_moments', parents=[
            parser_prep_orders, parser_prep_input_n,
            parser_prep_dtype, parser_common_output, parser_common],
        help="moments in regions: binned moment maps, whose pixels hold "
             "the value of their bin, to one value per bin")
    parser_prep_region_moments.add_argument(
        '--bins', type=str, required=True, metavar='BINS',
        help="the map of the bin of each pixel (negative for none)")
    parser_prep_region_spectra = parsers_prep.add_parser(
        'region_spectra', parents=[
            parser_prep_input_1, parser_prep_rest, parser_prep_roi_spec_1d,
            parser_prep_dtype, parser_common_output, parser_common],
        help="spectra in regions: the sums of the spectra of the bins of a "
             "cube")
    parser_prep_region_spectra.add_argument(
        '--bins', type=str, required=True, metavar='BINS',
        help="the map of the bin of each pixel of the cube (negative for "
             "none)")
    parser_prep_pixel_spectra = parsers_prep.add_parser(
        'pixel_spectra', parents=[
            parser_prep_input_1, parser_prep_rest,
            parser_prep_roi_spat_2d, parser_prep_roi_spec_1d,
            parser_prep_clip_1, parser_prep_ccl, parser_prep_nanpad,
            parser_prep_dtype, parser_prep_minify, parser_common_output,
            parser_common],
        help="spectra on a grid of pixels (a spectral cube)")
    parser_prep_pixel_spectra.add_argument(
        '--noise-error', action='store_true',
        help="estimate the errors from the noise of each channel, from its "
             "negative values (instead of an errors file; the noise must be "
             "of mean 0, e.g. without continuum)")
    parser_prep_psf_muse = parsers_prep.add_parser(
        'psf_muse', parents=[parser_common_output, parser_common],
        help="the PSF of a MUSE cube (from its header), as a table")
    parser_prep_psf_muse.add_argument('cube', type=str, help="the cube")
    parsers_prep.add_parser(
        'lsf_muse', parents=[parser_common_output, parser_common],
        help="the LSF of MUSE (Bacon et al. 2017), as a table")
    parser_prep_beams = parsers_prep.add_parser(
        'psf_casa_beams', parents=[parser_common_output, parser_common],
        help="the beams of the channels of a CASA cube, as a table")
    parser_prep_beams.add_argument('cube', type=str, help="the cube")

    #
    # Create parser for fit task
    #

    parser_fit = parsers_task.add_parser('fit', parents=[
        parser_common_config, parser_results_output, parser_common],
        help="fit model to data")
    parser_fit.add_argument(
        'config', type=str, nargs='?',
        help="configuration file path; json and yaml formats are supported")
    parser_fit.add_argument(
        '--resume', type=str, metavar='OUTPUT',
        help="resume the fit in this output directory (stopped or killed), "
             "from the configuration and the log of evaluations it has; "
             "instead of a configuration")
    parser_fit.add_argument(
        '--bootstrap', type=str, metavar='OUTPUT',
        help="bootstrap the errors of the fit in this output directory, by "
             "the section 'bootstrap' of the configuration given, and add "
             "them to its result; instead of fitting")
    parser_fit.add_argument(
        '--no-evaluation-log', action='store_true',
        help="do not log the evaluations of the fit (logs/evaluations.bin), "
             "which it resumes from")

    #
    # Create parser for plot task
    #

    parser_plot = parsers_task.add_parser('plot', parents=[
        _output_parser('overwrite', None, 'the folder plots of the run'),
        parser_common],
        help="plot fitting results")
    parser_plot.add_argument(
        'result_dir', type=str,
        metavar='result-dir',
        help="the path of the output directory of a fitting run (or of an "
             "evaluation of the objective)")
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
        help="the elements of the posterior plots (by default, the free "
             "ones)")

    #
    # Parse arguments and run the appropriate task
    #

    return parser


def _run_eval(args: argparse.Namespace) -> None:
    """Run the eval task."""
    import gbkfit.tasks.eval
    with parseutils.strict_mode(args.strict):
        gbkfit.tasks.eval.eval_(
            args.mode, args.config, args.profile,
            args.output_dir, args.output_dir_mode)


def _run_prep(args: argparse.Namespace) -> None:
    """Run the prep task."""
    import gbkfit.tasks.prep
    output = dict(
        output_dir=args.output_dir, output_dir_mode=args.output_dir_mode)
    if args.prep_task == 'pixel_brightness':
        gbkfit.tasks.prep.prep_pixel_brightness(
            args.data_d, args.data_e, args.data_m,
            args.roi_spat, args.clip_min, args.clip_max,
            args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
            args.sclip_sigma, args.sclip_iters,
            args.minify, args.nanpad, args.dtype,
            **output)
    elif args.prep_task == 'slit_spectra':
        gbkfit.tasks.prep.prep_slit_spectra(
            args.data_d, args.data_e, args.data_m,
            args.roi_spat, args.roi_spec, args.clip_min, args.clip_max,
            args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
            args.sclip_sigma, args.sclip_iters,
            args.minify, args.nanpad, args.dtype,
            args.velocity_rest, args.velocity, **output)
    elif args.prep_task == 'pixel_moments':
        gbkfit.tasks.prep.prep_pixel_moments(
            args.data_d, args.data_e, args.data_m,
            args.roi_spat, args.clip_min, args.clip_max,
            args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
            args.sclip_sigma, args.sclip_iters,
            args.minify, args.nanpad, args.dtype,
            **output)
    elif args.prep_task == 'region_moments':
        gbkfit.tasks.prep.prep_region_moments(
            args.bins, args.data_d, args.data_e, args.data_m, args.dtype,
            **output)
    elif args.prep_task == 'region_spectra':
        gbkfit.tasks.prep.prep_region_spectra(
            args.bins, args.data_d, args.data_e, args.data_m,
            args.roi_spec, args.dtype, args.velocity_rest, args.velocity,
            **output)
    elif args.prep_task == 'psf_muse':
        gbkfit.tasks.prep.prep_psf_muse(args.cube, **output)
    elif args.prep_task == 'lsf_muse':
        gbkfit.tasks.prep.prep_lsf_muse(**output)
    elif args.prep_task == 'psf_casa_beams':
        gbkfit.tasks.prep.prep_psf_casa_beams(args.cube, **output)
    elif args.prep_task == 'pixel_spectra':
        gbkfit.tasks.prep.prep_pixel_spectra(
            args.data_d, args.data_e, args.data_m,
            args.roi_spat, args.roi_spec, args.clip_min, args.clip_max,
            args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
            args.sclip_sigma, args.sclip_iters,
            args.minify, args.nanpad, args.dtype,
            args.velocity_rest, args.velocity, **output,
            noise_error=args.noise_error)


def _run_fit(args: argparse.Namespace) -> None:
    """Run the fit task (or resume a fit, or bootstrap one)."""
    import gbkfit.tasks.fit
    if args.resume is not None and args.bootstrap is not None:
        raise ConfigError(
            "give the output directory of a fit to resume (--resume) or to "
            "bootstrap (--bootstrap), not both")
    if args.bootstrap is not None and args.config is None:
        raise ConfigError(
            "a bootstrap (--bootstrap) needs the configuration of its "
            "section 'bootstrap'")
    if args.bootstrap is None and (args.config is None) == (
            args.resume is None):
        raise ConfigError(
            "give a configuration, or the output directory of a fit to "
            "resume (--resume), not both")
    with parseutils.strict_mode(args.strict):
        if args.resume is not None:
            gbkfit.tasks.fit.resume(args.resume)
        elif args.bootstrap is not None:
            gbkfit.tasks.fit.bootstrap(args.bootstrap, args.config)
        else:
            gbkfit.tasks.fit.fit(
                args.config, args.output_dir, args.output_dir_mode,
                not args.no_evaluation_log)


def _run_plot(args: argparse.Namespace) -> None:
    """Run the plot task."""
    import gbkfit.tasks.plot
    gbkfit.tasks.plot.plot(
        args.result_dir, args.output_dir, args.output_dir_mode,
        args.format, args.dpi, args.only_best, args.posterior_mode,
        args.posterior_params)


def main() -> None:
    """
    Run gbkfit-cli: parse the command line, and run its task. Errors in
    configurations are reported without a traceback (with one if
    --verbose), and exit with status 1.
    """
    parser = _make_parser()
    args = parser.parse_args()
    if args.task == 'prep' and args.prep_task in (
            'pixel_moments', 'region_moments'):
        _check_moment_counts(parser, args)

    # Configure log level before doing anything else
    configure_logging('DEBUG' if args.verbose else 'INFO')
    _log.debug(
        f"cli has been called with the following arguments: {vars(args)}")

    tasks = dict(eval=_run_eval, prep=_run_prep, fit=_run_fit, plot=_run_plot)
    try:
        tasks[args.task](args)
    except ConfigError as e:
        if args.verbose:
            raise
        _log.error(f"error: {e}")
        sys.exit(1)

    _log.info("So long and thanks for all the fish!")
