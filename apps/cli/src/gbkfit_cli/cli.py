"""
gbkfit-cli: the command-line tool of GBKFIT. It runs the tasks of the
library (see gbkfit.tasks) in-process.
"""

import argparse
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from gbkfit.tasks import exit_on_user_errors, log_to_console, run

_log = logging.getLogger(__name__)


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


def _data_file(text: str) -> str | tuple[str, int | str]:
    """
    Parse a file of data: a filename, or a filename and, in brackets, the
    HDU of a FITS file (its index or its name) or the path of the array
    in an ASDF file, as astropy writes them (e.g. 'muse.fits[STAT]',
    'cube.fits[1]').
    """
    match = re.fullmatch(r'(.+)\[([^\[\]]+)\]', text)
    if match is None:
        return text
    filename, key = match.groups()
    return filename, int(key) if key.isdigit() else key


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

    # (the results of eval and fit are never overwritten by default; they
    # go to the runs of their setup in a project, see gbkfit.project)
    parser_common_output = _output_parser('overwrite')
    parser_results_output = _output_parser(
        'unique', None,
        'runs/<name of the setup> for a setup of a project, else output')

    parser_common_config = argparse.ArgumentParser(add_help=False)
    parser_common_config.add_argument(
        '--lenient', action='store_true',
        help="ignore unknown configuration options, sections and "
             "parameters with a warning, instead of failing")

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

    _DATA_D_HELP = (
        "input data (measurements); the HDU of a FITS file in brackets, "
        "e.g. cube.fits[DATA]")
    _DATA_E_HELP = (
        "input data (uncertainties); the HDU in brackets, e.g. "
        "cube.fits[STAT]")
    _DATA_M_HELP = "input data (mask); the HDU in brackets, e.g. cube.fits[DQ]"
    _ERROR_KIND_HELP = (
        "what the uncertainties are: standard deviations (sigma, the "
        "default), variances or inverse variances; the prepared ones are "
        "standard deviations")
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
        '--data-d', type=_data_file, required=True,
        metavar='DATA',
        help=_DATA_D_HELP)
    parser_prep_input_1.add_argument(
        '--data-e', type=_data_file,
        metavar='ERRORS',
        help=_DATA_E_HELP)
    parser_prep_input_1.add_argument(
        '--data-m', type=_data_file,
        metavar='MASK',
        help=_DATA_M_HELP)
    parser_prep_input_1.add_argument(
        '--error-kind', type=str, default='sigma',
        choices=['sigma', 'variance', 'inverse_variance'],
        help=_ERROR_KIND_HELP)
    # ...
    parser_prep_input_n = argparse.ArgumentParser(add_help=False)
    parser_prep_input_n.add_argument(
        '--data-d', type=_data_file, nargs='+', required=True,
        metavar='DATA',
        help=_DATA_D_HELP)
    parser_prep_input_n.add_argument(
        '--data-e', type=_data_file, nargs='+',
        metavar='ERRORS',
        help=_DATA_E_HELP)
    parser_prep_input_n.add_argument(
        '--data-m', type=_data_file, nargs='+',
        metavar='MASK',
        help=_DATA_M_HELP)
    parser_prep_input_n.add_argument(
        '--error-kind', type=str, default='sigma',
        choices=['sigma', 'variance', 'inverse_variance'],
        help=_ERROR_KIND_HELP)
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
        '--bins', type=_data_file, required=True, metavar='BINS',
        help="the map of the bin of each pixel (negative for none)")
    parser_prep_region_moments.add_argument(
        '--bins-plane', type=int, metavar='PLANE',
        help="the plane of the map of bins, if the file has several along "
             "its last axis (e.g. 1 of the BINID of MaNGA's DAP)")
    parser_prep_region_spectra = parsers_prep.add_parser(
        'region_spectra', parents=[
            parser_prep_input_1, parser_prep_rest, parser_prep_roi_spec_1d,
            parser_prep_dtype, parser_common_output, parser_common],
        help="spectra in regions: the sums of the spectra of the bins of a "
             "cube")
    parser_prep_region_spectra.add_argument(
        '--bins', type=_data_file, required=True, metavar='BINS',
        help="the map of the bin of each pixel of the cube (negative for "
             "none)")
    parser_prep_region_spectra.add_argument(
        '--bins-plane', type=int, metavar='PLANE',
        help="the plane of the map of bins, if the file has several along "
             "its last axis")
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
             "of mean 0, e.g. without continuum; faint emission over much "
             "of a channel makes it too low)")
    parser_prep_gauss_hermite_moments = parsers_prep.add_parser(
        'gauss_hermite_moments', parents=[parser_common_output, parser_common],
        help="the mean and the dispersion of the Gauss-Hermite series of a "
             "table of stellar kinematics (e.g. of pPXF), for Jeans models")
    parser_prep_gauss_hermite_moments.add_argument(
        'table', type=str, help="the table, one row per bin")
    parser_prep_gauss_hermite_moments.add_argument(
        '--columns', type=str, nargs=4, required=True,
        metavar=('V', 'SIGMA', 'H3', 'H4'),
        help="the columns of V, sigma, h3 and h4")
    parser_prep_gauss_hermite_moments.add_argument(
        '--errors', type=str, nargs=4,
        metavar=('V_ERROR', 'SIGMA_ERROR', 'H3_ERROR', 'H4_ERROR'),
        help="the columns of their errors")
    parser_prep_gauss_hermite_moments.add_argument(
        '--format', type=str,
        help="the format of the table, if astropy cannot tell it")
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
    # Create parser for migrate task
    #

    parser_migrate = parsers_task.add_parser('migrate', parents=[
        parser_common],
        help="rewrite configurations in the current format")
    parser_migrate.add_argument(
        'path', type=str,
        help="a configuration file, or the folder of a project (rewritten "
             "in place)")
    parser_migrate.add_argument(
        'output', type=str, nargs='?',
        help="the file to write a configuration file to; by default the "
             "file itself")
    parser_migrate.add_argument(
        '--from-version', type=int,
        help="the version of the format of a configuration file without "
             "the section 'version': 1, before 2026-10-25; 2, before "
             "2026-10-28")

    #
    # Parse arguments and run the appropriate task
    #

    return parser


def _run_eval(args: argparse.Namespace) -> None:
    """Run the eval task."""
    run('eval', dict(
        mode=args.mode, config=args.config, profile_iters=args.profile,
        output_dir=args.output_dir, output_dir_mode=args.output_dir_mode),
        args.lenient)


def _run_prep(args: argparse.Namespace) -> None:
    """Run a prep task, with the options of the command line by name."""
    task = args.prep_task
    if task in ('pixel_brightness', 'slit_spectra', 'pixel_spectra',
                'pixel_moments'):
        options = dict(
            file_d=args.data_d, file_e=args.data_e, file_m=args.data_m,
            roi_spat=args.roi_spat, clip_min=args.clip_min,
            clip_max=args.clip_max, ccl_lcount=args.ccl_lcount,
            ccl_pcount=args.ccl_pcount, ccl_lratio=args.ccl_lratio,
            sclip_sigma=args.sclip_sigma, sclip_iters=args.sclip_iters,
            minify=args.minify, nanpad=args.nanpad, dtype=args.dtype,
            error_kind=args.error_kind)
    elif task in ('region_moments', 'region_spectra'):
        options = dict(
            file_bins=args.bins, bins_plane=args.bins_plane,
            file_d=args.data_d, file_e=args.data_e, file_m=args.data_m,
            dtype=args.dtype, error_kind=args.error_kind)
    elif task == 'gauss_hermite_moments':
        options = dict(
            file=args.table, columns=args.columns, errors=args.errors,
            format_=args.format)
    elif task in ('psf_muse', 'psf_casa_beams'):
        options = dict(file=args.cube)
    else:
        options = {}
    if task in ('slit_spectra', 'pixel_spectra', 'region_spectra'):
        options |= dict(
            roi_spec=args.roi_spec, velocity_rest=args.velocity_rest,
            velocity=args.velocity)
    if task == 'pixel_spectra':
        options['noise_error'] = args.noise_error
    run(f'prep_{task}', options | dict(
        output_dir=args.output_dir, output_dir_mode=args.output_dir_mode))


def _check_fit_arguments(
        parser: argparse.ArgumentParser,
        args: argparse.Namespace
) -> None:
    """
    Exit with an error unless the arguments of the fit task are those of
    one of its forms: a configuration, a fit to resume, or a fit and a
    configuration of its bootstrap.
    """
    if args.resume is not None and args.bootstrap is not None:
        parser.error(
            "give the output directory of a fit to resume (--resume) or to "
            "bootstrap (--bootstrap), not both")
    if args.bootstrap is not None and args.config is None:
        parser.error(
            "a bootstrap (--bootstrap) needs the configuration of its "
            "section 'bootstrap'")
    if args.bootstrap is None and (args.config is None) == (
            args.resume is None):
        parser.error(
            "give a configuration, or the output directory of a fit to "
            "resume (--resume), not both")
    # A resumed or bootstrapped fit writes into its own directory, with its
    # own log: these options would be taken and ignored
    ignored = [
        option for option, given in (
            ('--output-dir', args.output_dir is not None),
            ('--output-dir-mode', args.output_dir_mode != 'unique'),
            ('--no-evaluation-log', args.no_evaluation_log))
        if given]
    if (args.resume is not None or args.bootstrap is not None) and ignored:
        parser.error(
            f"a fit resumed (--resume) or bootstrapped (--bootstrap) writes "
            f"into its own output directory; {ignored} do not apply")


def _run_fit(args: argparse.Namespace) -> None:
    """Run the fit task (or resume a fit, or bootstrap one)."""
    if args.resume is not None:
        run('resume', dict(output_dir=args.resume), args.lenient)
    elif args.bootstrap is not None:
        run('bootstrap', dict(
            output_dir=args.bootstrap, config=args.config), args.lenient)
    else:
        run('fit', dict(
            config=args.config, output_dir=args.output_dir,
            output_dir_mode=args.output_dir_mode,
            evaluation_log=not args.no_evaluation_log), args.lenient)


def _run_plot(args: argparse.Namespace) -> None:
    """Run the plot task."""
    run('plot', dict(
        result_dir=args.result_dir, output_dir=args.output_dir,
        output_dir_mode=args.output_dir_mode, format_=args.format,
        dpi=args.dpi, only_best=args.only_best,
        posterior_mode=args.posterior_mode,
        posterior_params=args.posterior_params))


def _run_migrate(args: argparse.Namespace) -> None:
    """Run the migrate task."""
    run('migrate', dict(
        path=args.path, output=args.output, from_version=args.from_version))


def main() -> None:
    """
    Run gbkfit-cli: parse the command line, and run its task. Errors in
    configurations, and of files and folders, are reported without a
    traceback (with one if --verbose), and exit with status 1.
    """
    parser = _make_parser()
    args = parser.parse_args()
    if args.task == 'prep' and args.prep_task in (
            'pixel_moments', 'region_moments'):
        _check_moment_counts(parser, args)
    if args.task == 'fit':
        _check_fit_arguments(parser, args)

    # Configure log level before doing anything else
    log_to_console(
        logging.DEBUG if args.verbose else logging.INFO,
        ('gbkfit', 'gbkfit_cli'))
    _log.debug(
        f"cli has been called with the following arguments: {vars(args)}")

    tasks = dict(
        eval=_run_eval, prep=_run_prep, fit=_run_fit, plot=_run_plot,
        migrate=_run_migrate)
    with exit_on_user_errors(args.verbose):
        tasks[args.task](args)

    _log.info("So long and thanks for all the fish!")
