
import argparse
import logging.config
import os


logging.config.dictConfig({
    'version': 1,
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
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
})

log = logging.getLogger(__name__)


class _CheckDataCount(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(namespace.data) != len(values):
            parser.error(
                f"Argument '{option_string}' must have the same length with "
                f"argument 'data'")


def number_range(type_, min_, max_):

    if min_ is not None:
        assert isinstance(min_, type_)
    if max_ is not None:
        assert isinstance(max_, type_)

    def number_range_checker(arg):
        try:
            num = type_(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"must be of type {type_.__name__}")
        if min_ is not None and max_ is None:
            if num < min_:
                raise argparse.ArgumentTypeError(f"must be larger than {min_}")
        if max_ is not None and min_ is None:
            if num > max_:
                raise argparse.ArgumentTypeError(f"must be smaller than {max_}")
        if min_ is not None and max_ is not None:
            if num < min_ or num > max_:
                raise argparse.ArgumentTypeError(
                    f"must be in range [{min_}, {max_}]")
        return num

    return number_range_checker


def main():

    from gbkfit import __version__ as version

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version=version)
    parsers_task = parser.add_subparsers(dest='task')
    parsers_task.required = True

    parser_common = argparse.ArgumentParser(add_help=False)
    parser_common.add_argument('--workdir', type=str, default='.')

    #
    # Create parser for eval task
    #

    parser_eval = parsers_task.add_parser('eval', parents=[parser_common])
    parser_eval.add_argument(
        'config', type=str)
    parser_eval.add_argument(
        '--perf', type=int, default=0, help='')

    #
    # Create parser for prep task
    #

    # ...
    parser_prep_common = argparse.ArgumentParser(add_help=False)
    parser_prep_common.add_argument(
        '--dtype', type=str, default='float32',
        help='data type of the output')
    parser_prep_common.add_argument(
        '--minify', action='store_true',
        help='crop the edges of the input data until valid pixels are found')
    # ...
    parser_prep_input_1 = argparse.ArgumentParser(add_help=False)
    parser_prep_input_1.add_argument(
        'data', type=str,
        help="input data (measurements)")
    parser_prep_input_1.add_argument(
        '--data-e', type=str,
        metavar='ERRORS',
        help='input data (uncertainties)')
    parser_prep_input_1.add_argument(
        '--data-m', type=str,
        metavar='MASK',
        help='input data (mask)')
    # ...
    parser_prep_input_n = argparse.ArgumentParser(add_help=False)
    parser_prep_input_n.add_argument(
        'data', nargs='+', type=str,
        help="input data (measurements)")
    parser_prep_input_n.add_argument(
        '--data-e', nargs='+', type=str, action=_CheckDataCount,
        help='input data (uncertainties)')
    parser_prep_input_n.add_argument(
        '--data-m', nargs='+', type=str, action=_CheckDataCount,
        help='input data (mask)')
    # ...
    parser_prep_roi_spat_1d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spat_1d.add_argument(
        '--roi-spat', nargs=2, type=int,
        metavar=('MIN', 'MAX'),
        help='region of interest (spatial)')
    # ...
    parser_prep_roi_spat_2d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spat_2d.add_argument(
        '--roi-spat', nargs=4, type=int,
        metavar=('L', 'R', 'B', 'T'),
        help='region of interest (spatial)')
    # ...
    parser_prep_roi_spec_1d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spec_1d.add_argument(
        '--roi-spec', nargs=2, type=int,
        metavar=('MIN', 'MAX'),
        help='region of interest (spectral)')
    # ...
    parser_prep_clip_1 = argparse.ArgumentParser(add_help=False)
    parser_prep_clip_1.add_argument(
        '--clip-min', nargs=1, type=float,
        metavar='MIN',
        help='minimum clip threshold value; '
             'all values below this threshold will be set to nan')
    parser_prep_clip_1.add_argument(
        '--clip-max', nargs=1, type=float,
        metavar='MAX',
        help='maximum clip threshold value; '
             'all values above this threshold will be set to nan')
    parser_prep_clip_1.add_argument(
        '--sclip-sigma', type=float,
        metavar='SIGMA',
        help='the number of standard deviations to use for clipping')
    parser_prep_clip_1.add_argument(
        '--sclip-iters', type=int, default=5,
        metavar='ITERS',
        help='the maximum number of sigma-clipping iterations to perform')
    # ...
    parser_prep_clip_n = argparse.ArgumentParser(add_help=False)
    parser_prep_clip_n.add_argument(
        '--clip-min', nargs='+', type=float, action=_CheckDataCount,
        help='')
    parser_prep_clip_n.add_argument(
        '--clip-max', nargs='+', type=float, action=_CheckDataCount,
        help='')
    parser_prep_clip_n.add_argument(
        '--sclip-sigma', nargs='+', type=float, action=_CheckDataCount,
        help='')
    parser_prep_clip_n.add_argument(
        '--sclip-iters', nargs='+', type=int, default=1, action=_CheckDataCount,
        help='')
    # ...
    parser_prep_ccl = argparse.ArgumentParser(add_help=False)
    parser_prep_ccl.add_argument(
        '--ccl-lcount', type=number_range(int, 1, None),
        metavar='COUNT',
        help='connected component labeling; maximum number of labels')
    parser_prep_ccl.add_argument(
        '--ccl-pcount', type=number_range(int, 1, None),
        metavar='COUNT',
        help='connected component labeling; minimum area per label')
    parser_prep_ccl.add_argument(
        '--ccl-lratio', type=number_range(float, 0.0, 1.0),
        metavar='RATIO',
        help='connected component labeling; minimum area ratio per label')
    # ...
    parser_prep = parsers_task.add_parser('prep')
    parsers_prep = parser_prep.add_subparsers(dest='prep_task')
    parsers_prep.required = True
    parsers_prep.add_parser('image', parents=[
        parser_prep_input_1,
        parser_prep_roi_spat_2d,
        parser_prep_clip_1, parser_prep_ccl,
        parser_prep_common, parser_common])
    parsers_prep.add_parser('lslit', parents=[
        parser_prep_input_1,
        parser_prep_roi_spat_1d, parser_prep_roi_spec_1d,
        parser_prep_clip_1, parser_prep_ccl,
        parser_prep_common, parser_common])
    parsers_prep.add_parser('mmaps', parents=[
        parser_prep_input_n,
        parser_prep_roi_spat_2d,
        parser_prep_clip_n, parser_prep_ccl,
        parser_prep_common, parser_common])
    parsers_prep.add_parser('scube', parents=[
        parser_prep_input_1,
        parser_prep_roi_spat_2d, parser_prep_roi_spec_1d,
        parser_prep_clip_1, parser_prep_ccl,
        parser_prep_common, parser_common])

    #
    # Create parser for fit task
    #

    parser_fit = parsers_task.add_parser('fit', parents=[parser_common])
    parser_fit.add_argument('config', type=str)

    #
    # Create parser for plot task
    #

    parser_plot = parsers_task.add_parser('plot', parents=[parser_common])
    parser_plot.add_argument('result', type=str)
    parser_plot.add_argument('--show', action='store_true')
    parser_plot.add_argument('--params', nargs='+', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)
    os.chdir(args.workdir)

    if args.task == 'eval':
        import gbkfit.tasks.eval
        gbkfit.tasks.eval.eval_(args.config, args.perf)

    elif args.task == 'prep':
        import gbkfit.tasks.prep
        if args.prep_task == 'image':
            gbkfit.tasks.prep.prep_image(
                args.data, args.data_e, args.data_m,
                args.roi_spat, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters, args.minify, args.dtype)
        elif args.prep_task == 'lslit':
            gbkfit.tasks.prep.prep_lslit(
                args.data, args.data_e, args.data_m,
                args.roi_spat, args.roi_spec, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters, args.minify, args.dtype)
        elif args.prep_task == 'mmaps':
            gbkfit.tasks.prep.prep_mmaps(
                args.data, args.data_e, args.data_m,
                args.roi_spat, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters, args.minify, args.dtype)
        elif args.prep_task == 'scube':
            gbkfit.tasks.prep.prep_scube(
                args.data, args.data_e, args.data_m,
                args.roi_spat, args.roi_spec, args.clip_min, args.clip_max,
                args.ccl_lcount, args.ccl_pcount, args.ccl_lratio,
                args.sclip_sigma, args.sclip_iters, args.minify, args.dtype)

    elif args.task == 'fit':
        import gbkfit.tasks.fit
        gbkfit.tasks.fit.fit(args.config)

    elif args.task == 'plot':
        import gbkfit.tasks.plot
        gbkfit.tasks.plot.plot(args.result, args.show)

    print("So long and thanks for all the fish!")


if __name__ == '__main__':
    main()
