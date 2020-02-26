
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
    parser_eval.add_argument('config', type=str)
    parser_eval.add_argument('--perf', type=int, default=0)

    #
    # Create parser for prep task
    #

    parser_prep_common = argparse.ArgumentParser(add_help=False)
    parser_prep_common.add_argument('--minify', action='store_true')
    parser_prep_common.add_argument('--dtype', type=str, default='float32')
    parser_prep_input_1 = argparse.ArgumentParser(add_help=False)
    parser_prep_input_1.add_argument('data', type=str)
    parser_prep_input_1.add_argument('--data-e', type=str)
    parser_prep_input_1.add_argument('--data-m', type=str)
    parser_prep_input_n = argparse.ArgumentParser(add_help=False)
    parser_prep_input_n.add_argument('data', nargs='+', type=str)
    parser_prep_input_n.add_argument('--data-e', nargs='+', type=str, action=_CheckDataCount)
    parser_prep_input_n.add_argument('--data-m', nargs='+', type=str, action=_CheckDataCount)
    parser_prep_roi_spat_1d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spat_1d.add_argument('--roi-spat', nargs=2, type=int)
    parser_prep_roi_spat_2d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spat_2d.add_argument('--roi-spat', nargs=4, type=int)
    parser_prep_roi_spec_1d = argparse.ArgumentParser(add_help=False)
    parser_prep_roi_spec_1d.add_argument('--roi-spec', nargs=2, type=int)
    parser_prep_clip_1 = argparse.ArgumentParser(add_help=False)
    parser_prep_clip_1.add_argument('--clip-min', nargs=1, type=float)
    parser_prep_clip_1.add_argument('--clip-max', nargs=1, type=float)
    parser_prep_clip_1.add_argument('--sclip-sigma', type=float)
    parser_prep_clip_1.add_argument('--sclip-iters', type=int, default=1)
    parser_prep_clip_n = argparse.ArgumentParser(add_help=False)
    parser_prep_clip_n.add_argument('--clip-min', nargs='+', type=float, action=_CheckDataCount)
    parser_prep_clip_n.add_argument('--clip-max', nargs='+', type=float, action=_CheckDataCount)
    parser_prep_clip_n.add_argument('--sclip-sigma', nargs='+', type=float, action=_CheckDataCount)
    parser_prep_clip_n.add_argument('--sclip-iters', nargs='+', type=int, default=1, action=_CheckDataCount)
    parser_prep_ccl = argparse.ArgumentParser(add_help=False)
    parser_prep_ccl.add_argument('--ccl-lcount', type=int)
    parser_prep_ccl.add_argument('--ccl-pcount', type=int)
    parser_prep_ccl.add_argument('--ccl-lratio', type=float)

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
