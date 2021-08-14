
import gbkfit.params.utils as paramutils


def parse_parameters(info, descs, loader):
    infos, exprs = paramutils.parse_param_info(info, descs)[4:]
    for key, val in infos.items():
        try:
            infos[key] = loader(val)
        except RuntimeError as e:
            raise RuntimeError(
                f"could not parse information for parameter '{key}'; "
                f"reason: {str(e)}") from e
    return infos | exprs
