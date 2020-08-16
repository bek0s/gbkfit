
import gbkfit.math


def ensure_same_attrib_value(data, method, class_desc):
    attr = {k: getattr(v, method)() for k, v in data.items()}
    if len(set(attr.values())) > 1:
        raise RuntimeError(
            f"{class_desc} contains data items of different {method}: "
            f"{str(attr)}")


def pix2world(pos, step, rpix, rval, rota):
    pos = pos - rpix
    pos[0], pos[1] = gbkfit.math.transform_lh_rotate_z(pos[0], pos[1], rota)
    pos = pos * step + rval
    return pos
