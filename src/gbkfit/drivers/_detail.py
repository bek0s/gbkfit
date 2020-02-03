
import numpy as np


def check_dtype(classes, dtype):
    if dtype not in classes:
        requested = np.dtype(dtype).name
        supported = ', '.join([np.dtype(dt).name for dt in classes])
        raise RuntimeError(
            f"The requested data type is not supported "
            f"(requested: {requested}; supported: {supported}).")
