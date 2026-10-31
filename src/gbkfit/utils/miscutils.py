import pathlib



def make_unique_path(path: pathlib.Path) -> pathlib.Path:
    """
    Generate a unique file path by appending an incrementing number.

    If the given path already exists, appends '_1', '_2', etc.,
    until a unique path is found.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return path
    base, ext = path.stem, path.suffix
    i = 1
    while (new_path := path.with_name(f"{base}_{i}{ext}")).exists():
        i += 1
    return new_path
