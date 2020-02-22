
import itertools


def make_ranges(size, grid):
    dim_slices_nd = []
    for data_dim_size, grid_dim_size in zip(size, grid):
        dim_slices = []
        tile_size_q = data_dim_size // grid_dim_size
        tile_size_r = data_dim_size % grid_dim_size
        for i in range(grid_dim_size):
            dim_slices.append([i * tile_size_q, (i + 1) * tile_size_q - 1])
        dim_slices[-1][1] += tile_size_r
        dim_slices_nd.append(dim_slices)
    return list(itertools.product(*dim_slices_nd))


def make_range_slice(range_nd):
    return tuple([slice(range_1d[0], range_1d[1] + 1)
                  for range_1d in range_nd])


def make_range_prefix(range_nd):
    return '_'.join([':'.join(map(str, range_1d)) for range_1d in range_nd])


def rename_extra(extra, range_nd):
    return {
        f'{key}_tile_{make_range_prefix(range_nd)}': value
        for key, value in extra.items()
    }
