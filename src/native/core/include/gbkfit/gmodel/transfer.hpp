#pragma once

#include "gbkfit/constants.hpp"
#include "gbkfit/gmodel/gmodels.hpp"

namespace gbkfit {

// An emitter of the march of radiative transfer (see
// gmodel_transfer_march): its fields (see DiskArgs::fields), those of a
// volume (j, v and s: planes of the slab) or of a layer (j, v, s and the
// position along the depth axis, in voxels: images); its lines (the
// offset, scale and flux of each, see DiskArgs::lines), or in an image
// one row whose flux is that of its light; and a row of the ratios of the
// cubes of opacity for each line (see DiskArgs::opacity_ratios).
template<typename T>
struct TransferEmitter
{
    bool layer = false;
    const T* fields = nullptr;
    int nlines = 0;
    const T* lines = nullptr;
    const T* opacity_ratios = nullptr;
};

// Everything the march needs. Plain data, so that it can be passed by
// value to a cuda kernel.
template<typename T>
struct TransferArgs
{
    // The x and y axes of the grid, and the slab that is marched:
    // spat_size[2] planes from the plane slab_zero of a depth axis of
    // depth_size planes. A layer outside the axis is at its near or far
    // end (see gmodel_optical_depth).
    int spat_size[3] = {0, 0, 0};
    int slab_zero = 0;
    int depth_size = 0;

    // The spectral axis; that of an image has one channel, where the
    // light of each emitter is flat
    bool image = false;
    int spec_size = 0;
    T spec_step = 0;
    T spec_zero = 0;

    int nemitters = 0;
    TransferEmitter<T> emitters[TRANSFER_EMITTER_NUM_MAX];

    // The dust: nopacity cubes of the slab with the optical depth of each
    // voxel (at V, or at the wavelength of the observation: see the ratios
    // of the emitters), and nopacity images with the optical depth in
    // front of the slab (see gmodel_transfer_opacity_front)
    int nopacity = 0;
    const T* opacity = nullptr;
    T* opacity_front = nullptr;

    // The image or spectral cube, which the light adds to
    T* out = nullptr;
};

// March along the line of sight of the spaxel (x, y) through the planes of
// the slab, front to back, for the channels [c0, c1): add_light(c, value)
// gets the light of each voxel that reaches the viewer at the channel c.
// The light of a volume is mixed uniformly with the absorbers of its
// voxel, and that of a layer is at its position in its voxel, as the disk
// kernels dim them (see gmodel_attenuation).
template<typename T, typename AddLightT> constexpr void
gmodel_transfer_march(
        int x, int y, int c0, int c1, const TransferArgs<T>& a,
        AddLightT add_light)
{
    const long long nimage = 1LL * a.spat_size[0] * a.spat_size[1];
    const long long nslab = nimage * a.spat_size[2];
    const long long idx2 = index_2d_to_1d(x, y, a.spat_size[0]);

    // The optical depth of the dust in front of the voxel, and of its own
    VoxelOpacity<T> dust;
    for (int k = 0; k < a.nopacity; ++k)
        dust.front[k] = a.opacity_front[k * nimage + idx2];

    for (int z = 0; z < a.spat_size[2]; ++z)
    {
        const long long idx = idx2 + z * nimage;
        for (int k = 0; k < a.nopacity; ++k)
            dust.own[k] = a.opacity[k * nslab + idx];

        for (int e = 0; e < a.nemitters; ++e)
        {
            const TransferEmitter<T>& em = a.emitters[e];
            const long long i = em.layer ? idx2 : idx;
            const long long n = em.layer ? nimage : nslab;
            const T j = em.fields[i];
            if (!j)
                continue;
            // The opacity in front of the light. A layer is in its voxel
            // in the depth axis (that at the near or far end, if it is
            // outside), behind the fraction of the voxel in front of its
            // position (see gmodel_optical_depth). One at a position that
            // is not a number is in the first voxel, behind an optical
            // depth that is not a number either.
            VoxelOpacity<T> o = dust;
            if (em.layer)
            {
                const T pz = em.fields[3 * n + i];
                const T p = std::isnan(pz) ? pz : std::min(
                        std::max(pz + T{0.5}, T{0}), T(a.depth_size));
                const int zl = std::isnan(p)
                        ? 0 : std::min(int(p), a.depth_size - 1);
                if (zl != a.slab_zero + z)
                    continue;
                for (int k = 0; k < a.nopacity; ++k) {
                    o.front[k] += (p - zl) * o.own[k];
                    o.own[k] = 0;
                }
            }
            const T v = em.fields[n + i];
            const T s = em.fields[2 * n + i];
            for (int l = 0; l < em.nlines; ++l)
            {
                // The channels of the line (the flat light of an image
                // is in its one channel), and its light there
                const T* line = em.lines + 3 * l;
                const T centre = line[0] + line[1] * v;
                const T sigma = line[1] * s;
                int cmin = 0, cmax = 0;
                if (!a.image && !gmodel_line_range(
                        centre, sigma, c0, c1, a.spec_step, a.spec_zero,
                        cmin, cmax))
                    continue;
                const T flux = line[2] * (a.nopacity
                        ? j * gmodel_attenuation(
                                o, a.nopacity,
                                em.opacity_ratios + l * a.nopacity)
                        : j);
                if (a.image)
                    add_light(0, flux);
                else
                    gmodel_line_channels(
                            flux, centre, sigma, cmin, cmax,
                            a.spec_step, a.spec_zero, add_light);
            }
        }

        for (int k = 0; k < a.nopacity; ++k)
            dust.front[k] += dust.own[k];
    }
}

// Add the dust of the slab along the line of sight of the spaxel (x, y)
// to the optical depth in front of the slab, for the next slab
template<typename T> constexpr void
gmodel_transfer_opacity_front(int x, int y, const TransferArgs<T>& a)
{
    const long long nimage = 1LL * a.spat_size[0] * a.spat_size[1];
    const long long nslab = nimage * a.spat_size[2];
    const long long idx2 = index_2d_to_1d(x, y, a.spat_size[0]);
    for (int k = 0; k < a.nopacity; ++k)
    {
        T tau = a.opacity_front[k * nimage + idx2];
        for (int z = 0; z < a.spat_size[2]; ++z)
            tau += a.opacity[k * nslab + idx2 + z * nimage];
        a.opacity_front[k * nimage + idx2] = tau;
    }
}

} // namespace gbkfit
