#pragma once

#include "gbkfit/gmodel/component.hpp"

namespace gbkfit {

// A cone, as the smooth cone kernel takes it (with the grid it is
// evaluated on and the outputs: see GridArgs). Its frame is that of a
// thick disk whose spin axis is the axis of the cone: the apex at its
// centre, the positive side of the cone along +z. Plain data, so that it
// can be passed by value to a cuda kernel.
template<typename T>
struct ConeArgs
{
    // The sides of the cone that it has: the positive (z >= 0 in its
    // frame) and the negative (z <= 0)
    bool positive = true;
    bool negative = true;

    // The radial nodes (the shells, spheres around the apex), and the
    // geometry: one value each. wall is the angle (degrees) of the
    // mid-surface of its walls from its axis. There is no vsys without
    // velocity traits.
    int nrnodes = 0;
    const T* rnodes = nullptr;
    const T* vsys = nullptr;
    const T* xpos = nullptr;
    const T* ypos = nullptr;
    const T* posa = nullptr;
    const T* incl = nullptr;
    const T* wall = nullptr;

    // Density (r), velocity (v) and dispersion (d) polar and height
    // traits, and selection (s) polar traits. The polar traits are
    // functions of the distance r from the apex and the azimuth around the
    // axis, and the height traits of the distance from the mid-surface of
    // the walls; a height trait set has as many traits as its polar trait
    // set. The velocity polar traits are speeds, each along its direction
    // (VPT_DIRECTION_TAN or VPT_DIRECTION_OUT, see vp_cone_project).
    TraitSet<T> rpt, rht, vpt, vht, dpt, dht, spt;
    const int* vpt_directions = nullptr;
};

} // namespace gbkfit
