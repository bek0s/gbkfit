#pragma once

namespace gbkfit {

// The number of the moments of the light of orbits in a cell (see
// OrbitArgs::cell_moments)
constexpr int ORBITS_CELL_MOMENTS = 9;

// The potential of the gravity of a galaxy in the frame of a spheroid (see
// SpheroidArgs), symmetric about its axis and its plane, as a table: on
// the nodes u = ln r = range[0] + i range[1] (r in arcsec) for i in [0,
// nr) and b = j pi / (2 (nb - 1)) for j in [0, nb), b the angle from the
// plane (atan2(|z|, R)), 4 values at each node (values[4 (i nb + j) + k]):
// the potential Phi ((km/s)^2), its derivatives along u and b, and along
// both (k = 0 to 3). Between the nodes it is bicubic Hermite: one smooth
// function, whose forces are its gradient, so orbits conserve its energy.
// Below the first radius r0 it is Phi(r0, b) + Phi_u(r0, b) (r^2 / r0^2 -
// 1) / 2 (harmonic), and beyond the last one rn Phi(rn, b) + Phi_u(rn, b)
// (1 - rn / r) (the gravity of a point of the force at rn): each with the
// value and the slope along u of the table at its end.
// Plain data, so that it can be passed by value to a cuda kernel.
template<typename T>
struct PotentialArgs
{
    int nr = 0;
    int nb = 0;
    const T* range = nullptr;
    const T* values = nullptr;
};

// An integration of orbits in the potential of a galaxy (see
// PotentialArgs) and their recording (see orbits_trajectory and
// orbits_record_orbit), each orbit as it is seen from the sky: the time it
// spends in cells of its spheroid, and its light and velocities in
// targets of the sky (e.g. the Voronoi bins of observations), averaged
// over its time. Plain data, so that it can be passed by value to a cuda
// kernel.
template<typename T>
struct OrbitArgs
{
    PotentialArgs<T> potential;

    // The orbits: the start of each (starts[6k] to starts[6k + 5]: x, y, z
    // (arcsec) and vx, vy, vz (km/s), in the frame of the spheroid), each
    // integrated for turns periods of the circular orbit of its energy in
    // the plane (see orbits_circular_period), in time-symmetric
    // kick-drift-kick leapfrog steps (Hut,
    // Makino & McMillan 1995), each step the mean of accuracy / f at its
    // start and its end, f^2 = (|Phi_u| + v^2) / r^2 (the frequencies of
    // the circular orbit of its radius and of its crossing), and at most
    // max_steps of them. Each orbit is recorded at nsamples times evenly
    // spaced in its time (each at the time of its fraction 1 / nsamples)
    // and, on the sky, each time in 2 nphi copies (each a share of it):
    // turned about the axis by nphi angles evenly spaced (by a fraction
    // of the turn that changes from time to time), each also mirrored
    // about the plane. With twins, each orbit k is also recorded
    // reversed in time (its velocities negated): the orbits 2k and 2k + 1
    // of the outputs; otherwise k.
    int norbits = 0;
    const T* starts = nullptr;
    T turns = 0;
    T accuracy = 0;
    int max_steps = 0;
    int nsamples = 0;
    int nphi = 0;
    bool twins = false;

    // The geometry of the spheroid: its position angle and its
    // inclination (degrees; see transform_cpos_posa_incl)
    T posa = 0;
    T incl = 0;

    // The cells (optional): between the radii cell_r (arcsec, ncell_r + 1
    // edges, increasing) and the angles from the plane cell_b (radians,
    // ncell_b + 1 edges, increasing); the time fraction of orbit k in
    // cell (i, j) at cells[(k ncell_r + i) ncell_b + j]
    int ncell_r = 0;
    int ncell_b = 0;
    const T* cell_r = nullptr;
    const T* cell_b = nullptr;
    T* cells = nullptr;

    // The sky (optional): nsky grids of cells, grid s of grid_sizes[2s]
    // by grid_sizes[2s + 1] cells, a point (x, y) of the sky (arcsec from
    // the centre of the spheroid, as in transform_cpos_posa_incl) in its
    // cell (floor(cx), floor(cy)), cx = g[0] x + g[1] y + g[2] and cy =
    // g[3] x + g[4] y + g[5] (g = grids + 6s), its cell (x, y) the cell
    // grid_cells[s] + y nx + x of the table: each cell c of the table
    // gives its light to the targets targets[e], each the share
    // shares[e], for e in [offsets[c], offsets[c + 1]) (e.g. the shares
    // of the Voronoi bins of an observation in the light of a point in
    // the cell after the PSF). Velocities are on the line of sight,
    // relative to that of the centre.
    int nsky = 0;
    const int* grid_sizes = nullptr;
    const T* grids = nullptr;
    const long long* grid_cells = nullptr;
    const long long* offsets = nullptr;
    const int* targets = nullptr;
    const T* shares = nullptr;
    int ntargets = 0;

    // The moments of each orbit in the targets (optional): each target t
    // has a Gauss-Hermite series of the mean target_v[t] and the
    // dispersion target_s[t] (km/s); the moments of orbit k in target t,
    // at moments[(k ntargets + t) (order + 2) + m], are its light (m = 0)
    // and gamma h_n (m = n + 1, n in [0, order]) of its velocities in the
    // series of the target (see orbits_gauss_hermite).
    const T* target_v = nullptr;
    const T* target_s = nullptr;
    int order = 0;
    T* moments = nullptr;

    // The spectrum of all the orbits in each target (optional), of the
    // light of orbit k light[k]: its light at the velocity v in the
    // channel floor(channel_scale v + channel_offset) of nchannels, at
    // spectra[c ntargets + t] (which the orbits add to together)
    const T* light = nullptr;
    int nchannels = 0;
    T channel_scale = 0;
    T channel_offset = 0;
    T* spectra = nullptr;

    // The moments of the light of all the orbits in the cells (optional,
    // with the cells and light): in cell (i, j), at cell_moments[
    // ORBITS_CELL_MOMENTS (i ncell_b + j) + m], the sums over the orbits
    // of light[k] times the fraction of the time of orbit k in the cell
    // (m = 0) and times its velocities there (km/s, in the frame of the
    // spheroid; m = 1 to 8): v_r, v_R, v_phi, v_r^2, v_theta^2, v_phi^2,
    // v_R^2 and v_z^2 (r, theta and phi spherical, theta from the axis;
    // R, phi and z cylindrical; the means of v_theta and v_z are 0, of
    // orbits mirrored about the plane, which the sky sees)
    T* cell_moments = nullptr;

    // The number of steps of each orbit (optional), and its energy
    // ((km/s)^2, optional) at its start (energies[2k]) and its end
    // (energies[2k + 1])
    int* steps = nullptr;
    T* energies = nullptr;
    // The samples of the orbits, kept from their integration to their
    // recording: they are integrated, then recorded, nchunk at a time,
    // sample n of the k-th orbit of a chunk at samples[6 (k nsamples +
    // n)] (see orbits_trajectory)
    int nchunk = 0;
    T* samples = nullptr;
};

} // namespace gbkfit
