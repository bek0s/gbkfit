# Adaptive pixel sampling: tried and set aside (October 2026)

GBKFIT evaluates its smooth components (`smdisk`, `smcone`) at the centre
of each pixel of the model grid (of each subpixel, with the observation
option `scale`). Where the light or the velocity changes within a pixel,
the value at its centre is not its mean: at the cusp of a barely resolved
exponential disk, at sharp edges inside the data (the ends of the rings,
`rrange`, `azrange`, the cells of free maps), and most of all in spectral
cubes, where the velocity gradient within a pixel broadens its line, which
one sample cannot show (beam smearing below the pixel).

**What was built** (19-20 October 2026, 9 commits on c04de2c, about 2200
lines, host and CUDA): an observation option `sampling: {type: adaptive,
tolerance}`. A first pass sampled the centres; a correction in moment
space (q + lap(q)/24 for the light, the light times the velocity and the
light times v^2 + sigma^2) followed; a sparse second pass subdivided (up
to 8 x 8, blended between levels so that the model stays continuous in
its parameters) only the pixels whose estimated error was above the
tolerance, with the exact share of the pixels cut by known edges.

**What was measured:**

- Accuracy, worst pixel in % of the peak (centre / `scale` 2 / adaptive):
  thin-disk image 20 / 4.4 / 0.06; thin-disk cube 54 / 7.5 / 0.13;
  thick-disk image 2.9 / 0.34 / 0.13; cones 9-14 / 2-4 / 0.2-0.3. The total
  flux was within 0.5% for all three. Adaptive cost about 1.5-3 times the
  centre on CUDA for thin disks, more for thick disks and cones.
- Mock fits of a rotating exponential disk (scale length 2.5 pixels, PSF
  sigma 1 pixel, peak S/N 20), bias in error bars (centre / `scale` 2 /
  adaptive): dispersion +2.0 / +0.5 / +0.1; central brightness -1.8 / -0.4
  / 0.0; scale length +1.8 / +0.4 / 0.0; inclination -0.8 / -0.2 / 0.0. The
  biases are fixed, so in error bars they grow with the S/N. With a sharp
  edge inside the data, the centre was biased by 1-2 sigma, `scale` 2 gave
  error bars of the inclination 2.6 times too small, adaptive was right.
- The error estimate is the hard part: a curvature estimate from the
  neighbouring centres misses ridges narrower than about 1.5 pixels that
  fall between the centres (15-38 times the tolerance); a Richardson check
  is fooled by cusps and undercounts the widths of lines; the larger of
  the two worked.
- Real data: NGC 1313 (LVHIS moment maps; many pixels per beam) reached
  the same solution, at the same pace, with centre and with adaptive
  sampling (kinematics within 0.8 km/s, inclination within 0.05 degrees).
  The stall of that fit, which had looked like a sampling problem, was
  the float32 FFT convolution (now float64 by default).

**Decision** (20 October 2026): removed. It changed nothing in the real
fit, and it made the kernels much harder to read. Centre sampling stays,
and `scale` (as DysmalPy's oversampling) is the remedy where it matters.

**If it comes back** (the likely regime: data at z ~ 1.5, galaxies a few
pixels across, edges inside the data, high S/N): first show the problem
in a real fit, by running the same fit to the end with centre, `scale` 2
and adaptive sampling and comparing the chi-square, the parameters and
their error bars; try `scale` first. The code was the branch
`adaptive-sampling` (tip 56e344c); after the branch was deleted, it is
kept outside the repository as a git bundle, which restores it:
`git fetch adaptive-sampling.bundle adaptive-sampling:adaptive-sampling`.
