# DNest4

C++ diffusive nested sampler (Brewer & Foreman-Mackey 2018), with its
RJObject for models with a varying number of components. Used by the
dnest4 native module (src/native/dnest4).

- Source: https://github.com/eggplantbren/DNest4 (directory `code`,
  without its Examples and Templates)
- Commit: 40366c01384953900133e2c95e3d4998c082d29c (2026-07-13)
- License: MIT (see LICENSE)

The files in this directory are copies from upstream, with two changes,
marked `GBKFIT:`:

- SamplerImpl.h: Sampler::run writes no sampler_state.txt to the working
  directory when DNEST4_NO_SAMPLER_STATE is defined.
- Options.h: Options::set_files sets the names of the output files.

CommandLineOptions.cpp is not built (it needs POSIX getopt, and the
module takes no command line).
