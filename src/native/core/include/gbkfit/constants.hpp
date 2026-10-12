#pragma once

namespace gbkfit {

template<typename T>
constexpr T LINE_WIDTH_MULTIPLIER = 5;

constexpr int TRAIT_NUM_MAX = 4;

// The most cubes of opacity (one for each extinction law of the dust; see
// DiskArgs::opacity) that the disk kernels read
constexpr int OPACITY_NUM_MAX = 4;

// The most emitters that the march of radiative transfer takes (see
// TransferArgs)
constexpr int TRANSFER_EMITTER_NUM_MAX = 16;

} // namespace gbkfit
