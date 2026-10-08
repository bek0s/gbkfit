#pragma once

namespace gbkfit {

template<typename T>
constexpr T LINE_WIDTH_MULTIPLIER = 5;

constexpr int TRAIT_NUM_MAX = 4;

// Seed of the random numbers of the Monte Carlo clouds. Every evaluation
// uses the same seed, so a model is a deterministic function of its
// parameters.
constexpr unsigned int MCDISK_SEED = 0;

} // namespace gbkfit
