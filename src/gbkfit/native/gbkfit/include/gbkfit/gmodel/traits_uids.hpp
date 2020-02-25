#pragma once

namespace gbkfit {

// Density polar traits
constexpr int RP_TRAIT_UID_UNIFORM = 1;
constexpr int RP_TRAIT_UID_EXPONENTIAL = 2;
constexpr int RP_TRAIT_UID_GAUSS = 3;
constexpr int RP_TRAIT_UID_GGAUSS = 4;
constexpr int RP_TRAIT_UID_LORENTZ = 5;
constexpr int RP_TRAIT_UID_MOFFAT = 6;
constexpr int RP_TRAIT_UID_SECH2 = 7;
constexpr int RP_TRAIT_UID_MIXTURE_GGAUSS = 8;
constexpr int RP_TRAIT_UID_MIXTURE_MOFFAT = 9;
constexpr int RP_TRAIT_UID_NW_UNIFORM = 101;
constexpr int RP_TRAIT_UID_NW_HARMONIC = 102;
constexpr int RP_TRAIT_UID_NW_DISTORTION = 103;

// Density height traits
constexpr int RH_TRAIT_UID_UNIFORM = 1;
constexpr int RH_TRAIT_UID_EXPONENTIAL = 2;
constexpr int RH_TRAIT_UID_GAUSS = 3;
constexpr int RH_TRAIT_UID_GGAUSS = 4;
constexpr int RH_TRAIT_UID_LORENTZ = 5;
constexpr int RH_TRAIT_UID_SECH2 = 6;

// Velocity polar traits
constexpr int VP_TRAIT_UID_TAN_UNIFORM = 1;
constexpr int VP_TRAIT_UID_TAN_ARCTAN = 2;
constexpr int VP_TRAIT_UID_TAN_BOISSIER = 3;
constexpr int VP_TRAIT_UID_TAN_EPINAT = 4;
constexpr int VP_TRAIT_UID_TAN_LRAMP = 5;
constexpr int VP_TRAIT_UID_TAN_TANH = 6;
constexpr int VP_TRAIT_UID_NW_TAN_UNIFORM = 101;
constexpr int VP_TRAIT_UID_NW_TAN_HARMONIC = 102;
constexpr int VP_TRAIT_UID_NW_RAD_UNIFORM = 103;
constexpr int VP_TRAIT_UID_NW_RAD_HARMONIC = 104;
constexpr int VP_TRAIT_UID_NW_VER_UNIFORM = 105;
constexpr int VP_TRAIT_UID_NW_VER_HARMONIC = 106;
constexpr int VP_TRAIT_UID_NW_LOS_UNIFORM = 107;
constexpr int VP_TRAIT_UID_NW_LOS_HARMONIC = 108;

// Velocity height traits
constexpr int VH_TRAIT_UID_ONE = 1;

// Dispersion polar traits
constexpr int DP_TRAIT_UID_UNIFORM = 1;
constexpr int DP_TRAIT_UID_EXPONENTIAL = 2;
constexpr int DP_TRAIT_UID_GAUSS = 3;
constexpr int DP_TRAIT_UID_GGAUSS = 4;
constexpr int DP_TRAIT_UID_LORENTZ = 5;
constexpr int DP_TRAIT_UID_MOFFAT = 6;
constexpr int DP_TRAIT_UID_SECH2 = 7;
constexpr int DP_TRAIT_UID_MIXTURE_GGAUSS = 8;
constexpr int DP_TRAIT_UID_MIXTURE_MOFFAT = 9;
constexpr int DP_TRAIT_UID_NW_UNIFORM = 101;
constexpr int DP_TRAIT_UID_NW_HARMONIC = 102;
constexpr int DP_TRAIT_UID_NW_DISTORTION = 103;

// Dispersion height traits
constexpr int DH_TRAIT_UID_ONE = 1;

// Warp polar traits
constexpr int WP_TRAIT_UID_NW_UNIFORM = 101;
constexpr int WP_TRAIT_UID_NW_HARMONIC = 102;

// Selection polar traits
constexpr int SP_TRAIT_UID_AZRANGE = 1;
constexpr int SP_TRAIT_UID_NW_AZRANGE = 101;

} // namespace gbkfit
