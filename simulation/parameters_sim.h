#ifndef P_SIM_H
#define P_SIM_H

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <Eigen/Core>
#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

typedef double real;
//typedef float real;
typedef Eigen::Vector2<real> Vector2r;
typedef Eigen::Matrix2<real> Matrix2r;


// variables related to the formulation of the model

namespace icy { struct SimParams; }

struct icy::SimParams
{
public:
    constexpr static double pi = 3.14159265358979323846;
    constexpr static int dim = 2;
    constexpr static int nGridArrays = 3, nPtsArrays = 20;

    // index of the corresponding array in SoA
    constexpr static size_t posx = 0;
    constexpr static size_t posy = 1;
    constexpr static size_t idx_q_snow = 2;
    constexpr static size_t idx_NACCAlphaP = 3;
    constexpr static size_t velx = 4;
    constexpr static size_t vely = 5;
    constexpr static size_t Fe00 = 6;
    constexpr static size_t Fe01 = 7;
    constexpr static size_t Fe10 = 8;
    constexpr static size_t Fe11 = 9;
    constexpr static size_t Bp00 = 10;
    constexpr static size_t Bp01 = 11;
    constexpr static size_t Bp10 = 12;
    constexpr static size_t Bp11 = 13;
    constexpr static size_t idx_Jp = 14;
    constexpr static size_t idx_zeta = 15;  // accumulate shear
    // visualization only
    constexpr static size_t idx_p0 = 16;
    constexpr static size_t idx_p = 17;
    constexpr static size_t idx_q = 18;
    constexpr static size_t idx_psi = 19;

    real *grid_array;      // device-side grid data
    real *pts_array;
    size_t nPtsPitch, nGridPitch; // in bytes (!), for coalesced access on the device

    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation


    int PointsWanted, nPts;
    int GridX, GridY;
    real GridXDimension;

    real InitialTimeStep, SimulationEndTime;
    real Gravity, Density, PoissonsRatio, YoungsModulus;
    real lambda, mu; // Lame
    real kappa; // bulk modulus
    real IceFrictionCoefficient;

    real XiSnow, THT_C_snow, THT_S_snow;   // hardening, critical compression, critical stretch
    real IceCompressiveStrength, IceTensileStrength, IceShearStrength;
    real NACC_beta, NACC_M_sq;     // these are all computed
    real NACC_max_strain;
    real NACC_magic_coeff;

    // NACC - unused
    real NACC_xi, NACC_alpha;


    real cellsize, cellsize_inv, Dp_inv;

    int UpdateEveryNthStep; // run N steps without update

    real IndDiameter, IndRSq, IndVelocity, IndDepth;
    real BlockHeight, BlockLength;
    int HoldBlockOnTheRight;

    real ParticleVolume, ParticleMass, ParticleViewSize;

    int SimulationStep;
    real SimulationTime;

    real H0, H1, H2, H3;
    real SandYM;


    void Reset();
    std::string ParseFile(std::string fileName);

    void ComputeLame();
//    void ComputeCamClayParams();
    void ComputeCamClayParams2();
    void ComputeHelperVariables();
};

#endif
