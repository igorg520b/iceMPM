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
typedef Eigen::Vector2<real> Vector2r;
typedef Eigen::Matrix2<real> Matrix2r;

// variables related to the formulation of the model

namespace icy { struct SimParams; }

struct icy::SimParams
{
public:
    constexpr static double pi = 3.14159265358979323846;
    constexpr static int dim = 2;
    constexpr static int nGridArrays = 3, nPtsArrays = 14;

    // index of the corresponding array in SoA
    constexpr static size_t posx = 0;
    constexpr static size_t posy = 1;
    constexpr static size_t velx = 2;
    constexpr static size_t vely = 3;
    constexpr static size_t Bp00 = 4;
    constexpr static size_t Bp01 = 5;
    constexpr static size_t Bp10 = 6;
    constexpr static size_t Bp11 = 7;
    constexpr static size_t Fe00 = 8;
    constexpr static size_t Fe01 = 9;
    constexpr static size_t Fe10 = 10;
    constexpr static size_t Fe11 = 11;
    constexpr static size_t idx_NACCAlphaP = 12;
    constexpr static size_t idx_q = 13;

    real *grid_array;      // device-side grid data
    real *pts_array;
    size_t nPtsPitch, nGridPitch; // in bytes (!), for coalesced access on the device
    int PointsWanted, nPts;
    int GridX, GridY;

    real InitialTimeStep, SimulationEndTime;
    real Gravity, Density, PoissonsRatio, YoungsModulus;
    real lambda, mu; // Lame
    real kappa; // bulk modulus
    real IceFrictionCoefficient;

    real XiSnow, THT_C_snow, THT_S_snow;   // hardening, critical compression, critical stretch
    real NACC_xi, NACC_alpha, NACC_beta, NACC_M_sq;
    real NACC_friction_angle;

    real cellsize, cellsize_inv, Dp_inv;

    int UpdateEveryNthStep; // run N steps without update

    real IndDiameter, IndRSq, IndVelocity, IndDepth;
    real BlockHeight, BlockLength;

    real ParticleVolume, ParticleMass, ParticleViewSize;

    int SimulationStep;
    real SimulationTime;

    double H0, H1, H2, H3;

    void Reset();
    std::string ParseFile(std::string fileName);

    void ComputeLame();
    void ComputeCamClayParams();
    void ComputeHelperVariables();
};

#endif
