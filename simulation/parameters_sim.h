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
    constexpr static int nGridArrays = 3, nPtsArrays = 16;

    real *grid_arrays[nGridArrays];      // pointers to gpu-allocated arrays for simulation
    real *pts_arrays[nPtsArrays];

    real InitialTimeStep, SimulationEndTime;
    real Gravity, Density, PoissonsRatio, YoungsModulus;
    real lambda, mu; // Lame
    real kappa; // bulk modulus
    real IceFrictionCoefficient;

    real XiSnow, THT_C_snow, THT_S_snow;   // hardening, critical compression, critical stretch
    real NACC_xi, NACC_alpha, NACC_beta, NACC_M_sq;
    real NACC_friction_angle;

    int GridX, GridY, GridSize;
    real cellsize, cellsize_inv, Dp_inv;

    int UpdateEveryNthStep; // run N steps without update

    real IndDiameter, IndRSq, IndVelocity, IndDepth;
    int PointsWanted, PointCountActual;
    real BlockHeight, BlockLength;

    real ParticleVolume, ParticleMass, ParticleViewSize;

    int SimulationStep;
    real SimulationTime;

    void Reset();
    void ParseFile(std::string fileName, std::string &outputDirectory);

    void ComputeLame();
    void ComputeCamClayParams();
};

#endif
