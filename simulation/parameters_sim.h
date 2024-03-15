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
typedef Eigen::Array2<real> Array2r;


// variables related to the formulation of the model

namespace icy { struct SimParams; }

struct icy::SimParams
{
public:
    constexpr static double pi = 3.14159265358979323846;
    constexpr static int dim = 2;
    constexpr static int nGridArrays = 3;

    // index of the corresponding array in SoA
    constexpr static size_t idx_utility_data = 0;
    constexpr static size_t idx_Jp_inv = 1;
    constexpr static size_t posx = 2;
    constexpr static size_t velx = posx + 2;
    constexpr static size_t Fe00 = velx + 2;
    constexpr static size_t Bp00 = Fe00 + 4;
    constexpr static size_t nPtsArrays = Bp00 + 4;

    real *grid_array;      // device-side grid data
    real *pts_array;
    size_t nPtsPitch, nGridPitch; // in number of elements(!), for coalesced access on the device
    constexpr static int n_indenter_subdivisions = 360;
    real *indenter_force_accumulator; // size is 2*n_indenter_subdivisions
    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation

    int nPts;
    int GridX, GridY, GridTotal;
    real GridXDimension;

    real InitialTimeStep, SimulationEndTime;
    real Gravity, Density, PoissonsRatio, YoungsModulus;
    real lambda, mu, kappa; // Lame

    real IceCompressiveStrength, IceTensileStrength, IceShearStrength;
    real NACC_beta, NACC_M, NACC_Msq;     // these are all computed

    real DP_tan_phi, DP_threshold_p;

    real cellsize, cellsize_inv, Dp_inv;

    int UpdateEveryNthStep; // run N steps without update

    real IndDiameter, IndRSq, IndVelocity, IndDepth;
    real xmin, xmax, ymin, ymax;            // bounding box of the material
    int nxmin, nxmax, nymin, nymax;         // same, but nuber of grid cells

    real ParticleVolume, ParticleMass, ParticleViewSize;

    int SimulationStep;
    real SimulationTime;

    real indenter_x, indenter_x_initial, indenter_y, indenter_y_initial;
    real Volume;
    int SetupType;  // 0 - ice block horizontal indentation; 1 - cone uniaxial compression

    void Reset();
    std::string ParseFile(std::string fileName);

    void ComputeLame();
    void ComputeCamClayParams2();
    void ComputeHelperVariables();
    void ComputeIntegerBlockCoords();
};

#endif
