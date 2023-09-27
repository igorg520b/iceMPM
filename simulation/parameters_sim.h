#ifndef P_SIM_H
#define P_SIM_H

#include <iostream>
#include <Eigen/Core>

typedef double real;
typedef Eigen::Vector2<real> Vector2r;
typedef Eigen::Matrix2<real> Matrix2r;

// variables related to the formulation of the model

namespace icy { struct SimParams; }

struct icy::SimParams
{
public:
    constexpr static double pi = 3.14159265358979323846;


    real InitialTimeStep, SimulationEndTime;
    real Gravity, Density, PoissonsRatio, YoungsModulus;
    real lambda, mu; // Lame
    real kappa; // bulk modulus
    real IceFrictionCoefficient;

    real XiSnow, THT_C_snow, THT_S_snow;   // hardening, critical compression, critical stretch
    real NACC_xi, NACC_alpha, NACC_beta, NACC_M_sq;
    real NACC_friction_angle;

    int GridX, GridY;
    real cellsize, cellsize_inv, Dp_inv;

    int UpdateEveryNthStep;

    real IndDiameter, IndRSq, IndVelocity, IndDepth;
    int PointsWanted, PointCountActual;
    real BlockHeight, BlockLength;

    real ParticleVolume, ParticleMass, ParticleViewSize;

    int SimulationStep;
    real SimulationTime;
    real MemAllocGrid, MemAllocPoints, MemAllocTotal;

    bool useGPU;
    void Reset()
    {
#define PARAMS2
#ifdef PARAMS2
        InitialTimeStep = 8.e-6;
        YoungsModulus = 5.e8;
        NACC_beta = 2;
        NACC_xi = 5;
        NACC_alpha = std::log(1.-1.e-6);
        PointsWanted = 1'000'000;
        GridX = 512;
        GridY = 200;
        ParticleViewSize = 1.1f;
#else
        InitialTimeStep = 3.e-5;
        YoungsModulus = 5.e8;
        NACC_beta = 0.1;
        NACC_xi = 3;
        NACC_alpha = std::log(1.-5.e-5);
        PointsWanted = 35'000;
        GridX = 128;
        GridY = 55;
        ParticleViewSize = 3.5f;
        /*
        InitialTimeStep = 1e-4;//1.e-5;
        YoungsModulus = 1.e7;
        NACC_beta = 0.3;
        NACC_xi = 0.9;
        NACC_alpha = std::log(1.-5.e-5);
        PointsWanted = 25'000;
        GridX = 128;
        GridY = 55;
        ParticleViewSize = 5.5f;
        */
#endif

        NACC_friction_angle = 80;
        ComputeCamClayParams();

        useGPU = true;

        SimulationEndTime = 15;
        UpdateEveryNthStep = (int)(1.f/(200*InitialTimeStep));

        cellsize = (real)3.33/(GridX);
        cellsize_inv = 1./cellsize;
        Dp_inv = 4./(cellsize*cellsize);

        PoissonsRatio = 0.3;
        ComputeLame();
        Gravity = 9.81;
        Density = 980;
        IceFrictionCoefficient = 0;//0.03;

        IndDiameter = 0.324;
        IndRSq = IndDiameter*IndDiameter/4.;
        IndVelocity = 0.2;
        IndDepth = 0.101;

        BlockHeight = 1.0;
        BlockLength = 2.5;

        XiSnow = 10.;
        THT_C_snow = 2.0e-2;				// Critical compression
        THT_S_snow = 6.0e-3;				// Critical stretch

        SimulationStep = 0;
        SimulationTime = 0;
        MemAllocGrid = MemAllocPoints = MemAllocTotal = 0;
    }

    void ComputeLame()
    {
        lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
        mu = YoungsModulus/(2*(1+PoissonsRatio));
        kappa = mu*2./3. + lambda;
    }

    void ComputeCamClayParams()
    {
        constexpr int dim = 2;
        real sin_phi = std::sin(NACC_friction_angle / 180. * pi);
        real mohr_columb_friction = std::sqrt(2./3.)*2. * sin_phi / (3. - sin_phi);
        real NACC_M = mohr_columb_friction * (real)dim / std::sqrt(2. / (6. - dim));
        std::cout << "SimParams: NACC M is " << NACC_M << '\n';
        NACC_M_sq = NACC_M*NACC_M;
    }
};

#endif
