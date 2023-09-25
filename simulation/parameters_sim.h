#ifndef P_SIM_H
#define P_SIM_H

#include <iostream>

// variables related to the formulation of the model

namespace icy { struct SimParams; }

struct icy::SimParams
{
public:
    constexpr static double pi = 3.14159265358979323846;


    float InitialTimeStep, SimulationEndTime;
    float Gravity, Density, PoissonsRatio, YoungsModulus;
    float lambda, mu; // Lame
    float kappa; // bulk modulus
    float IceFrictionCoefficient;

    float XiSnow, THT_C_snow, THT_S_snow;   // hardening, critical compression, critical stretch
    float NACC_xi, NACC_alpha, NACC_beta, NACC_M_sq;
    float NACC_friction_angle;

    int GridX, GridY;
    float cellsize, Dp_inv;

    int UpdateEveryNthStep;

    float IndDiameter, IndRSq, IndVelocity, IndDepth;
    int PointsWanted, PointCountActual;
    float BlockHeight, BlockLength;

    float ParticleVolume, ParticleMass, ParticleViewSize;

    int SimulationStep;
    float SimulationTime;
    float MemAllocGrid, MemAllocPoints, MemAllocTotal;

    bool useGPU;
    void Reset()
    {
//#define PARAMS2
#ifdef PARAMS2
        InitialTimeStep = 5.e-6;
        YoungsModulus = 5.e8;
        NACC_beta = 0.4;
        NACC_xi = 0.9;
        NACC_alpha = std::log(1.-1.e-6);
        PointsWanted = 1'000'000;
        GridX = 512;
        GridY = 256;
        ParticleViewSize = 1.1f;
#else
        InitialTimeStep = 1.e-5;
        YoungsModulus = 1.e9;
        NACC_beta = 0.3;
        NACC_xi = 0.9;
        NACC_alpha = std::log(1.-1.e-8);
        PointsWanted = 25'000;
        GridX = 128;
        GridY = 64;
        ParticleViewSize = 5.5f;
#endif

        NACC_friction_angle = 45;//60;
        ComputeCamClayParams();

        useGPU = true;

        SimulationEndTime = 15;
        UpdateEveryNthStep = (int)(1.f/(200*InitialTimeStep));

        cellsize = 3.33/(GridX);  // this better have a form of 2^n, where n is an integer
        Dp_inv = 4.f/(cellsize*cellsize);

        PoissonsRatio = 0.3;
        ComputeLame();
        Gravity = 9.81;
        Density = 980;
        IceFrictionCoefficient = 0;//0.03;

        IndDiameter = 0.324;
        IndRSq = IndDiameter*IndDiameter/4.f;
        IndVelocity = 0.2;
        IndDepth = 0.101;

        BlockHeight = 1.0f;
        BlockLength = 2.5f;

        XiSnow = 10.f;
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

        kappa = mu*2.f/3 + lambda;
    }

    void ComputeCamClayParams()
    {
        constexpr int dim = 2;
        float sin_phi = std::sin(NACC_friction_angle / 180.f * pi);
        float mohr_columb_friction = std::sqrt(2.f / 3.f) * 2.f * sin_phi / (3.f - sin_phi);
        float NACC_M = mohr_columb_friction * (float)dim / std::sqrt(2.f / (6.f - dim));
        std::cout << "SimParams: NACC M is " << NACC_M << '\n';
        NACC_M_sq = NACC_M*NACC_M;
    }
};

#endif
