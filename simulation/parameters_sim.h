#ifndef P_SIM_H
#define P_SIM_H


// variables related to the formulation of the model

namespace icy { struct SimParams; }

struct icy::SimParams
{
public:
    SimParams() { Reset(); }

    constexpr static double pi = 3.14159265358979323846;
    constexpr static int dim = 2;

    float InitialTimeStep, SimulationEndTime;
    float Gravity, Density, PoissonsRatio, YoungsModulus;
    float lambda, mu; // Lame
    float kappa; // bulk modulus
    float IceFrictionCoefficient;

    float XiSnow, THT_C_snow, THT_S_snow;   // hardening, critical compression, critical stretch
    float NACC_xi, NACC_alpha, NACC_beta, NACC_M_sq;
    float NACC_friction_angle;
    bool NACC_hardening;

    int GridX, GridY;
    float cellsize;

    int UpdateEveryNthStep;

    float IndDiameter, IndVelocity, IndDepth;
    int PointsWanted, PointCountActual;
    float BlockHeight, BlockLength;

    float ParticleVolume, ParticleMass, ParticleViewSize;

//#define PARAMS2
    void Reset()
    {
#ifdef PARAMS2
        InitialTimeStep = 5e-5;
        YoungsModulus = 5.e7;
        PointsWanted = 400'000;
        GridX = 256;
        GridY = 100;
        ParticleViewSize = 1.5f;
#else
        InitialTimeStep = 2.5e-4;
        YoungsModulus = 1.e7;
        PointsWanted = 10'000;
        GridX = 128;
        GridY = 50;
        ParticleViewSize = 4.4f;
#endif

        NACC_beta = .8;
        NACC_xi = 3;
        NACC_alpha = std::log(0.999);
        NACC_hardening = true;

        NACC_friction_angle = 45;
        ComputeCamClayParams();

        SimulationEndTime = 15;
        UpdateEveryNthStep = (int)(1.f/(200*InitialTimeStep));

        cellsize = 4./(GridX);  // this better have a form of 2^n, where n is an integer

        PoissonsRatio = 0.3;
        ComputeLame();
        Gravity = 9.81;
        Density = 980;
        IceFrictionCoefficient = 0.03;

        IndDiameter = 0.324;
        IndVelocity = 0.2;
        IndDepth = 0.101;

        BlockHeight = 1.0f;
        BlockLength = 2.5f;

        XiSnow = 10.f;
        THT_C_snow = 2.0e-2;				// Critical compression
        THT_S_snow = 6.0e-3;				// Critical stretch
    }

    void ComputeLame()
    {
        lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
        mu = YoungsModulus/(2*(1+PoissonsRatio));

        kappa = mu*2.f/3 + lambda;
    }

    void ComputeCamClayParams()
    {
        float sin_phi = std::sin(NACC_friction_angle / 180.f * pi);
        float mohr_columb_friction = std::sqrt(2.f / 3.f) * 2.f * sin_phi / (3.f - sin_phi);
        float NACC_M = mohr_columb_friction * (float)dim / std::sqrt(2.f / (6.f - dim));
        NACC_M_sq = NACC_M*NACC_M;
    }
};

#endif
