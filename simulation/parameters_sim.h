#ifndef P_SIM_H
#define P_SIM_H

#include <QObject>
#include <QDebug>

#include <Eigen/Core>
#include <iostream>
#include <cmath>

// variables related to the formulation of the model

namespace icy { class SimParams; }

class icy::SimParams : public QObject
{
    Q_OBJECT

    Q_PROPERTY(double in_InitialTimeStep MEMBER InitialTimeStep NOTIFY propertyChanged)
    Q_PROPERTY(double in_SimulationTime MEMBER SimulationEndTime NOTIFY propertyChanged)
    Q_PROPERTY(int in_UpdateEvery READ getUpdateEveryNthStep NOTIFY propertyChanged)

    Q_PROPERTY(double p_Gravity MEMBER Gravity NOTIFY propertyChanged)
    Q_PROPERTY(double p_Density MEMBER Density NOTIFY propertyChanged)
    Q_PROPERTY(double p_YoungsModulus READ getYoungsModulus WRITE setYoungsModulus NOTIFY propertyChanged)
    Q_PROPERTY(double p_PoissonsRatio READ getPoissonsRatio WRITE setPoissonsRatio NOTIFY propertyChanged)
    Q_PROPERTY(double p_LameLambda READ getLambda NOTIFY propertyChanged)
    Q_PROPERTY(double p_LameMu READ getMu NOTIFY propertyChanged)
    Q_PROPERTY(double p_FrictionCoeff MEMBER IceFrictionCoefficient NOTIFY propertyChanged)
    Q_PROPERTY(double p_ParticleVolume READ getParticleVolume NOTIFY propertyChanged)
    Q_PROPERTY(double p_ParticleMass READ getParticleMass NOTIFY propertyChanged)
    Q_PROPERTY(double p_ParticleViewSize MEMBER ParticleViewSize NOTIFY propertyChanged)

    //Q_PROPERTY(double in_HHTalpha READ getHHTalpha WRITE setHHTalpha)

    // indenter
    Q_PROPERTY(double IndDiameter MEMBER IndDiameter NOTIFY propertyChanged)
    Q_PROPERTY(double IndVelocity MEMBER IndVelocity NOTIFY propertyChanged)
    Q_PROPERTY(double IndDepth MEMBER IndDepth NOTIFY propertyChanged)

    // ice block
    Q_PROPERTY(int b_PtWanted MEMBER PointsWanted NOTIFY propertyChanged)
    Q_PROPERTY(int b_PtActual READ getPointCountActual NOTIFY propertyChanged)


public:
    SimParams() { Reset(); }

    float InitialTimeStep, SimulationEndTime;
    float Gravity, Density, PoissonsRatio, YoungsModulus;
    float lambda, mu; // Lame
    float IceFrictionCoefficient;

    float XiSnow, THT_C_snow, THT_S_snow;   // hardening, critical compression, critical stretch
    float NACC_xi, NACC_alpha, NACC_beta, NACC_M;
    float NACC_friction_angle;
    bool NACC_hardening;

    int GridX, GridY;
    float cellsize;

    int UpdateEveryNthStep;

    float IndDiameter, IndVelocity, IndDepth;
    int PointsWanted, PointCountActual;
    float BlockHeight, BlockLength;

    float ParticleVolume, ParticleMass, ParticleViewSize;

    constexpr static double pi = 3.14159265358979323846;
    constexpr static int dim = 2;

#define PARAMS2
    void Reset()
    {
#ifdef PARAMS2
        InitialTimeStep = 5e-5;
        YoungsModulus = 5.e7;
        PointsWanted = 400'000;
        GridX = 256;
        GridY = 100;
        ParticleViewSize = 1.5f;
#elif
        InitialTimeStep = 2e-4;
        YoungsModulus = 1.e7;
        PointsWanted = 20'000;
        GridX = 128;
        GridY = 50;
        ParticleViewSize = 3.4f;
#endif

        SimulationEndTime = 15;
        UpdateEveryNthStep = (int)(1.f/(300*InitialTimeStep));

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

        NACC_beta = .8;
        NACC_xi = 3;
        NACC_alpha = std::log(0.999);
        NACC_hardening = true;

        NACC_friction_angle = 45;
        ComputeCamClayParams();
    }

    void ComputeLame()
    {
        lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
        mu = YoungsModulus/(2*(1+PoissonsRatio));
    }

    void ComputeCamClayParams()
    {
        float sin_phi = std::sin(NACC_friction_angle / 180.f * pi);
        float mohr_columb_friction = std::sqrt(2.f / 3.f) * 2.f * sin_phi / (3.f - sin_phi);
        NACC_M = mohr_columb_friction * (float)dim / std::sqrt(2.f / (6.f - dim));
    }

    double getLambda() {return lambda;}
    double getMu() {return mu;}
    double getYoungsModulus() {return YoungsModulus;}
    double getPoissonsRatio() {return PoissonsRatio;}
    void setYoungsModulus(double val) { YoungsModulus = (float)val; ComputeLame(); }
    void setPoissonsRatio(double val) { PoissonsRatio = (float)val; ComputeLame(); }
    int getPointCountActual() {return PointCountActual;}
    double getParticleVolume() {return ParticleVolume;}
    double getParticleMass() {return ParticleMass;}
    int getUpdateEveryNthStep() {return UpdateEveryNthStep;}

Q_SIGNALS:
    void propertyChanged();
};


#endif
