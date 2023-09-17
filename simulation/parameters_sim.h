#ifndef P_SIM_H
#define P_SIM_H

#include <QObject>
#include <QDebug>

#include <Eigen/Core>
#include <iostream>

// variables related to the formulation of the model

namespace icy { class SimParams; }

class icy::SimParams : public QObject
{
    Q_OBJECT

    Q_PROPERTY(double in_InitialTimeStep MEMBER InitialTimeStep NOTIFY propertyChanged)
    Q_PROPERTY(double p_Gravity MEMBER Gravity NOTIFY propertyChanged)
    Q_PROPERTY(double p_Density MEMBER Density NOTIFY propertyChanged)
    Q_PROPERTY(double p_YoungsModulus READ getYoungsModulus WRITE setYoungsModulus)
    Q_PROPERTY(double p_PoissonsRatio READ getPoissonsRatio WRITE setPoissonsRatio)
    Q_PROPERTY(double p_LameLambda READ getLambda)
    Q_PROPERTY(double p_LameMu READ getMu)
    Q_PROPERTY(double p_FrictionCoeff MEMBER IceFrictionCoefficient NOTIFY propertyChanged)


    //Q_PROPERTY(double in_HHTalpha READ getHHTalpha WRITE setHHTalpha)

    // indenter
    Q_PROPERTY(double IndDiameter MEMBER IndDiameter NOTIFY propertyChanged)
    Q_PROPERTY(double IndVelocity MEMBER IndVelocity NOTIFY propertyChanged)
    Q_PROPERTY(double IndDepth MEMBER IndDepth NOTIFY propertyChanged)

    // ice block
    Q_PROPERTY(int b_PtWanted MEMBER PointsWanted NOTIFY propertyChanged)
    Q_PROPERTY(int b_PtActual READ getPointCountActual)


public:
    SimParams() { Reset(); }

    float InitialTimeStep;
    float Gravity, Density, PoissonsRatio, YoungsModulus;
    float lambda, mu; // Lame
    float IceFrictionCoefficient;

    int GridX, GridY;
    float cellsize;

    int UpdateEveryNthStep;

    float IndDiameter, IndVelocity, IndDepth;
    int PointsWanted, PointCountActual;
    float BlockHeight, BlockLength;

    void Reset()
    {
        InitialTimeStep = 3e-4;
        Gravity = 9.81;
        Density = 980;
        PoissonsRatio = 0.3;
        YoungsModulus = 10e5;
        IceFrictionCoefficient = 0.03;

        GridX = 128; //64;
        GridY = 64;// 32;
        cellsize = 3./GridX;
        ComputeLame();

        UpdateEveryNthStep = 10;

        IndDiameter = 0.324;
        IndVelocity = 0.2;
        IndDepth = 0.101;

        PointsWanted = 15000;
        BlockHeight = 1.0f;
        BlockLength = 2.5f;
    }

    void ComputeLame()
    {
        lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
        mu = YoungsModulus/(2*(1+PoissonsRatio));
    }

    double getLambda() {return lambda;}
    double getMu() {return mu;}
    double getYoungsModulus() {return YoungsModulus;}
    double getPoissonsRatio() {return PoissonsRatio;}
    void setYoungsModulus(double val)
    {
        YoungsModulus = (float)val;
        ComputeLame();
        Q_EMIT propertyChanged();
    }
    void setPoissonsRatio(double val)
    {
        PoissonsRatio = (float)val;
        ComputeLame();
        Q_EMIT propertyChanged();
    }
    int getPointCountActual() {return PointCountActual;}


Q_SIGNALS:
    void propertyChanged();
};


#endif
