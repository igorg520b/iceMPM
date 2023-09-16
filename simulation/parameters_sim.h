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

    Q_PROPERTY(float in_InitialTimeStep MEMBER InitialTimeStep NOTIFY propertyChanged)
    Q_PROPERTY(float p_Gravity MEMBER Gravity NOTIFY propertyChanged)
    Q_PROPERTY(float p_Density MEMBER Density NOTIFY propertyChanged)
    Q_PROPERTY(float p_YoungsModulus MEMBER YoungsModulus NOTIFY propertyChanged)
    Q_PROPERTY(float p_PoissonsRatio MEMBER PoissonsRatio NOTIFY propertyChanged)

    // indenter
    Q_PROPERTY(float IndDiameter MEMBER IndDiameter NOTIFY propertyChanged)
    Q_PROPERTY(float IndVelocity MEMBER IndVelocity NOTIFY propertyChanged)
    Q_PROPERTY(float IndDepth MEMBER IndDepth NOTIFY propertyChanged)


public:
    SimParams() { Reset(); }

    float InitialTimeStep;
    float Gravity, Density, PoissonsRatio, YoungsModulus;
    float lambda, mu; // Lame

    int GridX, GridY;
    float cellsize;

    int UpdateEveryNthStep;

    float IndDiameter, IndVelocity, IndDepth;

    void Reset()
    {
        InitialTimeStep = 5e-4;
        Gravity = 9.81;
        Density = 980;
        PoissonsRatio = 0.3;
        YoungsModulus = 10e5;

        GridX = 64;
        GridY = 32;
        cellsize = 3./GridX;
        ComputeLame();

        UpdateEveryNthStep = 10;

        IndDiameter = 0.324;
        IndVelocity = 0.2;
        IndDepth = 0.101;
    }

    void ComputeLame()
    {
        lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
        mu = YoungsModulus/(2*(1+PoissonsRatio));
    }


Q_SIGNALS:
    void propertyChanged();
};


#endif
