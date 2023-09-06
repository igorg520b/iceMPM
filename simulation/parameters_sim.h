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

public:
    SimParams() { Reset(); }

    float InitialTimeStep;
    float Gravity, Density, PoissonsRatio, YoungsModulus;
    float lambda, mu; // Lame

    int GridX, GridY;
    float cellsize;



    void Reset()
    {
        InitialTimeStep = 1e-4;
        Gravity = 9.81;
        Density = 980;
        PoissonsRatio = 0.3;
        YoungsModulus = 10e4;

        GridX = GridY = 64;
        cellsize = 3./GridX;    // 3-meter space in horizotal direction

        ComputeLame();
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
