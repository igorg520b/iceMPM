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
    Q_PROPERTY(double p_YoungsModulus MEMBER YoungsModulus NOTIFY propertyChanged)
    Q_PROPERTY(double p_PoissonsRatio MEMBER PoissonsRatio NOTIFY propertyChanged)

public:
    SimParams() { Reset(); }

    double InitialTimeStep;
    double Gravity, Density, PoissonsRatio, YoungsModulus;

    int GridX, GridY;
    double cellsize;

    void Reset()
    {
        InitialTimeStep = 0.01; // 0.0005;
        Gravity = 9.81;
        Density = 980;
        PoissonsRatio = 0.3;
        YoungsModulus = 10e5;

        GridX = GridY = 64;
        cellsize = 3./GridX;    // 3-meter space in horizotal direction
    }


Q_SIGNALS:
    void propertyChanged();
};


#endif
