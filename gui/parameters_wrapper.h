#ifndef PARAMETERS_WRAPPER_H
#define PARAMETERS_WRAPPER_H


#include <QObject>
#include <QString>
#include "simulation/parameters_sim.h"

// wrapper for SimParams to display/edit them in GUI
class ParamsWrapper : public QObject
{
    Q_OBJECT

    icy::SimParams *prms;

    Q_PROPERTY(double in_TimeStep READ getTimeStep WRITE setTimeStep NOTIFY propertyChanged)
    double getTimeStep() {return prms->InitialTimeStep;}
    void setTimeStep(double val) { prms->InitialTimeStep = val; }

    Q_PROPERTY(double in_SimulationTime READ getSimulationTime WRITE setSimulationTime NOTIFY propertyChanged)
    double getSimulationTime() {return prms->SimulationEndTime;}
    void setSimulationTime(double val) { prms->SimulationEndTime = val; }

    Q_PROPERTY(int in_UpdateEvery READ getUpdateEveryNthStep NOTIFY propertyChanged)
    int getUpdateEveryNthStep() {return prms->UpdateEveryNthStep;}



    Q_PROPERTY(double p_Gravity READ getGravity NOTIFY propertyChanged)
    double getGravity() {return prms->Gravity;}

    Q_PROPERTY(double p_Density READ getDensity NOTIFY propertyChanged)
    double getDensity() {return prms->Density;}

    Q_PROPERTY(double p_YoungsModulus READ getYoungsModulus WRITE setYoungsModulus NOTIFY propertyChanged)
    double getYoungsModulus() {return prms->YoungsModulus;}
    void setYoungsModulus(double val) { prms->YoungsModulus = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(QString p_YM READ getYM NOTIFY propertyChanged)
    QString getYM() {return QString("%1").arg(prms->YoungsModulus, 0, 'e', 2);}

    Q_PROPERTY(double p_PoissonsRatio READ getPoissonsRatio WRITE setPoissonsRatio NOTIFY propertyChanged)
    double getPoissonsRatio() {return prms->PoissonsRatio;}
    void setPoissonsRatio(double val) { prms->PoissonsRatio = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(double p_LameLambda READ getLambda NOTIFY propertyChanged)
    double getLambda() {return prms->lambda;}

    Q_PROPERTY(double p_LameMu READ getMu NOTIFY propertyChanged)
    double getMu() {return prms->mu;}

    Q_PROPERTY(double p_FrictionCoeff READ getIceFrictionCoefficient NOTIFY propertyChanged)
    double getIceFrictionCoefficient() {return prms->IceFrictionCoefficient;}

    Q_PROPERTY(double p_ParticleViewSize READ getParticleViewSize WRITE setParticleViewSize NOTIFY propertyChanged)
    double getParticleViewSize() {return prms->ParticleViewSize;}
    void setParticleViewSize(double val) {prms->ParticleViewSize=val;}


    // indenter
    Q_PROPERTY(double IndDiameter READ getIndDiameter NOTIFY propertyChanged)
    double getIndDiameter() {return prms->IndDiameter;}

    Q_PROPERTY(double IndVelocity READ getIndVelocity NOTIFY propertyChanged)
    double getIndVelocity() {return prms->IndVelocity;}

    Q_PROPERTY(double IndDepth READ getIndDepth NOTIFY propertyChanged)
    double getIndDepth() {return prms->IndDepth;}

    // ice block
    Q_PROPERTY(int b_PtActual READ getPointCountActual NOTIFY propertyChanged)
    int getPointCountActual() {return prms->PointCountActual;}


public:
    ParamsWrapper(icy::SimParams *p)
    {
        this->prms = p;
        Reset();
    }

//#define PARAMS2
    void Reset()
    {
#ifdef PARAMS2
        prms->InitialTimeStep = 5e-5;
        prms->YoungsModulus = 5.e7;
        prms->PointsWanted = 400'000;
        prms->GridX = 256;
        prms->GridY = 100;
        prms->ParticleViewSize = 1.5f;
#else
        prms->InitialTimeStep = 1.5e-4;
        prms->YoungsModulus = 1.e7;
        prms->PointsWanted = 5'000;
        prms->GridX = 128;
        prms->GridY = 50;
        prms->ParticleViewSize = 4.4f;
#endif

        prms->NACC_beta = .8;
        prms->NACC_xi = 3;
        prms->NACC_alpha = std::log(0.999);
        prms->NACC_hardening = true;
        prms->NACC_friction_angle = 45;

        prms->ComputeCamClayParams();
        prms->ComputeLame();
    }


Q_SIGNALS:
    void propertyChanged();
};



#endif // PARAMETERS_WRAPPER_H
