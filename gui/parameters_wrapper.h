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

    Q_PROPERTY(QString in_TimeStep_ READ getTimeStep_ NOTIFY propertyChanged)
    QString getTimeStep_() {return QString("%1 s").arg(prms->InitialTimeStep,0,'e',1);}

    Q_PROPERTY(double in_SimulationTime READ getSimulationTime WRITE setSimulationTime NOTIFY propertyChanged)
    double getSimulationTime() {return prms->SimulationEndTime;}
    void setSimulationTime(double val) { prms->SimulationEndTime = val; }

    Q_PROPERTY(int in_UpdateEvery READ getUpdateEveryNthStep NOTIFY propertyChanged)
    int getUpdateEveryNthStep() {return prms->UpdateEveryNthStep;}

    Q_PROPERTY(double p_YoungsModulus READ getYoungsModulus WRITE setYoungsModulus NOTIFY propertyChanged)
    double getYoungsModulus() {return prms->YoungsModulus;}
    void setYoungsModulus(double val) { prms->YoungsModulus = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(QString p_YM READ getYM NOTIFY propertyChanged)
    QString getYM() {return QString("%1 Pa").arg(prms->YoungsModulus, 0, 'e', 2);}

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
    int getPointCountActual() {return prms->nPts;}

    Q_PROPERTY(QString b_Grid READ getGridDimensions NOTIFY propertyChanged)
    QString getGridDimensions() {return QString("%1 x %2").arg(prms->GridX).arg(prms->GridY);}

    Q_PROPERTY(double nacc_beta READ getNaccBeta WRITE setNaccBeta NOTIFY propertyChanged)
    double getNaccBeta() {return prms->NACC_beta;}
    void setNaccBeta(double val) {prms->NACC_beta = val;}

    Q_PROPERTY(double nacc_xi READ getNaccXi WRITE setNaccXi NOTIFY propertyChanged)
    double getNaccXi() {return prms->NACC_xi;}
    void setNaccXi(double val) {prms->NACC_xi = val;}

    Q_PROPERTY(double nacc_a_exp READ getNaccAlphaExp WRITE setNaccAlphaExp NOTIFY propertyChanged)
    double getNaccAlphaExp() {return exp(prms->NACC_alpha);}
    void setNaccAlphaExp(double val) {prms->NACC_alpha = log(val);}

    Q_PROPERTY(double nacc_p0 READ getNaccP0 NOTIFY propertyChanged)
    double getNaccP0() { return prms->kappa*sinh(prms->NACC_xi*(-prms->NACC_alpha))/1e6;}

    Q_PROPERTY(double nacc_angle READ getNaccAngle WRITE setNaccAngle NOTIFY propertyChanged)
    double getNaccAngle() {return prms->NACC_friction_angle;}
    void setNaccAngle(double val) {prms->NACC_friction_angle=val; prms->ComputeCamClayParams();}



    Q_PROPERTY(double nacc_p0_beta READ getNaccP0beta NOTIFY propertyChanged)
    double getNaccP0beta() { return prms->kappa*sinh(prms->NACC_xi*(-prms->NACC_alpha))*prms->NACC_beta/1e6;}


    Q_PROPERTY(double sand_H0 READ getH0 WRITE setH0 NOTIFY propertyChanged)
    double getH0() {return prms->H0 * 180 / icy::SimParams::pi;}
    void setH0(double val) {prms->H0 = val * icy::SimParams::pi / 180;}

    Q_PROPERTY(double sand_H1 READ getH1 WRITE setH1 NOTIFY propertyChanged)
    double getH1() {return prms->H1 * 180 / icy::SimParams::pi;}
    void setH1(double val) {prms->H1 = val * icy::SimParams::pi / 180;}

    Q_PROPERTY(double sand_H2 READ getH2 WRITE setH2 NOTIFY propertyChanged)
    double getH2() {return prms->H2;}
    void setH2(double val) {prms->H2 = val;}

    Q_PROPERTY(double sand_H3 READ getH3 WRITE setH3 NOTIFY propertyChanged)
    double getH3() {return prms->H3 * 180 / icy::SimParams::pi;}
    void setH3(double val) {prms->H3 = val * icy::SimParams::pi / 180;}


    Q_PROPERTY(double snow_Xi READ getSnowXi WRITE setSnowXi NOTIFY propertyChanged)
    double getSnowXi() {return prms->XiSnow;}
    void setSnowXi(double val) {prms->XiSnow = val;}

    Q_PROPERTY(double snow_ThC READ getSnowThC WRITE setSnowThC NOTIFY propertyChanged)
    double getSnowThC() {return prms->THT_C_snow;}
    void setSnowThC(double val) {prms->THT_C_snow = val;}

    Q_PROPERTY(double snow_ThS READ getSnowThS WRITE setSnowThS NOTIFY propertyChanged)
    double getSnowThS() {return prms->THT_S_snow;}
    void setSnowThS(double val) {prms->THT_S_snow = val;}

public:
    ParamsWrapper(icy::SimParams *p)
    {
        this->prms = p;
        Reset();
    }

    void Reset()
    {
        // it is possible to change parameters here
    }


Q_SIGNALS:
    void propertyChanged();
};



#endif // PARAMETERS_WRAPPER_H
