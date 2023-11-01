#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <Eigen/Core>

#include "parameters_sim.h"

namespace icy { struct Point; }

struct icy::Point
{
    Vector2r pos, velocity;
    Matrix2r Bp, Fe; // refer to "The Material Point Method for Simulating Continuum Materials"
    real NACC_alpha_p;

    double q; // Drucker Prager hardening paremeter


    static real wc(Vector2r dx);
    static Vector2r gradwc(Vector2r dx);
    static real wq(Vector2r dx);
    static Vector2r gradwq(Vector2r dx);
/*
    Eigen::Matrix2f SnowConstitutiveModel(const float &XiSnow,
                                          const float &prmsMu,
                                          const float &prmsLambda,
                                          const float &particle_volume);
    void SnowUpdateDeformationGradient(const float &dt,
                                       const float &THT_C_snow,
                                       const float &THT_S_snow,
                                       const Eigen::Matrix2f &FModifier);
*/

    Matrix2r NACCConstitutiveModel(const real &prmsMu,
                                          const real &prmsLambda,
                                          const real &particle_volume) const;

    void NACCUpdateDeformationGradient(const real &dt,
                                       const Matrix2r &FModifier,
                                       const icy::SimParams &prms);


    Matrix2r ElasticConstitutiveModel(const real &prmsMu,
                                          const real &prmsLambda,
                                          const real &particle_volume) const;
    void ElasticUpdateDeformationGradient(const real &dt, const Matrix2r &FModifier);

    static Matrix2r polar_decomp_R(const Matrix2r &val);
private:
    static real wcs(real x);   // cubic spline
    static real dwcs(real x);  // cubic spline derivative
    static real wqs(real x);   // cubic spline
    static real dwqs(real x);  // cubic spline derivative
};


#endif // PARTICLE_H
