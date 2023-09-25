#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <Eigen/Core>

#include "parameters_sim.h"

namespace icy { struct Point; }

struct icy::Point
{
    Eigen::Vector2f pos, velocity;
    Eigen::Matrix2f Bp, Fe; // refer to "The Material Point Method for Simulating Continuum Materials"
//    Eigen::Matrix2f Fp; // for snow
//    float visualized_value;
    float NACC_alpha_p;

    static float wc(Eigen::Vector2f dx, double h);
    static Eigen::Vector2f gradwc(Eigen::Vector2f dx, double h);
    static float wq(Eigen::Vector2f dx);
    static Eigen::Vector2f gradwq(Eigen::Vector2f dx);
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

    Eigen::Matrix2f NACCConstitutiveModel(const float &prmsMu,
                                          const float &prmsLambda,
                                          const float &particle_volume) const;
    void NACCUpdateDeformationGradient(const float &dt,
                                       const Eigen::Matrix2f &FModifier,
                                       const icy::SimParams &prms);


    Eigen::Matrix2f ElasticConstitutiveModel(const float &prmsMu,
                                          const float &prmsLambda,
                                          const float &particle_volume) const;
    void ElasticUpdateDeformationGradient(const float &dt,
                                       const Eigen::Matrix2f &FModifier);

private:
    static float wcs(float x);   // cubic spline
    static float dwcs(float x);  // cubic spline derivative
    static float wqs(float x);   // cubic spline
    static float dwqs(float x);  // cubic spline derivative
    static Eigen::Matrix2f polar_decomp_R(const Eigen::Matrix2f &val);

};




#endif // PARTICLE_H
