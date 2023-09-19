#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <Eigen/Core>


namespace icy { struct Point; }

struct icy::Point
{
    Eigen::Vector2f pos, velocity;
    Eigen::Matrix2f Bp, Fe, Fp; // refer to "The Material Point Method for Simulating Continuum Materials"

    static float wc(Eigen::Vector2f dx, double h);
    static Eigen::Vector2f gradwc(Eigen::Vector2f dx, double h);

private:
    static float wcs(float x);   // cubic spline
    static float dwcs(float x);  // cubic spline derivative
};




#endif // PARTICLE_H
