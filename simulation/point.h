#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <Eigen/Core>


namespace icy { struct Point; }





struct icy::Point
{
    float mass, volume;
    Eigen::Vector2f pos, velocity;
    Eigen::Matrix2f Bp, Fe; // refer to "The Material Point Method for Simulating Continuum Materials"

//    Eigen::Matrix2f Ap; // temporary variable

    static float wcs(float x);   // cubic spline
    static float dwcs(float x);  // cubic spline derivative
    static float wc(Eigen::Vector2f dx, double h);
    static Eigen::Vector2f gradwc(Eigen::Vector2f dx, double h);



};




#endif // PARTICLE_H
