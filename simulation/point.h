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

    double q; // Drucker Prager hardening paremeter
    double Jp_inv; // track the change in det(Fp)
    double zeta; // track shear accumulation
    Vector2r pos_initial; // for resetting

    double visualize_p, visualize_q, visualize_p0, visualize_psi;

    void Reset();
};


#endif // PARTICLE_H
