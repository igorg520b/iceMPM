#ifndef PARTICLE_H
#define PARTICLE_H

#include <cmath>
#include <Eigen/Core>

#include "parameters_sim.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace icy { struct Point; }

struct icy::Point
{
    Vector2r pos, velocity;
    Matrix2r Bp, Fe; // refer to "The Material Point Method for Simulating Continuum Materials"

    real q;
    real Jp_inv; // track the change in det(Fp)
    real zeta; // track shear accumulation
    Vector2r pos_initial; // for resetting

    real visualize_p, visualize_q, visualize_p0, visualize_psi;
    real case_when_Jp_first_changes;

    void Reset();
};


#endif // PARTICLE_H
