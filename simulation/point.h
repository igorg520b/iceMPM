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

    real Jp_inv; // track the change in det(Fp)
    short grain;
    char q;

    void Reset();
    void TransferToBuffer(real *buffer, const int pitch, const int point_index) const;  // distribute to SOA
    static Vector2r getPos(const real *buffer, const int pitch, const int point_index);
    static char getQ(const real *buffer, const int pitch, const int point_index);

    static double getJp_inv(const real *buffer, const int pitch, const int point_index);
    static short getGrain(const real *buffer, const int pitch, const int point_index);

    /*    void PullFromBuffer(const real *buffer, const int pitch, const int point_index);

    static Vector2r getVelocity(const real *buffer, const int pitch, const int point_index);
    static void setPos_Q_Jpinv(Eigen::Vector2f _pos, float _Jp_inv, real *buff, const int pitch, const int pt_idx);
*/
};


#endif // PARTICLE_H
