#ifndef GRIDNODE_H
#define GRIDNODE_H

#include <Eigen/Core>
#include "parameters_sim.h"

namespace icy {struct GridNode;}

struct icy::GridNode
{
    Vector2r velocity, force;
    real mass;

    void Reset()
    {
        velocity.setZero();
        force.setZero();
        mass = 0;
    }

};


#endif // GRIDNODE_H
