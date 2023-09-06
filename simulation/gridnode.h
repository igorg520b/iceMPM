#ifndef GRIDNODE_H
#define GRIDNODE_H

#include <Eigen/Core>

namespace icy {struct GridNode;}

struct icy::GridNode
{
    Eigen::Vector2f momentum, velocity, force;
    float mass;

    void Reset()
    {
        momentum.setZero();
        velocity.setZero();
        force.setZero();
        mass = 0;
    }

};


#endif // GRIDNODE_H
