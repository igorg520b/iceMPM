#ifndef GRIDNODE_H
#define GRIDNODE_H

#include <Eigen/Core>

namespace icy {struct GridNode;}

struct icy::GridNode
{
    Eigen::Vector2f velocity, force;
    float mass;

    void Reset()
    {
        velocity.setZero();
        force.setZero();
        mass = 0;
    }

};


#endif // GRIDNODE_H
