#ifndef PARTICLE_H
#define PARTICLE_H

#include <Eigen/Core>


namespace icy { struct Point; struct GridNode;}


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


struct icy::Point
{
    Eigen::Vector2f pos, velocity;
    float mass, volume;
    Eigen::Matrix2f Bp, Fe, Fp; // refer to "The Material Point Method for Simulating Continuum Materials"

    static float weight_spline_quadratic(float x)
    {
        x = abs(x);
        if(x<=0.5) return 0.75-x*x;
        if(x<1.5) return 0.5*(1.5-x)*(1.5-x);
        return 0;
    }

    static float weight_spline_quadratic_derivative(float x)
    {
        if(x>0)
        {
            if(x<=0.5) return -2*x;
            if(x<1.5) return x-1.5;
        }
        else
        {
            if(x>=-0.5) return -2*x;
            if(x>-1.5) return x+1.5;
        }
        return 0;
    }

    static float weight_spline_cubic(float x)
    {
        x = abs(x);
        if(x<=1.) return x*x*x/2. - x*x + 2./3.;
        if(x<2) return (2.-x)*(2.-x)*(2.-x)/6.;
        return 0;
    }

    static float weight_spline_cubic_derivative(float x)
    {
        if(x>=0)
        {
            if(x<1) return (-12*x+9*x*x)/6.;
            if(x<2) return -(x-2)*(x-2)/2.;
        }
        else
        {
            if(x>-1) return (-12*x-9*x*x)/6.;
            if(x>-2) return (x+2)*(x+2)/2.;
        }
        return 0;
    }

    static float weight_q(Eigen::Vector2f dx, double h)
    {
        return weight_spline_quadratic(dx[0]/h)*weight_spline_quadratic(dx[1]/h);
    }

    static Eigen::Vector2f weight_grad_q(Eigen::Vector2f dx, double h)
    {
        Eigen::Vector2f result;
        result[0] = weight_spline_quadratic_derivative(dx[0]/h)*weight_spline_quadratic(dx[1]/h)/h;
        result[1] = weight_spline_quadratic(dx[0]/h)*weight_spline_quadratic_derivative(dx[1]/h)/h;
        return result;
    }

};




#endif // PARTICLE_H
