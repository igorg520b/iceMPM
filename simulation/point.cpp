#include "point.h"


float icy::Point::wcs(float x)
{
    x = abs(x);
    if(x < 1) return x*x*x/2.f - x*x + 2.f/3.f;
    else if(x < 2) return (2-x)*(2-x)*(2-x)/6.f;
    else return 0;
}

float icy::Point::dwcs(float x)
{
    float xabs = abs(x);
    if(xabs<1) return 1.5f*x*xabs - 2.f*x;
    else if(xabs<2) return -.5f*x*xabs + 2*x -2*x/xabs;
    else return 0;
}

float icy::Point::wc(Eigen::Vector2f dx, double h)
{
    return wcs(dx[0]/h)*wcs(dx[1]/h);
}

Eigen::Vector2f icy::Point::gradwc(Eigen::Vector2f dx, double h)
{
    Eigen::Vector2f result;
    result[0] = dwcs(dx[0]/h)*wcs(dx[1]/h)/h;
    result[1] = wcs(dx[0]/h)*dwcs(dx[1]/h)/h;
    return result;
}
