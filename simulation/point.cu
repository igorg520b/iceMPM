#include "point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>


void icy::Point::Reset()
{
    pos = pos_initial;
    Fe.setIdentity();
    velocity.setZero();
    Bp.setZero();
    q = 0;
    Jp_inv = 1;
    zeta = 1;
    case_when_Jp_first_changes = 0;
}



/*


real icy::Point::wcs(real x)
{
    x = abs(x);
    if(x < 1) return x*x*x/2. - x*x + 2./3.;
    else if(x < 2) return (2-x)*(2-x)*(2-x)/6.;
    else return 0;
}

real icy::Point::dwcs(real x)
{
    real xabs = abs(x);
    if(xabs<1) return 1.5*x*xabs - 2.*x;
    else if(xabs<2) return -.5*x*xabs + 2*x -2*x/xabs;
    else return 0;
}

real icy::Point::wc(Vector2r dx)
{
    return wcs(dx[0])*wcs(dx[1]);
}

Vector2r icy::Point::gradwc(Vector2r dx)
{
    Vector2r result;
    result[0] = dwcs(dx[0])*wcs(dx[1]);
    result[1] = wcs(dx[0])*dwcs(dx[1]);
    return result;
}

Matrix2r icy::Point::polar_decomp_R(const Matrix2r &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    real th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Matrix2r result;
    result << cos(th), -sin(th), sin(th), cos(th);
    return result;
}


real icy::Point::wqs(real x)
{
    x = std::abs(x);
    if (x < .5) return -x * x + 3 / 4.;
    else if (x < 1.5) return x * x / 2. - 3. * x / 2. + 9. / 8.;
    return 0;
}

real icy::Point::dwqs(real x)
{
    real x_abs = std::abs(x);
    if (x_abs < .5) return -2. * x;
    else if (x_abs < 1.5) return x - 3 / 2.0 * x / x_abs;
    return 0;
}

real icy::Point::wq(Vector2r dx)
{
    return wqs(dx[0])*wqs(dx[1]);
}

Vector2r icy::Point::gradwq(Vector2r dx)
{
    Vector2r result;
    result[0] = dwqs(dx[0])*wqs(dx[1]);
    result[1] = wqs(dx[0])*dwqs(dx[1]);
    return result;
}
*/
