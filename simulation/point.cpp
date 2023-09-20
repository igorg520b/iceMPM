#include "point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>


Eigen::Matrix2f icy::Point::ElasticConstitutiveModel(const float &prmsMu,
                                      const float &prmsLambda,
                                      const float &particle_volume) const
{
    // elastic material (same for snow)
    Eigen::Matrix2f Re = polar_decomp_R(Fe);
    float Je = Fe.determinant();
    Eigen::Matrix2f dFe = 2.f * prmsMu*(Fe - Re)* Fe.transpose() +
            prmsLambda * (Je - 1.f) * Je * Eigen::Matrix2f::Identity();
    Eigen::Matrix2f Ap = dFe * particle_volume;
    return Ap;
}

void icy::Point::ElasticUpdateDeformationGradient(const float &dt,
                                   const Eigen::Matrix2f &FModifier)
{
    Fe = (Eigen::Matrix2f::Identity() + dt*FModifier) * Fe;
}



Eigen::Matrix2f icy::Point::SnowConstitutiveModel(const float &XiSnow,
                                  const float &prmsMu,
                                  const float &prmsLambda,
                                  const float &particle_volume)
{
    float exp = std::exp(XiSnow*(1.f - Fp.determinant()));
    visualized_value = Fp.determinant();
    const float mu = prmsMu * exp;
    const float lambda = prmsLambda * exp;
    Eigen::Matrix2f Re = polar_decomp_R(Fe);
    float Je = Fe.determinant();
    Eigen::Matrix2f dFe = 2.f * mu*(Fe - Re)* Fe.transpose() +
            lambda * (Je - 1.f) * Je * Eigen::Matrix2f::Identity();
    Eigen::Matrix2f Ap = dFe * particle_volume;
    return Ap;
}

void icy::Point::SnowUpdateDeformationGradient(const float &dt,
                                               const float &THT_C_snow,
                                               const float &THT_S_snow,
                                               const Eigen::Matrix2f &FModifier)
{
    Eigen::Matrix2f FeTr = (Eigen::Matrix2f::Identity() + dt * FModifier) * this->Fe;

    Eigen::JacobiSVD<Eigen::Matrix2f,Eigen::NoQRPreconditioner> svd(FeTr,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2f U = svd.matrixU();
    Eigen::Matrix2f V = svd.matrixV();
    Eigen::Vector2f Sigma = svd.singularValues();

    Eigen::Vector2f SigmaClamped;
    for(int k=0;k<2;k++) SigmaClamped[k] = std::clamp(Sigma[k], 1.f - THT_C_snow,1.f + THT_S_snow);

    Fe = U*SigmaClamped.asDiagonal()*V.transpose();
    Fp = V*SigmaClamped.asDiagonal().inverse()*Sigma.asDiagonal()*V.transpose()*Fp;
}




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

Eigen::Matrix2f icy::Point::polar_decomp_R(const Eigen::Matrix2f &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    float th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Eigen::Matrix2f result;
    result << cos(th), -sin(th), sin(th), cos(th);
    return result;
}


float icy::Point::wqs(float x)
{
    x = std::abs(x);
    if (x < 0.5f) return -x * x + 3 / 4.0f;
    else if (x < 1.5f) return x * x / 2.0f - 3 * x / 2.0f + 9 / 8.0f;
    return 0;
}

float icy::Point::dwqs(float x)
{
    float x_abs = std::abs(x);
    if (x_abs < 0.5f) return -2.0f * x;
    else if (x_abs < 1.5f) return x - 3 / 2.0f * x / x_abs;
    return 0;
}

float icy::Point::wq(Eigen::Vector2f dx, double h)
{
    return wqs(dx[0]/h)*wqs(dx[1]/h);
}

Eigen::Vector2f icy::Point::gradwq(Eigen::Vector2f dx, double h)
{
    Eigen::Vector2f result;
    result[0] = dwqs(dx[0]/h)*wqs(dx[1]/h)/h;
    result[1] = wqs(dx[0]/h)*dwqs(dx[1]/h)/h;
    return result;
}
