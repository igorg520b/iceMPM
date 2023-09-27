#include "point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>



Matrix2r icy::Point::NACCConstitutiveModel(const real &mu,
                                      const real &lambda,
                                      const real &particle_volume) const
{
    // elastic material (same for snow)
    Matrix2r Re = icy::Point::polar_decomp_R(Fe);
    real Je = Fe.determinant();
    Matrix2r dFe = 2.f * mu*(Fe - Re)* Fe.transpose() +
            lambda * (Je - 1.f) * Je * Matrix2r::Identity();
    Matrix2r Ap = dFe * particle_volume;
    return Ap;
}


void icy::Point::NACCUpdateDeformationGradient(const real &dt,
                                    const Matrix2r &FModifier,
                                    const icy::SimParams &prms)
{
    constexpr real magic_epsilon = 1.e-5;
    constexpr int d = 2; // dimensions
    const real &mu = prms.mu;
    const real &kappa = prms.kappa; // bulk modulus
    const real &xi = prms.NACC_xi;
    const real &beta = prms.NACC_beta;
    const real &M_sq = prms.NACC_M_sq;
    real &alpha = NACC_alpha_p;

    Matrix2r FeTr = (Matrix2r::Identity() + dt * FModifier) * this->Fe;

    Eigen::JacobiSVD<Matrix2r,Eigen::NoQRPreconditioner> svd(FeTr,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix2r U = svd.matrixU();
    Matrix2r V = svd.matrixV();
    Vector2r Sigma = svd.singularValues();

    // line 4
    real p0 = kappa * (magic_epsilon + std::sinh(xi * std::max(-alpha, 0.)));

    // line 5
    real Je_tr = Sigma[0]*Sigma[1];    // this is for 2D

    // line 6
    Matrix2r SigmaMatrix = Sigma.asDiagonal();
    Matrix2r SigmaSquared = SigmaMatrix*SigmaMatrix;
    Matrix2r SigmaSquaredDev = SigmaSquared - SigmaSquared.trace()/2.*Matrix2r::Identity();
    real J_power_neg_2_d_mulmu = mu * std::pow(Je_tr, -2./(real)d);///< J^(-2/dim) * mu
    Matrix2r s_hat_tr = J_power_neg_2_d_mulmu * SigmaSquaredDev;

    // line 7
    real psi_kappa_partial_J = (kappa/2.) * (Je_tr - 1./Je_tr);

    // line 8
    real p_trial = -psi_kappa_partial_J * Je_tr;

    // line 9 (case 1)
    real y = (1. + 2.*beta)*(3.-(real)d/2.)*s_hat_tr.norm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(p_trial > p0)
    {
        real Je_new = std::sqrt(-2.*p0 / kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        Fe = U*Sigma_new*V.transpose();
        if(true) alpha += std::log(Je_tr / Je_new);
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        real Je_new = std::sqrt(2.*beta*p0/kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        Fe = U*Sigma_new*V.transpose();
        if(true) alpha += std::log(Je_tr / Je_new);
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon*10)
    {
        if(true && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
        {
            real p_c = (1.-beta)*p0/2.;  // line 23
            real q_tr = sqrt(3.-d/2.)*s_hat_tr.norm();   // line 24
            Vector2r direction(p_c-p_trial, -q_tr);  // line 25
            direction.normalize();
            real C = M_sq*(p_c-beta*p0)*(p_c-p0);
            real B = M_sq*direction[0]*(2.*p_c-p0+beta*p0);
            real A = M_sq*direction[0]*direction[0]+(1.+2.*beta)*direction[1]*direction[1];  // line 30
            real l1 = (-B+sqrt(B*B-4.*A*C))/(2.*A);
            real l2 = (-B-sqrt(B*B-4.*A*C))/(2.*A);
            real p1 = p_c + l1*direction[0];
            real p2 = p_c + l2*direction[0];
            real p_x = (p_trial-p_c)*(p1-p_c) > 0 ? p1 : p2;
            real Je_x = sqrt(abs(-2.*p_x/kappa + 1.));
            if(Je_x > magic_epsilon*10) alpha += std::log(Je_tr / Je_x);
        }

        real expr_under_root = (-M_sq*(p_trial+beta*p0)*(p_trial-p0))/((1+2.*beta)*(3.-d/2.));
        Matrix2r B_hat_E_new = sqrt(expr_under_root)*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() +
                Matrix2r::Identity()*SigmaSquared.trace()/(real)d;
        Matrix2r Sigma_new;
        Sigma_new << sqrt(B_hat_E_new(0,0)), 0,
                0, sqrt(B_hat_E_new(1,1));
        Fe = U*Sigma_new*V.transpose();
    }
    else
    {
        Fe = FeTr;
    }
}




Matrix2r icy::Point::ElasticConstitutiveModel(const real &prmsMu,
                                      const real &prmsLambda,
                                      const real &particle_volume) const
{
    // elastic material (same for snow)
    Matrix2r Re = icy::Point::polar_decomp_R(Fe);
    real Je = Fe.determinant();
    Matrix2r dFe = 2. * prmsMu*(Fe - Re)* Fe.transpose() +
            prmsLambda * (Je - 1.) * Je * Matrix2r::Identity();

    Matrix2r Ap = dFe * particle_volume;

    return Ap;
}

void icy::Point::ElasticUpdateDeformationGradient(const real &dt,
                                   const Matrix2r &FModifier)
{
    Fe = (Matrix2r::Identity() + dt*FModifier) * Fe;
}


/*
Matrix2r icy::Point::SnowConstitutiveModel(const real &XiSnow,
                                  const real &prmsMu,
                                  const real &prmsLambda,
                                  const real &particle_volume)
{
    real exp = std::exp(XiSnow*(1.f - Fp.determinant()));
    visualized_value = Fp.determinant();
    const real mu = prmsMu * exp;
    const real lambda = prmsLambda * exp;
    Matrix2r Re = icy::Point::polar_decomp_R(Fe);
    real Je = Fe.determinant();
    Matrix2r dFe = 2.f * mu*(Fe - Re)* Fe.transpose() +
            lambda * (Je - 1.f) * Je * Matrix2r::Identity();
    Matrix2r Ap = dFe * particle_volume;
}

void icy::Point::SnowUpdateDeformationGradient(const real &dt,
                                               const real &THT_C_snow,
                                               const real &THT_S_snow,
                                               const Matrix2r &FModifier)
{
    Matrix2r FeTr = (Matrix2r::Identity() + dt * FModifier) * this->Fe;

    Eigen::JacobiSVD<Matrix2r,Eigen::NoQRPreconditioner> svd(FeTr,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix2r U = svd.matrixU();
    Matrix2r V = svd.matrixV();
    Vector2r Sigma = svd.singularValues();

    Vector2r SigmaClamped;
    for(int k=0;k<2;k++) SigmaClamped[k] = std::clamp(Sigma[k], 1.f - THT_C_snow,1.f + THT_S_snow);

    Fe = U*SigmaClamped.asDiagonal()*V.transpose();
    Fp = V*SigmaClamped.asDiagonal().inverse()*Sigma.asDiagonal()*V.transpose()*Fp;
}

*/


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
