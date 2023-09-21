#include "point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>



Eigen::Matrix2f icy::Point::NACCConstitutiveModel(const float &prmsMu,
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


void icy::Point::NACCUpdateDeformationGradient(const float &dt,
                                   const Eigen::Matrix2f &FModifier,
                                               const icy::SimParams &prms)
{
    constexpr float magic_epsilon = 1e-5f;
    const float &mu = prms.mu;
    const float kappa = prms.mu*2.f/3 + prms.lambda; // bulk modulus
    const float &xi = prms.NACC_xi;
    const float &beta = prms.NACC_beta;
    const float &M = prms.NACC_M;
    const float M_sq = M*M;
    const int &d = prms.dim;
    float &alpha = NACC_alpha_p;

//    float dAlpha; //change in logJp, or the change in volumetric plastic strain
//    float dOmega; //change in logJp from q hardening (only for q hardening)

    Eigen::Matrix2f FeTr = (Eigen::Matrix2f::Identity() + dt * FModifier) * this->Fe;

    Eigen::JacobiSVD<Eigen::Matrix2f,Eigen::NoQRPreconditioner> svd(FeTr,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2f U = svd.matrixU();
    Eigen::Matrix2f V = svd.matrixV();
    Eigen::Vector2f Sigma = svd.singularValues();


    // line 4
    float p0 = kappa * (magic_epsilon + std::sinh(xi * std::max(-alpha, 0.f)));

    // line 5
    float Je_tr = Sigma[0]*Sigma[1];    // this is for 2D

    // line 6
    Eigen::Matrix2f SigmaMatrix = Sigma.asDiagonal();
    Eigen::Matrix2f SigmaSquared = SigmaMatrix*SigmaMatrix;
    Eigen::Matrix2f SigmaSquaredDev = SigmaSquared - SigmaSquared.trace()/2.f*Eigen::Matrix2f::Identity();
    float J_power_neg_2_d_mulmu = mu * std::pow(Je_tr, -2.f / (float)d);///< J^(-2/dim) * mu
    Eigen::Matrix2f s_hat_tr = J_power_neg_2_d_mulmu * SigmaSquaredDev;

    // line 7
    float psi_kappa_partial_J = (kappa/2.f) * (Je_tr - 1.f / Je_tr);

    // line 8
    float p_trial = -psi_kappa_partial_J * Je_tr;

    // line 9 (case 1)
    float y = (1.f + 2.f*beta)*(3.f-(float)d/2.f)*s_hat_tr.norm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    Eigen::Matrix2f Fe_new; // result of all this
    if(p_trial > p0)
    {
        float Je_new = std::sqrt(-2.f*p0 / kappa + 1.f);
        Eigen::Matrix2f Sigma_new = Eigen::Matrix2f::Identity() * pow(Je_new, 1.f/(float)d);
        Fe_new = U*Sigma_new*V.transpose();
        if(prms.NACC_hardening) alpha += std::log(Je_tr / Je_new);
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        float Je_new = std::sqrt(2.f*beta*p0/kappa + 1.f);
        Eigen::Matrix2f Sigma_new = Eigen::Matrix2f::Identity() * pow(Je_new, 1.f/(float)d);
        Fe_new = U*Sigma_new*V.transpose();
        if(prms.NACC_hardening) alpha += std::log(Je_tr / Je_new);
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon*10)
    {
        if(prms.NACC_hardening && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
        {
            float p_c = (1.f-beta)*p0/2.f;  // line 23
            float q_tr = sqrt(3.f-d/2.f)*s_hat_tr.norm();   // line 24
            Eigen::Vector2f direction(p_c-p_trial, -q_tr);  // line 25
            direction.normalize();
            float C = M_sq*(p_c-beta*p0)*(p_c-p0);
            float B = M_sq*direction[0]*(2.f*p_c-p0+beta*p0);
            float A = M_sq*direction[0]*direction[0]+(1.f+2.f*beta)*direction[1]*direction[1];  // line 30
            float l1 = (-B+sqrt(B*B-4.f*A*C))/(2.f*A);
            float l2 = (-B-sqrt(B*B-4.f*A*C))/(2.f*A);
            float p1 = p_c + l1*direction[0];
            float p2 = p_c + l2*direction[0];

            float p_x = (p_trial-p_c)*(p1-p_c) > 0 ? p1 : p2;
            float Je_x = sqrt(abs(-2.f*p_x/kappa + 1.f));
            if(Je_x > magic_epsilon*10) alpha += std::log(Je_tr / Je_x);

        }

        float expr_under_root = (-M*M*(p_trial+beta*p0)*(p_trial-p0))/((1+2.f*beta)*(3.f-d/2.));
        Eigen::Matrix2f B_hat_E_new = sqrt(expr_under_root)*(pow(Je_tr,2.f/d)/mu)*s_hat_tr.normalized() +
                Eigen::Matrix2f::Identity()*SigmaSquared.trace()/(float)d;
        Eigen::Matrix2f Sigma_new;
        Sigma_new.setZero();
        Sigma_new(0,0) = sqrt(B_hat_E_new(0,0));
        Sigma_new(1,1) = sqrt(B_hat_E_new(1,1));
        Fe_new = U*Sigma_new*V.transpose();
    }
    else
    {
        Fe_new = FeTr;
    }
    Fe = Fe_new;
}




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
