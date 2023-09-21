#include "point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Core>



Eigen::Matrix2f icy::Point::NACCConstitutiveModel(const float &prmsMu,
                                      const float &prmsLambda,
                                      const float &particle_volume) const
{
    Eigen::Matrix2f Ap;

    return Ap;
}


void icy::Point::NACCUpdateDeformationGradient(const float &dt,
                                   const Eigen::Matrix2f &FModifier,
                                               const icy::SimParams &prms)
{
    float dAlpha; //change in logJp, or the change in volumetric plastic strain
    float dOmega; //change in logJp from q hardening (only for q hardening)

    Eigen::Matrix2f FeTr = (Eigen::Matrix2f::Identity() + dt * FModifier) * this->Fe;

    Eigen::JacobiSVD<Eigen::Matrix2f,Eigen::NoQRPreconditioner> svd(FeTr,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2f U = svd.matrixU();
    Eigen::Matrix2f V = svd.matrixV();
    Eigen::Vector2f Sigma = svd.singularValues();

    constexpr float magic_epsilon = 1e-5f;
    const float kappa = prms.mu*2.f/3 + prms.lambda; // bulk modulus
    const float &xi = prms.NACC_xi;
    float p0 = kappa * (magic_epsilon + std::sinh(xi * std::max(-logJp, 0)));


/*
    float p0	= data.bm * (static_cast<float>(0.00001) + sinh(data.xi * (-data.log_jp > 0 ? -data.log_jp : 0)));
    float p_min = -data.beta * p0;

    float Je_trial = S[0] * S[1] * S[2];

*/

    /*


    T J = 1.;
    for (int i = 0; i < dim; ++i) J *= sigma(i);

    //Step 1, compute pTrial and see if case 1, 2, or 3
    TV B_hat_trial;
    for (int i = 0; i < dim; ++i)
        B_hat_trial(i) = sigma(i) * sigma(i);
    TV s_hat_trial = c.mu * std::pow(J, -(T)2 / (T)dim) * deviatoric(B_hat_trial);

    T prime = c.kappa / (T)2 * (J - 1 / J);
    T p_trial = -prime * J;

    //Cases 1 and 2 (Ellipsoid Tips)
    //Project to the tips
    T pMin = beta * p0;
    T pMax = p0;
    if (p_trial > pMax) {
        T Je_new = std::sqrt(-2 * pMax / c.kappa + 1);
        sigma = TV::Ones() * std::pow(Je_new, (T)1 / dim);
        Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
        TM Fe = U * sigma_m * V.transpose();
        strain = Fe;
        if (hardeningOn) {
            logJp += log(J / Je_new);
        }
        return false;
    }
    else if (p_trial < -pMin) {
        T Je_new = std::sqrt(2 * pMin / c.kappa + 1);
        sigma = TV::Ones() * std::pow(Je_new, (T)1 / dim);
        Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
        TM Fe = U * sigma_m * V.transpose();
        strain = Fe;
        if (hardeningOn) {
            logJp += log(J / Je_new);
        }
        return false;
    }

    //Case 3 --> check if inside or outside YS
    T y_s_half_coeff = ((T)6 - dim) / (T)2 * ((T)1 + (T)2 * beta);
    T y_p_half = M * M * (p_trial + pMin) * (p_trial - pMax);
    T y = y_s_half_coeff * s_hat_trial.squaredNorm() + y_p_half;

    //Case 3a (Inside Yield Surface)
    //Do nothing
    if (y < 1e-4) return false;

    //Case 3b (Outside YS)
    // project to yield surface
    TV B_hat_new = std::pow(J, (T)2 / (T)dim) / c.mu * std::sqrt(-y_p_half / y_s_half_coeff) * s_hat_trial / s_hat_trial.norm();
    B_hat_new += (T)1 / dim * B_hat_trial.sum() * TV::Ones();

    for (int i = 0; i < dim; ++i)
        sigma(i) = std::sqrt(B_hat_new(i));
    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    strain = Fe;

    //Step 2: Hardening
    //Three approaches to hardening:
    //0 -> hack the hardening by computing a fake delta_p
    //1 -> q based
    if (p0 > 1e-4 && p_trial < pMax - 1e-4 && p_trial > 1e-4 - pMin) {
        T p_center = p0 * ((1 - beta) / (T)2);
        T q_trial = std::sqrt(((T)6 - (T)dim) / (T)2) * s_hat_trial.norm();
        Vector<T, 2> direction;
        direction(0) = p_center - p_trial;
        direction(1) = 0 - q_trial;
        direction = direction / direction.norm();

        T C = M * M * (p_center + beta * p0) * (p_center - p0);
        T B = M * M * direction(0) * (2 * p_center - p0 + beta * p0);
        T A = M * M * direction(0) * direction(0) + (1 + 2 * beta) * direction(1) * direction(1);

        T l1 = (-B + std::sqrt(B * B - 4 * A * C)) / (2 * A);
        T l2 = (-B - std::sqrt(B * B - 4 * A * C)) / (2 * A);

        T p1 = p_center + l1 * direction(0);
        T p2 = p_center + l2 * direction(0);
        T p_fake = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;

        //Only for pFake Hardening
        T Je_new_fake = sqrt(std::abs(-2 * p_fake / c.kappa + 1));
        dAlpha = log(J / Je_new_fake);

        //Only for q Hardening
        T qNPlus = sqrt(M * M * (p_trial + pMin) * (pMax - p_trial) / ((T)1 + (T)2 * beta));
        T Jtrial = J;
        T zTrial = sqrt(((q_trial * pow(Jtrial, ((T)2 / (T)dim))) / (c.mu * sqrt(((T)6 - (T)dim) / (T)2))) + 1);
        T zNPlus = sqrt(((qNPlus * pow(Jtrial, ((T)2 / (T)dim))) / (c.mu * sqrt(((T)6 - (T)dim) / (T)2))) + 1);
        if (p_trial > p_fake) {
            dOmega = -1 * log(zTrial / zNPlus);
        }
        else {
            dOmega = log(zTrial / zNPlus);
        }

        if (hardeningOn) {
            if (!qHard) {
                if (Je_new_fake > 1e-4) {
                    logJp += dAlpha;
                }
            }
            else if (qHard) {
                if (zNPlus > 1e-4) {
                    logJp += dOmega;
                }
            }
        }
    }
*/
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
