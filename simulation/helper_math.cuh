#ifndef HELPER_MATH_CUH
#define HELPER_MATH_CUH

#include "givens.cuh"
#include <Eigen/Core>
#include <cuda.h>
#include <cuda_runtime.h>

#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"

__constant__ icy::SimParams gprms;
__device__ int gpu_error_indicator;
__device__ icy::Point *gpu_points;
__device__ icy::GridNode *gpu_nodes;

__device__ void svd2x2(const Eigen::Matrix2f &mA, Eigen::Matrix2f &mU, Eigen::Matrix2f &mS, Eigen::Matrix2f &mV);

__device__ void NACCUpdateDeformationGradient(icy::Point &p, Eigen::Matrix2f &FModifier)
{
    constexpr float magic_epsilon = 1e-5f;
    constexpr int d = 2; // dimensions
    float &alpha = p.NACC_alpha_p;
    const float &mu = gprms.mu;
    const float &kappa = gprms.kappa;
    const float &beta = gprms.NACC_beta;
    const float &M_sq = gprms.NACC_M_sq;
    const float &xi = gprms.NACC_xi;
    const bool &hardening = gprms.NACC_hardening;
    const float &dt = gprms.InitialTimeStep;

    Eigen::Matrix2f FeTr = (Eigen::Matrix2f::Identity() + dt * FModifier) * p.Fe;

    Eigen::Matrix2f U, V, Sigma;

    svd2x2(FeTr, U, Sigma, V);

    // line 4
    float p0 = kappa * (magic_epsilon + sinhf(xi * fmaxf(-alpha, 0.f)));

    // line 5
    float Je_tr = Sigma(0,0)*Sigma(1,1);    // this is for 2D

    // line 6
    Eigen::Matrix2f SigmaSquared = Sigma*Sigma;
    Eigen::Matrix2f SigmaSquaredDev = SigmaSquared - SigmaSquared.trace()/2.f*Eigen::Matrix2f::Identity();
    float J_power_neg_2_d_mulmu = mu * powf(Je_tr, -2.f / (float)d);///< J^(-2/dim) * mu
    Eigen::Matrix2f s_hat_tr = J_power_neg_2_d_mulmu * SigmaSquaredDev;

    // line 7
    float psi_kappa_partial_J = (kappa/2.f) * (Je_tr - 1.f / Je_tr);

    // line 8
    float p_trial = -psi_kappa_partial_J * Je_tr;

    // line 9 (case 1)
    float y = (1.f + 2.f*beta)*(3.f-(float)d/2.f)*s_hat_tr.norm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(p_trial > p0)
    {
        float Je_new = sqrtf(-2.f*p0 / kappa + 1.f);
        Eigen::Matrix2f Sigma_new = Eigen::Matrix2f::Identity() * powf(Je_new, 1.f/(float)d);
        p.Fe = U*Sigma_new*V.transpose();
        if(hardening) alpha += logf(Je_tr / Je_new);
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        float Je_new = sqrtf(2.f*beta*p0/kappa + 1.f);
        Eigen::Matrix2f Sigma_new = Eigen::Matrix2f::Identity() * pow(Je_new, 1.f/(float)d);
        p.Fe = U*Sigma_new*V.transpose();
        if(hardening) alpha += logf(Je_tr / Je_new);
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon*10)
    {
        if(hardening && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
        {
            float p_c = (1.f-beta)*p0/2.f;  // line 23
            float q_tr = sqrtf(3.f-d/2.f)*s_hat_tr.norm();   // line 24
            Eigen::Vector2f direction(p_c-p_trial, -q_tr);  // line 25
            direction.normalize();
            float C = M_sq*(p_c-beta*p0)*(p_c-p0);
            float B = M_sq*direction[0]*(2.f*p_c-p0+beta*p0);
            float A = M_sq*direction[0]*direction[0]+(1.f+2.f*beta)*direction[1]*direction[1];  // line 30
            float l1 = (-B+sqrtf(B*B-4.f*A*C))/(2.f*A);
            float l2 = (-B-sqrtf(B*B-4.f*A*C))/(2.f*A);
            float p1 = p_c + l1*direction[0];
            float p2 = p_c + l2*direction[0];
            float p_x = (p_trial-p_c)*(p1-p_c) > 0 ? p1 : p2;
            float Je_x = sqrtf(fabsf(-2.f*p_x/kappa + 1.f));
            if(Je_x > magic_epsilon*10) alpha += logf(Je_tr / Je_x);
        }

        float expr_under_root = (-M_sq*(p_trial+beta*p0)*(p_trial-p0))/((1+2.f*beta)*(3.f-d/2.));
        Eigen::Matrix2f B_hat_E_new = sqrtf(expr_under_root)*(powf(Je_tr,2.f/d)/mu)*s_hat_tr.normalized() +
                Eigen::Matrix2f::Identity()*SigmaSquared.trace()/(float)d;
        Eigen::Matrix2f Sigma_new;
        Sigma_new << sqrt(B_hat_E_new(0,0)), 0,
                0, sqrt(B_hat_E_new(1,1));
        p.Fe = U*Sigma_new*V.transpose();
    }
    else
    {
        p.Fe = FeTr;
    }
    //p.visualized_value = alpha;
}




__device__ Eigen::Matrix2f polar_decomp_R(const Eigen::Matrix2f &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    float th = atan2f(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Eigen::Matrix2f result;
    result << cosf(th), -sinf(th), sinf(th), cosf(th);
    return result;
}



__device__ float wqs(float x)
{
    x = fabsf(x);
    if (x < 0.5f) return -x * x + 3 / 4.0f;
    else if (x < 1.5f) return x * x / 2.0f - 3 * x / 2.0f + 9 / 8.0f;
    return 0;
}

__device__ float dwqs(float x)
{
    float x_abs = fabsf(x);
    if (x_abs < 0.5f) return -2.0f * x;
    else if (x_abs < 1.5f) return x - 3 / 2.0f * x / x_abs;
    return 0;
}

__device__ float wq(Eigen::Vector2f dx, double h)
{
    return wqs(dx[0]/h)*wqs(dx[1]/h);
}

__device__ Eigen::Vector2f gradwq(Eigen::Vector2f dx, double h)
{
    Eigen::Vector2f result;
    result[0] = dwqs(dx[0]/h)*wqs(dx[1]/h)/h;
    result[1] = wqs(dx[0]/h)*dwqs(dx[1]/h)/h;
    return result;
}


/**
 \brief 2x2 polar decomposition.
 \param[in] A matrix.
 \param[out] R Robustly a rotation matrix in givens form
 \param[out] S_Sym Symmetric. Whole matrix is stored

 Whole matrix S is stored since its faster to calculate due to simd vectorization
 Polar guarantees negative sign is on the small magnitude singular value.
 S is guaranteed to be the closest one to identity.
 R is guaranteed to be the closest rotation to A.
 */
template<typename T>
__device__ void polar_decomposition(const T a[4],
                GivensRotation<T>& r,
                T s[4]) {
    double x[2]		   = {a[0] + a[3], a[1] - a[2]};
    double denominator = sqrt(x[0] * x[0] + x[1] * x[1]);
    r.c				   = (T) 1;
    r.s				   = (T) 0;
    if(denominator != 0) {
        /*
      No need to use a tolerance here because x(0) and x(1) always have
      smaller magnitude then denominator, therefore overflow never happens.
    */
        r.c = x[0] / denominator;
        r.s = -x[1] / denominator;
    }
    for(int i = 0; i < 4; ++i) {
        s[i] = a[i];
    }
    r.template mat_rotation<2, T>(s);
}

/**
\brief 2x2 polar decomposition.
\param[in] A matrix.
\param[out] R Robustly a rotation matrix.
\param[out] S_Sym Symmetric. Whole matrix is stored

Whole matrix S is stored since its faster to calculate due to simd vectorization
Polar guarantees negative sign is on the small magnitude singular value.
S is guaranteed to be the closest one to identity.
R is guaranteed to be the closest rotation to A.
*/
template<typename T>
__device__ void polar_decomposition(const T a[4], T r[4], T s[4]) {
    GivensRotation<T> rotation(0, 1);
    polar_decomposition(a, rotation, s);
    rotation.fill<2>(r);
}


template <typename T> __device__ void inline my_swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}


/**
\brief 2x2 SVD (singular value decomposition) A=USV'
\param[in] A Input matrix.
\param[out] u Robustly a rotation matrix in Givens form
\param[out] sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
\param[out] V Robustly a rotation matrix in Givens form
*/
template<typename T>
__device__ void singular_value_decomposition(
        const T aa[4],
        GivensRotation<double>& u,
        T sigma[2],
        GivensRotation<double>& v) {

    double s_sym[4];///< column-major
    double a[4] {aa[0], aa[2], aa[1], aa[3]};
    polar_decomposition(a, u, s_sym);
    double cosine;
    double sine;
    double x  = s_sym[0];
    double y  = s_sym[2];
    double z  = s_sym[3];
    double y2 = y * y;
    if(y2 == 0) {
        // S is already diagonal
        cosine	 = 1;
        sine	 = 0;
        sigma[0] = x;
        sigma[1] = z;
    } else {
        double tau = T(0.5) * (x - z);
        double w   = sqrt(tau * tau + y2);
        // w > y > 0
        double t;
        if(tau > 0) {
            // tau + w > w > y > 0 ==> division is safe
            t = y / (tau + w);
        } else {
            // tau - w < -w < -y < 0 ==> division is safe
            t = y / (tau - w);
        }
        cosine = T(1) / sqrt(t * t + T(1));
        sine   = -t * cosine;
        /*
      v = [cosine -sine; sine cosine]
      sigma = v'SV. Only compute the diagonals for efficiency.
      Also utilize symmetry of S and don't form v yet.
    */
        double c2  = cosine * cosine;
        double csy = 2 * cosine * sine * y;
        double s2  = sine * sine;
        sigma[0]   = c2 * x - csy + s2 * z;
        sigma[1]   = s2 * x + csy + c2 * z;
    }

    // Sorting
    // Polar already guarantees negative sign is on the small magnitude singular value.
    if(sigma[0] < sigma[1])
    {
        my_swap(sigma[0], sigma[1]);
        v.c = -sine;
        v.s = cosine;
    } else {
        v.c = cosine;
        v.s = sine;
    }
    u *= v;
}


/**
\brief 2x2 SVD (singular value decomposition) a=USV'
\param[in] a Input matrix.
\param[out] u Robustly a rotation matrix.
\param[out] sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
\param[out] v Robustly a rotation matrix.
*/
__device__ void svd(const float a[4], float u[4], float sigma[2], float v[4])
{
    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(a, gu, sigma, gv);
    gu.template fill<2, float>(u);
    gv.template fill<2, float>(v);
}


__device__ void svd2x2(const Eigen::Matrix2f &mA,
                       Eigen::Matrix2f &mU,
                       Eigen::Matrix2f &mS,
                       Eigen::Matrix2f &mV)
{
    float U[4], V[4], S[2];
    float a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);

    mU << U[0],U[1],U[2],U[3];
    mS << S[0],0,0,S[1];
    mV << V[0],V[1],V[2],V[3];
}


#endif // HELPER_MATH_CUH
