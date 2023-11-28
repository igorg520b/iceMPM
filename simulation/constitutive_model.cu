#include "gpu_implementation3_sand.h"

extern __constant__ icy::SimParams gprms;


__device__ void NACCUpdateDeformationGradient_q_hardening_2(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    constexpr real magic_epsilon = 1.e-5;
    constexpr int d = 2; // dimensions
    const real &dt = gprms.InitialTimeStep;
    const real &ms = gprms.NACC_max_strain;


    real kappa = gprms.kappa;
    real mu = gprms.mu;
    real &J_inv = p.Jp_inv;
    real &zeta = p.zeta;

    if(J_inv > 1) J_inv = 1;
    if(zeta > 1 ) zeta = 1;
    kappa *= exp((J_inv-1)/ms);
    mu *= exp(((J_inv*zeta)-1)/ms);

    real beta = gprms.NACC_beta;
    beta *= exp((J_inv-1)/ms);



    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;
    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    real p0 = gprms.IceCompressiveStrength;

    // compute p_trial
    real Je_tr = Sigma(0,0)*Sigma(1,1);    // this is for 2D
    real psi_kappa_prime = (kappa/2.) * (Je_tr - 1./Je_tr);
    real p_trial = -psi_kappa_prime * Je_tr;
//    if(p_trial > 0.5 * gprms.IceCompressiveStrength) zeta = 0.5*(1+zeta);

    real M = gprms.NACC_M;
    real coeff1 = 1;
    real coeff2 = 0.8;
    M *= coeff2 + (1-coeff2)*exp((zeta-1)/coeff1);
    const real M_sq = M*M;

    // compute s_trial and q_trial
    Matrix2r SigmaSquared = Sigma*Sigma;
    Matrix2r s_hat_tr = mu/Je_tr * dev(SigmaSquared); //mu * pow(Je_tr, -2. / (real)d)* dev(SigmaSquared);
    real q_tr = sqrt((6-d)/2.)*s_hat_tr.norm();

    p.visualize_p = p_trial;
    p.visualize_q = s_hat_tr.norm()*sqrt((6-d)/2.);

    // line 9 (case 1)
    real y = (1. + 2.*beta)*(3.-(real)d/2.)*s_hat_tr.squaredNorm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(p_trial > p0)
    {
        p.q = 1;
        Matrix2r Sigma_new;
        if(J_inv < 1)
        {
            real Je_new = sqrt(-2.*p0 / kappa + 1.);
            Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
            J_inv *= Je_new/Je_tr;
            J_inv = min(J_inv, 1.);
        }
        else
        {
            Sigma_new = Matrix2r::Identity() * pow(Je_tr, 1./(real)d);
        }
        p.Fe = U*Sigma_new*V.transpose();
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        p.q = 2;
        real Je_new = sqrt(2.*beta*p0/kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        J_inv *= Je_new/Je_tr;
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
    {
        // projection
        real expr_under_root = (-M_sq*(p_trial+beta*p0)*(p_trial-p0))/((1+2.*beta)*(3.-d/2.));
        //        Matrix2r B_hat_E_new = sqrt(expr_under_root)*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2r::Identity()*(SigmaSquared.trace()/d);
        Matrix2r B_hat_E_new = sqrt(expr_under_root)*Je_tr/mu*s_hat_tr.normalized() + Matrix2r::Identity()*(SigmaSquared.trace()/d);
        Matrix2r Sigma_new;
        Sigma_new << sqrt(B_hat_E_new(0,0)), 0, 0, sqrt(B_hat_E_new(1,1));
        p.Fe = U*Sigma_new*V.transpose();

        // update hardening
        // zeta_tr
        real zeta_tr = sqrt((q_tr*Je_tr)/(mu*sqrt((6-d)/2.)) + 1);

        // zeta_n_1
        real Je_n_1 = Sigma_new(0,0)*Sigma_new(1,1);
        Matrix2r SigmaSquared_n_1 = Sigma_new*Sigma_new;
        Matrix2r s_hat_n_1 = mu/Je_n_1 * dev(SigmaSquared_n_1); //mu * pow(Je_tr, -2. / (real)d)* dev(SigmaSquared);
        real q_n_1 = sqrt((6-d)/2.)*s_hat_n_1.norm();
        real zeta_n_1 = sqrt((q_n_1*Je_tr)/(mu*sqrt((6-d)/2.)) + 1);

        p.zeta *= zeta_n_1/zeta_tr;
        p.q = 3;
    }
    else
    {
        p.Fe = FeTr;
    }
}


/*
__device__ void NACCUpdateDeformationGradient_q_hardening(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    constexpr real magic_epsilon = 1.e-5;
    constexpr int d = 2; // dimensions
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &beta = gprms.NACC_beta;
    const real &M_sq = gprms.NACC_M_sq;
    const real &dt = gprms.InitialTimeStep;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;
    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    // line 4

    const real &ms = gprms.NACC_max_strain;
    real p0 = gprms.IceCompressiveStrength/ms * (p.Jp_inv - (1.-ms));
    p0 = max(magic_epsilon, p0);
    p.visualize_p0 = p0;

    // line 5
    real Je_tr = Sigma(0,0)*Sigma(1,1);    // this is for 2D

    // line 6
    Matrix2r SigmaSquared = Sigma*Sigma;
    Matrix2r s_hat_tr = mu/Je_tr * dev(SigmaSquared); //mu * pow(Je_tr, -2. / (real)d)* dev(SigmaSquared);

    // line 7
    real psi_kappa_prime = (kappa/2.) * (Je_tr - 1./Je_tr);

    // line 8
    real p_trial = -psi_kappa_prime * Je_tr;
    p.visualize_p = p_trial;
    p.visualize_q = s_hat_tr.norm()*sqrt((6-d)/2.);

    // line 9 (case 1)
    real y = (1. + 2.*beta)*(3.-(real)d/2.)*s_hat_tr.squaredNorm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(p_trial > p0)
    {
        p.q = 1;
        if(p.Jp_inv < 1)
        {
            real Je_new = sqrt(-2.*p0 / kappa + 1.);
            Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
            p.Fe = U*Sigma_new*V.transpose();
            p.Jp_inv *= Je_new/Je_tr;
        }
        else
        {
            Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_tr, 1./(real)d);
            p.Fe = U*Sigma_new*V.transpose();
        }
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        p.q = 2;
        real Je_new = sqrt(2.*beta*p0/kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        p.Jp_inv *= Je_new/Je_tr;
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
    {
        // projection
        real expr_under_root = (-M_sq*(p_trial+beta*p0)*(p_trial-p0))/((1+2.*beta)*(3.-d/2.));
        //        Matrix2r B_hat_E_new = sqrt(expr_under_root)*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2r::Identity()*(SigmaSquared.trace()/d);
        Matrix2r B_hat_E_new = sqrt(expr_under_root)*Je_tr/mu*s_hat_tr.normalized() + Matrix2r::Identity()*(SigmaSquared.trace()/d);
        Matrix2r Sigma_new;
        Sigma_new << sqrt(B_hat_E_new(0,0)), 0, 0, sqrt(B_hat_E_new(1,1));
        p.Fe = U*Sigma_new*V.transpose();

        // update hardening
        // zeta_tr
        real q_tr = sqrt((6-d)/2.)*s_hat_tr.norm();
        real zeta_tr = sqrt((q_tr*Je_tr)/(mu*sqrt((6-d)/2.)) + 1);

        // zeta_n_1
        real Je_n_1 = Sigma_new(0,0)*Sigma_new(1,1);
        Matrix2r SigmaSquared_n_1 = Sigma_new*Sigma_new;
        Matrix2r s_hat_n_1 = mu/Je_n_1 * dev(SigmaSquared_n_1); //mu * pow(Je_tr, -2. / (real)d)* dev(SigmaSquared);
        real q_n_1 = sqrt((6-d)/2.)*s_hat_n_1.norm();
        real zeta_n_1 = sqrt((q_n_1*Je_tr)/(mu*sqrt((6-d)/2.)) + 1);

        real p_c = (1.-beta)*p0/2.;
        //        real coeff = gprms.NACC_magic_coeff;
        //        real p_c = coeff*p0 - (1.-coeff)*beta*p0;
        if(p_trial > p_c)
        {
            p.Jp_inv *= zeta_tr/zeta_n_1;           // original version
            p.q = 3;
        }
        else
        {
            p.Jp_inv *= zeta_n_1/zeta_tr;
            p.q = 4;
        }
    }
    else
    {
        p.Fe = FeTr;
    }
}
*/

/*
__device__ void NACCUpdateDeformationGradient_Alt(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    constexpr real magic_epsilon = 1.e-5;
    constexpr int d = 2; // dimensions
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &beta = gprms.NACC_beta;
    real M_sq = gprms.NACC_M_sq;
    const real &xi = gprms.NACC_xi;
    const real &dt = gprms.InitialTimeStep;

    real exp1 = p.Jp < 1 ? 1 : exp(xi*(1.0 - p.Jp));


    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;
    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    // line 4
    real p0 = gprms.IceCompressiveStrength*exp1;

    // line 5
    real Je_tr = Sigma(0,0)*Sigma(1,1);    // this is for 2D

    // line 6
    Matrix2r SigmaSquared = Sigma*Sigma;
    Matrix2r s_hat_tr = mu/Je_tr * dev(SigmaSquared); //mu * pow(Je_tr, -2. / (real)d)* dev(SigmaSquared);

    // line 7
    real psi_kappa_prime = (kappa/2.) * (Je_tr - 1./Je_tr);

    // line 8
    real p_trial = -psi_kappa_prime * Je_tr;

    // line 9 (case 1)
    real y = (1. + 2.*beta)*(3.-(real)d/2.)*s_hat_tr.squaredNorm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(p_trial > p0)
    {
        real Je_new = sqrt(-2.*p0 / kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        p.Jp *= (Je_tr / Je_new);
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        real Je_new = sqrt(2.*beta*p0/kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        p.Jp *= (Je_tr / Je_new);
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
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
        if(Je_x > magic_epsilon) p.Jp *= (Je_tr / Je_x);

        real expr_under_root = (-M_sq*(p_trial+beta*p0)*(p_trial-p0))/((1+2.*beta)*(3.-d/2.));
        //        Matrix2r B_hat_E_new = sqrt(expr_under_root)*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2r::Identity()*(SigmaSquared.trace()/d);
        Matrix2r B_hat_E_new = sqrt(expr_under_root)*Je_tr/mu*s_hat_tr.normalized() + Matrix2r::Identity()*(SigmaSquared.trace()/d);
        Matrix2r Sigma_new;
        Sigma_new << sqrt(B_hat_E_new(0,0)), 0, 0, sqrt(B_hat_E_new(1,1));
        p.Fe = U*Sigma_new*V.transpose();
    }
    else
    {
        p.Fe = FeTr;
    }
}
*/
