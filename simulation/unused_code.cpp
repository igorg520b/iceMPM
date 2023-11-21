/*
__device__ Matrix2r KirchhoffStress_Klar(const double mu, const double lambda, const Matrix2r &F)
{
    Matrix2r U, V, Sigma;
    svd2x2(F, U, Sigma, V);
    Matrix2r lnSigma,invSigma;
    lnSigma << log(Sigma(0,0)),0,0,log(Sigma(1,1));
    invSigma = Sigma.inverse();
    Matrix2r PFt = U*(2*mu*invSigma*lnSigma + lambda*lnSigma.trace()*invSigma)*V.transpose()*F.transpose();
    return PFt;
}

__device__ Matrix2r KirchhoffStress_Sifakis(const double mu, const double lambda, const Matrix2r &F)
{
    real Je = F.determinant();
    Matrix2r PFt = mu*F*F.transpose() + (-mu+lambda*log(Je))* Matrix2r::Identity();     // Neo-Hookean; Sifakis
    return PFt;
}


__device__ Matrix2r KirchhoffStress_Stomakhin(const double mu, const double lambda, const Matrix2r &F)
{
    Matrix2r Re = polar_decomp_R(F);
    real Je = F.determinant();
    Matrix2r PFt = 2.*mu*(F - Re)* F.transpose() + lambda * (Je - 1.) * Je * Matrix2r::Identity();
    return PFt;
}


__device__ void SnowUpdateDeformationGradient(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    const real &dt = gprms.InitialTimeStep;
    const real &THT_C_snow = gprms.THT_C_snow;
    const real &THT_S_snow = gprms.THT_S_snow;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;

    Matrix2r U, V, Sigma, SigmaClamped;
    svd2x2(FeTr, U, Sigma, V);
    SigmaClamped.setZero();
    SigmaClamped(0,0) = clamp(Sigma(0,0), 1.0 - THT_C_snow, 1.0 + THT_S_snow);
    SigmaClamped(1,1) = clamp(Sigma(1,1), 1.0 - THT_C_snow, 1.0 + THT_S_snow);
    p.Fe = U*SigmaClamped*V.transpose();

    p.Jp *= (1/SigmaClamped.determinant())*Sigma.determinant();
//    p.Jp *= (V*SigmaClamped.inverse()*Sigma*V.transpose()).determinant();
}


__device__ void DruckerPragerUpdateDeformationGradient(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    //    constexpr real magic_epsilon = 1.e-15;
    const real &mu = gprms.mu;
    const real &lambda = gprms.lambda;
    const real &dt = gprms.InitialTimeStep;
    const real &H0 = gprms.H0;
    const real &H1 = gprms.H1;
    const real &H2 = gprms.H2;
    const real &H3 = gprms.H3;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;

    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    Vector2r T;
    Matrix2r Tmatrix; // from DrySand::Plasticity()

    // Projection
    double dq = 0;
    Matrix2r lnSigma, e_c;
    lnSigma << log(Sigma(0,0)),0,0,log(Sigma(1,1));
    e_c = dev(lnSigma);  // deviatoric part

    //    if(e_c.norm() < magic_epsilon || lnSigma.trace()>0)
    if(e_c.norm() ==0 || lnSigma.trace()>0)
    {
        // Projection to the tip of the cone
        Tmatrix.setIdentity();
        dq = lnSigma.norm();
    }
    else
    {
        double phi = H0 + (H1 *p.q - H3)*exp(-H2 * p.q);
        double alpha = sqrt(2.0 / 3.0) * (2.0 * sin(phi)) / (3.0 - sin(phi));
        double dg = e_c.norm() + (lambda + mu) / mu * lnSigma.trace() * alpha;

        if (dg <= 0)
        {
            Tmatrix = Sigma;
            dq = 0;
        }
        else
        {
            Matrix2r Hm = lnSigma - e_c * (dg / e_c.norm());
            Tmatrix << exp(Hm(0,0)), 0, 0, exp(Hm(1,1));
            dq = dg;
        }
    }

    p.Fe = U*Tmatrix*V.transpose();
    p.q += dq; // hardening
}

__device__ void NACCUpdateDeformationGradient(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    constexpr real magic_epsilon = 1.e-5;
    constexpr int d = 2; // dimensions
    real &alpha = p.NACC_alpha_p;
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &beta = gprms.NACC_beta;
    const real &M_sq = gprms.NACC_M_sq;
    const real &xi = gprms.NACC_xi;
    const real &dt = gprms.InitialTimeStep;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;
    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    // line 4
    real p0 = kappa * (magic_epsilon + sinh(xi * max(-alpha, 0.)));

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
        if(true) alpha += log(Je_tr / Je_new);
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        real Je_new = sqrt(2.*beta*p0/kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        if(true) alpha += log(Je_tr / Je_new);
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
        if(Je_x > magic_epsilon) alpha += log(Je_tr / Je_x);

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




bool icy::Model::Step()
{
    if(isTimeToUpdate()) spdlog::info("step {} started", prms.SimulationStep);

    indenter_x = indenter_x_initial + prms.SimulationTime*prms.IndVelocity;
    if(isTimeToUpdate()) gpu.start_timing();
    if(prms.useGPU)
    {
        gpu.cuda_reset_grid(grid.size());
        gpu.cuda_p2g(points.size());
        gpu.cuda_update_nodes(grid.size(),indenter_x, indenter_y);
        gpu.cuda_g2p(points.size());
    }
    else
    {
        ResetGrid();
        P2G();
        if(abortRequested) return false;
        UpdateNodes();
        if(abortRequested) return false;
        G2P();
        if(abortRequested) return false;
    }

    prms.SimulationStep++;
    prms.SimulationTime += prms.InitialTimeStep;
    if(isTimeToUpdate())
    {
        spdlog::info("step {} completed\n", prms.SimulationStep-1);
        if(prms.useGPU)
        {
            compute_time_per_cycle = gpu.end_timing()/prms.UpdateEveryNthStep;
            gpu.cuda_device_synchronize();
            visual_update_mutex.lock();
            gpu.cuda_transfer_from_device(points);
            visual_update_mutex.unlock();
        }
    }

    if(prms.SimulationTime >= prms.SimulationEndTime) return false;
    return true;
}


void icy::Model::ResetGrid()
{
    if(isTimeToUpdate()) spdlog::info("s {}; reset grid", prms.SimulationStep);
    memset(grid.data(), 0, grid.size()*sizeof(icy::GridNode));
}


void icy::Model::P2G()
{
    if(isTimeToUpdate()) spdlog::info("s {}; p2g", prms.SimulationStep);

    const real &h = prms.cellsize;
    const real &dt = prms.InitialTimeStep;
    const real &Dinv = prms.Dp_inv;
    const real &vol = prms.ParticleVolume;
    const real &particle_mass = prms.ParticleMass;

#pragma omp parallel for
    for(int pt_idx=0; pt_idx<points.size(); pt_idx++)
    {
        Point &p = points[pt_idx];

//        Matrix2r Ap;
        //Ap = p.NACCConstitutiveModel(prms.mu, prms.lambda, prms.ParticleVolume);
        //Ap = p.SnowConstitutiveModel(prms.XiSnow, prms.mu, prms.lambda, prms.ParticleVolume);
//        Ap = p.ElasticConstitutiveModel(prms.mu, prms.lambda, prms.ParticleVolume);

        Matrix2r Re = icy::Point::polar_decomp_R(p.Fe);
        real Je = p.Fe.determinant();
        Matrix2r dFe = 2. * prms.mu*(p.Fe - Re)* p.Fe.transpose() +
                prms.lambda * (Je - 1.) * Je * Matrix2r::Identity();

        Matrix2r stress = - (dt * vol) * (Dinv * dFe);

        // Fused APIC momentum + MLS-MPM stress contribution
         // See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
         // Eqn 29
        Matrix2r affine = stress + particle_mass * p.Bp;

        constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
        const int i0 = (int)((p.pos[0])/h - offset);
        const int j0 = (int)((p.pos[1])/h - offset);

        Vector2r base_coord(i0,j0);
        Vector2r fx = p.pos/h - base_coord;

        Vector2r v0(1.5-fx[0],1.5-fx[1]);
        Vector2r v1(fx[0]-1.,fx[1]-1.);
        Vector2r v2(fx[0]-.5,fx[1]-.5);

        Vector2r w[3];
        w[0] << .5*v0[0]*v0[0],  .5*v0[1]*v0[1];
        w[1] << .75-v1[0]*v1[0], .75-v1[1]*v1[1];
        w[2] << .5*v2[0]*v2[0],  .5*v2[1]*v2[1];


        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                real Wip = w[i].x()*w[j].y();

                Vector2r dpos((i-fx[0])*h, (j-fx[1])*h);
                Vector2r incV = Wip*(p.velocity*particle_mass+affine*dpos);
                real incM = Wip*particle_mass;

                int idx_gridnode = (i+i0) + (j+j0)*prms.GridX;
                if((i+i0) < 0 || (j+j0) < 0 || (i+i0) >=prms.GridX || (j+j0)>=prms.GridY || idx_gridnode < 0 || idx_gridnode >= grid.size())
                {
                    spdlog::critical("point {} in cell [{}, {}]", pt_idx, (i+i0), (j+j0));
                    throw std::runtime_error("particle is out of grid bounds");
                }

                GridNode &gn = grid[idx_gridnode];
#pragma omp atomic
                gn.mass += incM;

#pragma omp atomic
                gn.velocity[0] += incV[0];
#pragma omp atomic
                gn.velocity[1] += incV[1];
            }
    }
}


void icy::Model::UpdateNodes()
{
    if(isTimeToUpdate()) spdlog::info("s {}; update nodes", prms.SimulationStep);

    const real dt = prms.InitialTimeStep;
    const Vector2r gravity(0,-prms.Gravity);
    const real indRsq = prms.IndRSq;
    const Vector2r vco(prms.IndVelocity,0);  // velocity of the collision object (indenter)
    const Vector2r indCenter(indenter_x, indenter_y);

#pragma omp parallel for schedule (dynamic)
    for (int idx = 0; idx < grid.size(); idx++)
    {
        GridNode &gn = grid[idx];
        if(gn.mass == 0) continue;
        gn.velocity /= gn.mass;
        gn.velocity[1] -= dt*prms.Gravity;

        int idx_x = idx % prms.GridX;
        int idx_y = idx / prms.GridX;

        // indenter
        Vector2r gnpos(idx_x * prms.cellsize,idx_y * prms.cellsize);
        Vector2r n = gnpos - indCenter;
        if(n.squaredNorm() < indRsq)
        {
            // grid node is inside the indenter
            Vector2r vrel = gn.velocity - vco;
            n.normalize();
            real vn = vrel.dot(n);   // normal component of the velocity
            if(vn < 0)
            {
                Vector2r vt = vrel - n*vn;   // tangential portion of relative velocity
                gn.velocity = vco + vt + prms.IceFrictionCoefficient*vn*vt.normalized();
            }
        }

        // attached bottom layer
        if(idx_y <= 3) gn.velocity.setZero();
        else if(idx_y >= prms.GridY-4 && gn.velocity[1]>0) gn.velocity[1] = 0;
        if(idx_x <= 3 && gn.velocity.x()<0) gn.velocity[0] = 0;
        else if(idx_x >= prms.GridX-5) gn.velocity[0] = 0;
    }

}



void icy::Model::G2P()
{
    if(isTimeToUpdate()) spdlog::info("s {}; g2p", prms.SimulationStep);

    const real &dt = prms.InitialTimeStep;
    const real &h = prms.cellsize;
    constexpr real offset = 0.5;  // 0 for cubic


    visual_update_mutex.lock();
#pragma omp parallel for
    for(int idx_p = 0; idx_p<points.size(); idx_p++)
    {
        icy::Point &p = points[idx_p];

        const int i0 = (int)((p.pos[0])/h - offset);
        const int j0 = (int)((p.pos[1])/h - offset);

        Vector2r base_coord(i0,j0);
        Vector2r fx = p.pos/h - base_coord;

        Vector2r v0(1.5-fx[0],1.5-fx[1]);
        Vector2r v1(fx[0]-1.,fx[1]-1.);
        Vector2r v2(fx[0]-.5,fx[1]-.5);

        Vector2r w[3];
        w[0] << 0.5f*v0[0]*v0[0], 0.5f*v0[1]*v0[1];
        w[1] << 0.75f-v1[0]*v1[0], 0.75f-v1[1]*v1[1];
        w[2] << 0.5f*v2[0]*v2[0], 0.5f*v2[1]*v2[1];

        //const Vector2r pointPos_copy = p.pos;
        //p.pos.setZero();
        p.velocity.setZero();
        p.Bp.setZero();

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                Vector2r dpos = Vector2r(i, j) - fx;
                real weight = w[i].x() * w[j].y();

                int idx_gridnode = i+i0 + (j+j0)*prms.GridX;
                const icy::GridNode &node = grid[idx_gridnode];
                const Vector2r &grid_v = node.velocity;
                p.velocity += weight * grid_v;
                p.Bp += (4./h)*weight *(grid_v*dpos.transpose());
            }

        // Advection
        p.pos += dt * p.velocity;


        p.Fe = (Matrix2r::Identity() + dt*p.Bp) * p.Fe;


//        p.NACCUpdateDeformationGradient(dt,T,prms);
//        p.SnowUpdateDeformationGradient(dt,prms.THT_C_snow,prms.THT_S_snow,T);
//        p.ElasticUpdateDeformationGradient(dt,T);

    }
    visual_update_mutex.unlock();


}


*/
