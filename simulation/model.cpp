#include "model.h"



void icy::Model::P2G()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; p2g", currentStep);

    const float &h = prms.cellsize;
//    const float Dp_inv = 3.f/(h*h); // cubic
    const float Dp_inv = 4.f/(h*h); // quadratic

#pragma omp parallel for
    for(int pt_idx=0; pt_idx<points.size(); pt_idx++)
    {
        Point &p = points[pt_idx];

        Eigen::Matrix2f Ap;
        Ap = p.NACCConstitutiveModel(prms.mu, prms.lambda, prms.ParticleVolume);
        //Ap = p.SnowConstitutiveModel(prms.XiSnow, prms.mu, prms.lambda, prms.ParticleVolume);
        //Ap = p.ElasticConstitutiveModel(prms.mu, prms.lambda, prms.ParticleVolume);

        constexpr float offset = 0.5f;  // 0 for cubic; 0.5 for quadratic
        const int i0 = (int)(p.pos[0]/h - offset);
        const int j0 = (int)(p.pos[1]/h - offset);

        for (int i = i0; i < i0+3; i++)
            for (int j = 0; j < j0+3; j++)
            {
                int idx_gridnode = i + j*prms.GridX;
                if(i < 0 || j < 0 || i >=prms.GridX || j>=prms.GridY || idx_gridnode < 0 || idx_gridnode >= grid.size())
                {
                    spdlog::critical("point {} in cell [{}, {}]", pt_idx, i, j);
                    throw std::runtime_error("particle is out of grid bounds");
                }

                Eigen::Vector2f pos_node(i*h, j*h);
                Eigen::Vector2f d = p.pos - pos_node;
                float Wip = Point::wq(d, h);   // weight
                Eigen::Vector2f dWip = Point::gradwq(d, h);    // weight gradient

                // APIC increments
                float incM = Wip * prms.ParticleMass;
                Eigen::Vector2f incV = incM * (p.velocity + Dp_inv * p.Bp * (-d));
                Eigen::Vector2f incFi = Ap * dWip;

                // Udpate mass, velocity and force
                GridNode &gn = grid[idx_gridnode];
                #pragma omp atomic
                gn.mass += incM;

                #pragma omp atomic
                gn.velocity[0] += incV[0];
                #pragma omp atomic
                gn.velocity[1] += incV[1];

                #pragma omp atomic
                gn.force[0] += incFi[0];
                #pragma omp atomic
                gn.force[1] += incFi[1];
            }
    }
}


void icy::Model::G2P()
{
    visual_update_mutex.lock();
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; g2p", currentStep);
    const float &dt = prms.InitialTimeStep;
    const float &h = prms.cellsize;
    constexpr float offset = 0.5f;  // 0 for cubic

#pragma omp parallel for
    for(int idx_p = 0; idx_p<points.size(); idx_p++)
    {
        icy::Point &p = points[idx_p];
        p.velocity.setZero();
        p.Bp.setZero();

        const int i0 = (int)((p.pos[0])/h - offset);
        const int j0 = (int)((p.pos[1])/h - offset);
        const Eigen::Vector2f pointPos_copy = p.pos;
        p.pos.setZero();

        Eigen::Matrix2f T;
        T.setZero();

        for (int i = i0; i < i0+3; i++)
            for (int j = j0; j < j0+3; j++)
            {
                int idx_gridnode = i + j*prms.GridX;
                const icy::GridNode &node = grid[idx_gridnode];

                Eigen::Vector2f pos_node(i*prms.cellsize, j*prms.cellsize);
                Eigen::Vector2f d = pointPos_copy - pos_node;   // dist
                float Wip = Point::wq(d, prms.cellsize);   // weight
                Eigen::Vector2f dWip = Point::gradwq(d, prms.cellsize);    // weight gradient

                p.velocity += Wip * node.velocity;
                p.Bp += Wip *(node.velocity*(-d).transpose());
                // Update position and nodal deformation
                p.pos += Wip * (pos_node + dt * node.velocity);
                T += node.velocity * dWip.transpose();
            }
        p.NACCUpdateDeformationGradient(dt,T,prms);
//        p.SnowUpdateDeformationGradient(dt,prms.THT_C_snow,prms.THT_S_snow,T);
//        p.ElasticUpdateDeformationGradient(dt,T);

    }
    visual_update_mutex.unlock();
}


void icy::Model::UpdateNodes()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; update nodes", currentStep);

    const float dt = prms.InitialTimeStep;
    const Eigen::Vector2f gravity(0,-prms.Gravity);
    const float indRsq = prms.IndDiameter*prms.IndDiameter/4.f;
    const Eigen::Vector2f vco(prms.IndVelocity,0);  // velocity of the collision object (indenter)
    const Eigen::Vector2f indCenter(indenter_x, indenter_y);

#pragma omp parallel for schedule (dynamic)
    for (int idx = 0; idx < grid.size(); idx++)
    {
        GridNode &gn = grid[idx];
        if(gn.mass == 0) continue;
        gn.velocity = gn.velocity/gn.mass + dt*(-gn.force/gn.mass + gravity);

        int idx_x = idx % prms.GridX;
        int idx_y = idx / prms.GridX;

        // indenter
        Eigen::Vector2f gnpos(idx_x * prms.cellsize,idx_y * prms.cellsize);
        Eigen::Vector2f n = gnpos - indCenter;
        if(n.squaredNorm() < indRsq)
        {
            // grid node is inside the indenter
            Eigen::Vector2f vrel = gn.velocity - vco;
            n.normalize();
            float vn = vrel.dot(n);   // normal component of the velocity
            if(vn < 0)
            {
                Eigen::Vector2f vt = vrel - n*vn;   // tangential portion of relative velocity
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

void icy::Model::Reset()
{
    spdlog::info("icy::Model::Reset()");
    currentStep = 0;
    simulationTime = 0;

    const float &block_length = prms.BlockLength;
    const float &block_height = prms.BlockHeight;
    const float &h = prms.cellsize;

    const float kRadius = sqrt(block_length*block_height/(prms.PointsWanted*(0.5*SimParams::pi)));
    const std::array<float, 2>kXMin{5.0f*h, 2.0f*h};
    const std::array<float, 2>kXMax{5.0f*h+block_length, 2.0f*h+block_height};
    spdlog::info("starting thinks::PoissonDiskSampling");
    std::vector<std::array<float, 2>> prresult = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
    const size_t nPoints = prresult.size();
    points.resize(nPoints);
    prms.PointCountActual = nPoints;
    spdlog::info("finished thinks::PoissonDiskSampling; {} ", nPoints);

    prms.ParticleVolume = block_length*block_height/nPoints;
    prms.ParticleMass = prms.ParticleVolume*prms.Density;
    for(int k = 0; k<nPoints; k++)
    {
        Point &p = points[k];
        p.pos[0] = prresult[k][0];
        p.pos[1] = prresult[k][1];
//        p.velocity.x() = 1.f + (p.pos.y()-1.5)/2;
//        p.velocity.y() = 2.f + (-p.pos.x()-1.5)/2;
        p.velocity.setZero();
        p.Fe.setIdentity();
        p.Fp.setIdentity();
        p.Bp.setZero();
        p.visualized_value = 0;
        p.NACC_alpha_p = prms.NACC_alpha;
    }
    grid.resize(prms.GridX*prms.GridY);
    indenter_y = block_height + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    indenter_x = indenter_x_initial = 5*h - prms.IndDiameter/2 - h;
    spdlog::info("icy::Model::Reset() done");
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
}


bool icy::Model::Step()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info(" ");
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("step {} started", currentStep);

    indenter_x = indenter_x_initial + simulationTime*prms.IndVelocity;

    ResetGrid();
    P2G();
    if(abortRequested) return false;
    UpdateNodes();
    if(abortRequested) return false;
    G2P();
    if(abortRequested) return false;

    currentStep++;
    simulationTime += prms.InitialTimeStep;
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("step {} completed\n", currentStep);

    if(currentStep % prms.UpdateEveryNthStep == 0) Q_EMIT stepCompleted();

    if(simulationTime >= prms.SimulationEndTime) return false;
    return true;
}



void icy::Model::ResetGrid()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; reset grid", currentStep);
    memset(grid.data(), 0, grid.size()*sizeof(icy::GridNode));
    // for(int i =0;i<grid.size();i++) grid[i].Reset();
}

