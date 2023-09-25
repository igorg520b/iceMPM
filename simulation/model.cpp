#include "model.h"
#include <spdlog/spdlog.h>





void icy::Model::P2G()
{
    if(isTimeToUpdate()) spdlog::info("s {}; p2g", prms.SimulationStep);

    const float &h = prms.cellsize;
//    const float Dp_inv = 3.f/(h*h); // cubic

#pragma omp parallel for
    for(int pt_idx=0; pt_idx<points.size(); pt_idx++)
    {
        Point &p = points[pt_idx];

        Eigen::Matrix2f Ap;
        //Ap = p.NACCConstitutiveModel(prms.mu, prms.lambda, prms.ParticleVolume);
        //Ap = p.SnowConstitutiveModel(prms.XiSnow, prms.mu, prms.lambda, prms.ParticleVolume);
        Ap = p.ElasticConstitutiveModel(prms.mu, prms.lambda, prms.ParticleVolume);


        // TODO: this is unfinished
        constexpr float offset = 0.5f;  // 0 for cubic; 0.5 for quadratic
        const int i0 = (int)((p.pos[0])/h - offset);
        const int j0 = (int)((p.pos[1])/h - offset);

        for (int i = i0; i < i0+3; i++)
            for (int j = j0; j < j0+3; j++)
            {
                int idx_gridnode = i + j*prms.GridX;
                if(i < 0 || j < 0 || i >=prms.GridX || j>=prms.GridY || idx_gridnode < 0 || idx_gridnode >= grid.size())
                {
                    spdlog::critical("point {} in cell [{}, {}]", pt_idx, i, j);
                    throw std::runtime_error("particle is out of grid bounds");
                }

                Eigen::Vector2f pos_node(i, j);
                Eigen::Vector2f d = p.pos/h - pos_node;
                float Wip = Point::wq(d);   // weight
                Eigen::Vector2f dWip = Point::gradwq(d);    // weight gradient

                // APIC increments
                float incM = Wip * prms.ParticleMass;
                Eigen::Vector2f incV = incM * (p.velocity - p.Bp * (d*4.f/h));
                Eigen::Vector2f incFi = Ap * dWip/h;

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
    if(isTimeToUpdate()) spdlog::info("s {}; g2p", prms.SimulationStep);

    const float &dt = prms.InitialTimeStep;
    const float &h = prms.cellsize;
    constexpr float offset = 0.5f;  // 0 for cubic

    visual_update_mutex.lock();
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

                Eigen::Vector2f pos_node(i, j);
                Eigen::Vector2f d = pointPos_copy/h - pos_node;   // dist
                float Wip = Point::wq(d);   // weight
                Eigen::Vector2f dWip = Point::gradwq(d);    // weight gradient

                p.velocity += Wip * node.velocity;
                p.Bp += Wip *(node.velocity*(-d*h).transpose());
                // Update position and nodal deformation
                p.pos += Wip * (pos_node*h + dt * node.velocity);
                T += node.velocity * dWip.transpose()/h;
            }
//        p.NACCUpdateDeformationGradient(dt,T,prms);
//        p.SnowUpdateDeformationGradient(dt,prms.THT_C_snow,prms.THT_S_snow,T);
        p.ElasticUpdateDeformationGradient(dt,T);

    }
    visual_update_mutex.unlock();
}


void icy::Model::UpdateNodes()
{
    if(isTimeToUpdate()) spdlog::info("s {}; update nodes", prms.SimulationStep);

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
    // this should be called after prms are set as desired (either via GUI or CLI)
    spdlog::info("icy::Model::Reset()");

    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    compute_time_per_cycle = 0;

    const float &block_length = prms.BlockLength;
    const float &block_height = prms.BlockHeight;
    const float &h = prms.cellsize;

    const float kRadius = sqrt(block_length*block_height/(prms.PointsWanted*(0.5*SimParams::pi)*100./97.));
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
        //p.Fp.setIdentity();
        p.Bp.setZero();
        //p.visualized_value = 0;
        p.NACC_alpha_p = prms.NACC_alpha;
    }
    grid.resize(prms.GridX*prms.GridY);
    indenter_y = block_height + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    indenter_x = indenter_x_initial = 5*h - prms.IndDiameter/2 - h;

    prms.MemAllocGrid = (float)sizeof(GridNode)*grid.size()/(1024*1024);
    prms.MemAllocPoints = (float)sizeof(Point)*points.size()/(1024*1024);
    prms.MemAllocTotal = prms.MemAllocGrid + prms.MemAllocPoints;
    spdlog::info("icy::Model::Reset(); grid {:03.2f} Mb; points {:03.2f} Mb ; total {:03.2f} Mb",
                 prms.MemAllocGrid, prms.MemAllocPoints, prms.MemAllocTotal);

    gpu.cuda_allocate_arrays(grid.size(), points.size());
    gpu.transfer_ponts_to_device(points);
    spdlog::info("icy::Model::Reset() done");
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    gpu.cuda_update_constants(prms);
    abortRequested = false;
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

icy::Model::Model()
{
    prms.Reset();
    spdlog::info("num threads {}", omp_get_max_threads());
    int nthreads, tid;
#pragma omp parallel
    { spdlog::info("thread {}", omp_get_thread_num()); }
    std::cout << std::endl;
    spdlog::info("sizeof(Point) = {}", sizeof(icy::Point));
    spdlog::info("sizeof(GridNode) = {}", sizeof(icy::GridNode));

}
