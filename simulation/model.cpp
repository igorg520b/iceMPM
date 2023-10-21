#include "model.h"
#include <spdlog/spdlog.h>


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



void icy::Model::Reset()
{
    // this should be called after prms are set as desired (either via GUI or CLI)
    spdlog::info("icy::Model::Reset()");

    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    compute_time_per_cycle = 0;

    const real &block_length = prms.BlockLength;
    const real &block_height = prms.BlockHeight;
    const real &h = prms.cellsize;

    const real kRadius = sqrt(block_length*block_height/(prms.PointsWanted*(0.5*SimParams::pi)*100./97.));
    const std::array<real, 2>kXMin{5.0*h, 2.0*h};
    const std::array<real, 2>kXMax{5.0*h+block_length, 2.0*h+block_height};
    spdlog::info("starting thinks::PoissonDiskSampling");
    std::vector<std::array<real, 2>> prresult = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
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
        p.Fe.setIdentity();
        p.velocity.setZero();
//        p.velocity.x() = 1.f + (p.pos.y()-1.5)/2;
//        p.velocity.y() = 2.f + (-p.pos.x()-1.5)/2;
        //p.Fp.setIdentity();
        p.Bp.setZero();
        p.NACC_alpha_p = prms.NACC_alpha;
    }
    grid.resize(prms.GridX*prms.GridY);
    indenter_y = block_height + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    indenter_x = indenter_x_initial = 5*h - prms.IndDiameter/2 - h;

    prms.MemAllocGrid = (real)sizeof(GridNode)*grid.size()/(1024*1024);
    prms.MemAllocPoints = (real)sizeof(Point)*points.size()/(1024*1024);
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
