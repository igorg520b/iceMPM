#include <vtkPointData.h>
#include <QtGlobal>
#include "model.h"
#include "spdlog/spdlog.h"
#include <random>

#include <Eigen/SVD>
#include <Eigen/LU>

bool icy::Model::Step()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info(" ");
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("step {} started", currentStep);
    ResetGrid();
    P2G();
    if(abortRequested) return false;
    UpdateNodes();
    if(abortRequested) return false;
    G2P();
    if(abortRequested) return false;
//    ParticleAdvection();

    currentStep++;
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("step {} completed\n", currentStep);

    if(currentStep % prms.UpdateEveryNthStep == 0) Q_EMIT stepCompleted();
    return true;
}



void icy::Model::ResetGrid()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; reset grid", currentStep);

//#pragma omp parallel for
//    for(int i=0;i<grid.size();i++) grid[i].Reset();

    memset(grid.data(), 0, grid.size()*sizeof(icy::GridNode));
}


std::pair<int,int> icy::Model::PosToGrid(Eigen::Vector2f pos)
{
    float h = prms.cellsize;
    int idx_x = std::clamp((int)(pos[0]/h),0,prms.GridX-1);
    int idx_y = std::clamp((int)(pos[1]/h),0,prms.GridY-1);
    return {idx_x, idx_y};
}



// polar decomposition
// http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
Eigen::Matrix2f icy::Model::polar_decomp_R(const Eigen::Matrix2f &val) const
{
    float th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Eigen::Matrix2f result;
    result << cos(th), -sin(th), sin(th), cos(th);
    return result;
}


void icy::Model::P2G()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; p2g", currentStep);

    const float Dp_inv = 3.f/(prms.cellsize*prms.cellsize);
    const float mu = prms.mu;
    const float lambda = prms.lambda;

#pragma omp parallel for
    for(int pt_idx=0; pt_idx<points.size(); pt_idx++)
    {
        Point &p = points[pt_idx];

        // this is for elastic material only
        Eigen::Matrix2f Re;

//        Eigen::JacobiSVD<Eigen::Matrix2f, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(p.Fe);
//        Re = svd.matrixU()*svd.matrixV();
        Re = polar_decomp_R(p.Fe);
        float Je = p.Fe.determinant();
        Eigen::Matrix2f dFe = 2.f * mu*(p.Fe - Re)* p.Fe.transpose() +
                lambda * (Je - 1.f) * Je * Eigen::Matrix2f::Identity();

        Eigen::Matrix2f Ap = dFe * p.volume;

        auto [i0, j0] = PosToGrid(p.pos);

        for (int di = -1; di < 3; di++)
            for (int dj = -1; dj < 3; dj++)
            {
                int i = i0+di;
                int j = j0+dj;
                int idx_gridnode = i + j*prms.GridX;
                if(idx_gridnode < 0 || idx_gridnode >= points.size())
                {
                    spdlog::critical("point {} in cell {}-{}", pt_idx, i, j);
                    throw std::runtime_error("particle is out of grid bounds");
                }

                Eigen::Vector2f pos_node(i*prms.cellsize, j*prms.cellsize);
                Eigen::Vector2f d = p.pos - pos_node;
                float Wip = Point::wc(d, prms.cellsize);   // weight
                Eigen::Vector2f dWip = Point::gradwc(d, prms.cellsize);    // weight gradient

                // APIC increments
                float incM = Wip * p.mass;
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


void icy::Model::UpdateNodes()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; update nodes", currentStep);

    const float dt = prms.InitialTimeStep;
    const Eigen::Vector2f gravity(0,-prms.Gravity);

#pragma omp parallel for schedule (dynamic)
    for (int idx = 0; idx < grid.size(); idx++)
    {
        GridNode &gn = grid[idx];
        if(gn.mass == 0) continue;
        gn.velocity = gn.velocity/gn.mass + dt*(-gn.force/gn.mass + gravity);

        int idx_x = idx % prms.GridX;
        int idx_y = idx / prms.GridX;

        // basic non-sticky "floor" boundary
        if(idx_y <= 2 && gn.velocity.y()<0) gn.velocity.y() = 0;
        else if(idx_y >= prms.GridY-3 && gn.velocity.y()>0) gn.velocity.y() = 0;

        if(idx_x <= 2 && gn.velocity.x()<0) gn.velocity.x() *= 0;
        else if(idx_x >= prms.GridX-3 && gn.velocity.x()>0) gn.velocity.x() = 0;
    }
}


void icy::Model::G2P()
{
    visual_update_mutex.lock();
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; g2p", currentStep);
    const float dt = prms.InitialTimeStep;

#pragma omp parallel for
    for(int idx_p = 0; idx_p<points.size(); idx_p++)
    {
        icy::Point &p = points[idx_p];
        p.velocity.setZero();
        p.Bp.setZero();

        auto [i0, j0] = PosToGrid(p.pos);
        Eigen::Vector2f pointPos_copy = p.pos;
        p.pos.setZero();

        Eigen::Matrix2f T;
        T.setZero();

        for (int di = -1; di < 3; di++)
            for (int dj = -1; dj < 3; dj++)
            {
                int i = i0+di;
                int j = j0+dj;
                int idx_gridnode = i + j*prms.GridX;
                if(idx_gridnode < 0 || idx_gridnode >= points.size())
                {
                    spdlog::critical("point {} in cell {}-{}", idx_p, i, j);
                    throw std::runtime_error("particle is out of grid bounds");
                }

                const icy::GridNode &node = grid[idx_gridnode];

                Eigen::Vector2f pos_node(i*prms.cellsize, j*prms.cellsize);
                Eigen::Vector2f d = pointPos_copy - pos_node;   // dist
                float Wip = Point::wc(d, prms.cellsize);   // weight
                Eigen::Vector2f dWip = Point::gradwc(d, prms.cellsize);    // weight gradient

                p.velocity += Wip * node.velocity;
                p.Bp += Wip *(node.velocity*(-d).transpose());
                // Update position and nodal deformation
                p.pos += Wip * (pos_node + dt * node.velocity);
                T += node.velocity * dWip.transpose();
            }

        // Update particle deformation gradient (elasticity, plasticity etc...)
        p.Fe = (Eigen::Matrix2f::Identity() + dt*T) * p.Fe;
    }
    visual_update_mutex.unlock();
}


void icy::Model::Reset()
{
    currentStep = 0;
    simulationTime = 0;

    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("icy::Model::Reset() done");

    std::default_random_engine generator;

    grid_x = prms.GridX;
    grid_y = prms.GridY;
    h = prms.cellsize;   // 3 meters across
    const int pts_x = 100, pts_y=100;
    const float vol = grid_x*h*grid_y*h/(pts_x*pts_y);
    std::normal_distribution<double> distribution(0,1./(pts_x*100));

    points.resize(pts_x*pts_y);
    for(int i=0;i<pts_x;i++)
        for(int j=0;j<pts_y;j++)
        {
            Point &p = points[i+j*pts_x];
            float x = (float)i/(float)pts_x + distribution(generator) + 1.;
            float y = (float)j/(float)pts_y + distribution(generator) + 1.;

            p.pos = Eigen::Vector2f(x,y);

            //p.velocity.setZero();
            p.velocity.x() = 1.f + (p.pos.y()-1.5)/2;
            p.velocity.y() = 1.f + (-p.pos.x()-1.5)/2;
            p.Fe.setIdentity();
            p.volume = vol;
            p.mass = vol*prms.Density;
            p.Bp.setZero();
        }
    grid.resize(grid_x*grid_y);
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
}




