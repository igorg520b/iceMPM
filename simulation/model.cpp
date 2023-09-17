#include "model.h"
#include <vtkPointData.h>
#include <QtGlobal>

#include "spdlog/spdlog.h"
#include "poisson_disk_sampling.h"

#include <cmath>

#include <Eigen/SVD>
#include <Eigen/LU>

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
    return true;
}



void icy::Model::ResetGrid()
{
    if(currentStep % prms.UpdateEveryNthStep == 0) spdlog::info("s {}; reset grid", currentStep);
    memset(grid.data(), 0, grid.size()*sizeof(icy::GridNode));
}

void icy::Model::PosToGrid(Eigen::Vector2f pos, int &idx_x, int &idx_y)
{
    idx_x = std::clamp((int)(pos[0]/prms.cellsize),0,prms.GridX-1);
    idx_y = std::clamp((int)(pos[1]/prms.cellsize),0,prms.GridY-1);
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
        Eigen::Matrix2f Re = polar_decomp_R(p.Fe);
        float Je = p.Fe.determinant();
        Eigen::Matrix2f dFe = 2.f * mu*(p.Fe - Re)* p.Fe.transpose() +
                lambda * (Je - 1.f) * Je * Eigen::Matrix2f::Identity();

        Eigen::Matrix2f Ap = dFe * particle_volume;

        int i0, j0;
        PosToGrid(p.pos, i0, j0);

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
                float incM = Wip * particle_mass;
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

        // attached bottom layer
        if(idx_y <= 3) gn.velocity.setZero();
        else if(idx_y >= prms.GridY-3 && gn.velocity[1]>0) gn.velocity[1] = 0;

        if(idx_x <= 2 && gn.velocity.x()<0) gn.velocity[0] = 0;
        else if(idx_x >= prms.GridX-3 && gn.velocity[0]>0) gn.velocity[0] = 0;

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

        int i0, j0;
        PosToGrid(p.pos, i0, j0);
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
    spdlog::info("icy::Model::Reset()");
    currentStep = 0;
    simulationTime = 0;

    const float &block_length = prms.BlockLength;
    const float &block_height = prms.BlockHeight;
    const float &h = prms.cellsize;

    const float kRadius = sqrt(block_length*block_height/(prms.PointsWanted*(0.5*M_PI)));
    const std::array<float, 2>kXMin{5.0f*h, 2.0f*h};
    const std::array<float, 2>kXMax{5.0f*h+block_length, 2.0f*h+block_height};
    spdlog::info("starting thinks::PoissonDiskSampling");
    std::vector<std::array<float, 2>> prresult = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
    const size_t nPoints = prresult.size();
    points.resize(nPoints);
    prms.PointCountActual = nPoints;
    spdlog::info("finished thinks::PoissonDiskSampling; {} ", nPoints);

    particle_volume = block_length*block_height/nPoints;
    for(int k = 0; k<nPoints; k++)
    {
        Point &p = points[k];
        p.pos[0] = prresult[k][0];
        p.pos[1] = prresult[k][1];
        p.velocity.x() = 1.f + (p.pos.y()-1.5)/2;
        p.velocity.y() = 2.f + (-p.pos.x()-1.5)/2;
        p.Fe.setIdentity();
        p.Bp.setZero();
    }
    this->particle_mass = particle_volume*prms.Density;

    const int &grid_x = prms.GridX;
    const int &grid_y = prms.GridY;
    grid.resize(grid_x*grid_y);

    indenter_y = block_height + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    indenter_x = indenter_x_initial = 5*h - prms.IndDiameter/2 - h;

    spdlog::info("icy::Model::Reset() done");
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
}
