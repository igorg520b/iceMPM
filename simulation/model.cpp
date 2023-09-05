#include <vtkPointData.h>
#include <QtGlobal>
#include "model.h"
#include "spdlog/spdlog.h"
#include <random>

bool icy::Model::Step()
{
    ResetGrid();
    /*
    reset();

    // particle to grid
    p2gTransfer();
    computeGridVelocities();

    // updates - force, velocity, DG
    computeForces();
    velocityUpdate();
    updateDGs();

    // grid to particle
    g2pTransfer();
    particleAdvection();
*/



    QThread::msleep(500);
    currentStep++;
    spdlog::info("step {}", currentStep);

    Q_EMIT stepCompleted();
    return true;
}



void icy::Model::ResetGrid()
{
    spdlog::info("s {}; reset grid", currentStep);

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


void icy::Model::P2GTransfer()
{
    spdlog::info("s {}; p2g", currentStep);
#pragma omp parallel for
    for(int i=0; i<points.size(); i++)
    {
        Point &p = points[i];
        auto [idx_x, idx_y] = PosToGrid(p.pos);

    }

    /*
void ParticleGrid::computeWeights() {
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);

        p->weight = weightFunc(p->pos, loc);
        p->weightGrad = weightGradFunc(p->pos, loc);
    }
}
*/

    /*
    // pre pass - compute particle weights
    computeWeights();

//    std::cout << "p2g ";
    for (int i = 0; i < sampler->finalSamples.size(); i++) {
        Particle* p = &(sampler->finalSamples[i]);
        glm::vec3 loc = posToGrid(p->pos);
        glm::vec3 gridPos = gridToPos(loc.x, loc.y, loc.z);

        // weight (wip)
        float wip = p->weight;
        glm::mat3 Dp = calcDp(p->pos, gridPos);

        // updates - mass and momentum
        float currMass = wip * p->mass;
        gridMasses[loc[0]][loc[1]][loc[2]] += currMass;

        glm::vec3 currMom = wip * p->mass * (p->vp + (p->Bp * Dp * (gridPos - p->pos)));
        gridMomentums[loc[0]][loc[1]][loc[2]] += currMom;
    }
*/

}



void icy::Model::ComputeGridVelocities() {}
void icy::Model::ComputeForces() {}
void icy::Model::VelocityUpdates() {}
void icy::Model::UpdateDGs() {}
void icy::Model::G2PTransfer() {}
void icy::Model::ParticleAdvection() {}


void icy::Model::Reset()
{
    currentStep = 0;
    simulationTime = 0;

    spdlog::info("icy::Model::Reset() done");

    std::default_random_engine generator;

    grid_x = prms.GridX;
    grid_y = prms.GridY;
    h = prms.cellsize;   // 3 meters across
    int pts_x = 100, pts_y=100;
    float vol = grid_x*h*grid_y*h/(pts_x*pts_y);
    std::normal_distribution<double> distribution(0,1./(pts_x*100));

    points.resize(pts_x*pts_y);
    for(int i=0;i<pts_x;i++)
        for(int j=0;j<pts_y;j++)
        {
            Point &p = points[i+j*pts_x];
            float x = (float)i/(float)pts_x + distribution(generator);// + 1.;
            float y = (float)j/(float)pts_y + distribution(generator);// + 1.;
            p.pos = Eigen::Vector2f(x,y);
            p.velocity = Eigen::Vector2f::Zero();
            p.Fe = p.Fp = Eigen::Matrix2f::Identity();
            p.volume = vol;
        }
    grid.resize(grid_x*grid_y);
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
}




