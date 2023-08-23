#include <vtkPointData.h>
#include <QtGlobal>
#include "model.h"
#include "spdlog/spdlog.h"


#include <random>

void icy::Model::Reset()
{
    currentStep = 0;
    simulationTime = 0;

    spdlog::info("icy::Model::Reset() done");

    std::default_random_engine generator;

    grid_x = grid_y = 64;
    h = 3/grid_x;   // 3 meters across
    int pts_x = 100, pts_y=100;
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
        }
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
}


bool icy::Model::Step()
{
    QThread::msleep(500);
    currentStep++;
    spdlog::info("step {}", currentStep);

    Q_EMIT stepCompleted();
    return true;
}




