#ifndef MODEL_H
#define MODEL_H

#include <QFileInfo>
#include <QObject>
#include <QMutex>
#include <QThread>

#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>

#include "parameters_sim.h"
#include "modelcontrollerinterface.h"
#include "point.h"
#include "gridnode.h"

#include <Eigen/Core>
#include <spdlog/spdlog.h>

namespace icy { class Model; }

class icy::Model : public QObject, public ModelControllerInterface
{
    Q_OBJECT

    // ModelController
public:
    Model() {Reset();}
    void Reset();
    void Prepare() override;        // invoked once, at simulation start
    bool Step() override;           // either invoked by Worker or via GUI
    void RequestAbort() override {abortRequested = true;};   // invoked from GUI

private:
    bool abortRequested;
    constexpr static int colWidth = 12;    // table column width when logging

Q_SIGNALS:
    void stepCompleted();

    // Model
public:
    SimParams prms;

    int currentStep;
    double simulationTime;
    double h; // grid spacing
    int grid_x, grid_y;

    std::vector<Point> points;
    std::vector<GridNode> grid;

    QMutex visual_update_mutex; // to prevent modifying mesh data while updating VTK representation
    bool visual_update_requested = false;  // true when signal has been already emitted to update vtk geometry

private:
    void ResetGrid();
    void P2G();
    void UpdateNodes();
    void G2P();
    void ParticleAdvection();

    // helper functions
    std::pair<int,int> PosToGrid(Eigen::Vector2f position);
    Eigen::Matrix2f polar_decomp_R(const Eigen::Matrix2f &val) const;
};

#endif
