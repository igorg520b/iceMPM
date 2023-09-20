#ifndef MODEL_H
#define MODEL_H

#include <QFileInfo>
#include <QObject>
#include <QMutex>
#include <QThread>
#include <QtGlobal>

#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <random>

#include "parameters_sim.h"
#include "modelcontrollerinterface.h"
#include "point.h"
#include "gridnode.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <spdlog/spdlog.h>
#include "poisson_disk_sampling.h"


namespace icy { class Model; }

class icy::Model : public QObject, public ModelControllerInterface
{
    Q_OBJECT
    Q_PROPERTY(int iCurrentStep MEMBER currentStep)

    // ModelController
public:
    Model() {Reset();}
    void Reset();
    void Prepare() override;        // invoked once, at simulation start
    bool Step() override;           // either invoked by Worker or via GUI
    void RequestAbort() override {abortRequested = true;}   // invoked from GUI

private:
    bool abortRequested;
    constexpr static int colWidth = 12;    // table column width when logging

Q_SIGNALS:
    void stepCompleted();

    // Model
public:
    icy::SimParams prms;

    int currentStep;
    double simulationTime;

    float indenter_x, indenter_x_initial, indenter_y;

    std::vector<Point> points;
    std::vector<GridNode> grid;

    QMutex visual_update_mutex; // to prevent modifying mesh data while updating VTK representation
    bool visual_update_requested = false;  // true when signal has been already emitted to update vtk geometry

private:
    void ResetGrid();
    void P2G();
    void UpdateNodes();
    void G2P();

    // helper functions
    void PosToGrid(Eigen::Vector2f position, int &idx_x, int &idx_y);
};

#endif
