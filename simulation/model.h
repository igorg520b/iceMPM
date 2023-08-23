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

#include "parameters_sim.h"
#include "modelcontrollerinterface.h"
#include "point.h"

#include <Eigen/Core>

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
    void Aborting() {abortRequested = false; Q_EMIT stepAborted();};       // called before exiting Step() if aborted
    constexpr static int colWidth = 12;    // table column width when logging

Q_SIGNALS:
    void stepCompleted();
    void stepAborted();

    // Model
public:
    SimParams prms;

    int currentStep;
    double simulationTime;
    double h; // grid spacing
    int grid_x, grid_y;

    std::vector<Point> points;

QMutex visual_update_mutex; // to prevent modifying mesh data while updating VTK representation
    bool visual_update_requested = false;  // true when signal has been already emitted to update vtk geometry
};

#endif
