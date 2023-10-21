#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <random>
#include <mutex>
#include <iostream>

#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"
#include "poisson_disk_sampling.h"
#include "gpu_implementation2.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>



namespace icy { class Model; }

class icy::Model
{
public:
    Model();
    void Reset();
    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void RequestAbort() {abortRequested = true;}   // asynchronous stop

private:
    GPU_Implementation2 gpu;

public:
    icy::SimParams prms;
    float compute_time_per_cycle;

    real indenter_x, indenter_x_initial, indenter_y;

    std::vector<Point> points;
    std::vector<GridNode> grid;

    std::mutex visual_update_mutex; // to prevent modifying mesh data while updating VTK representation

    bool isTimeToUpdate() { return prms.SimulationStep % prms.UpdateEveryNthStep == 0;}
private:
    void ResetGrid();
    void P2G();
    void UpdateNodes();
    void G2P();

    bool abortRequested;
};

#endif
