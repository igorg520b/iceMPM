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
#include <string>

#include "parameters_sim.h"
#include "point.h"
#include "poisson_disk_sampling.h"
#include "gpu_implementation3_sand.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>



namespace icy { class Model; }

class icy::Model
{
public:
    Model(){prms.Reset(); gpu.prms = &prms;};
    void Reset();
    void ResetToStep0();
    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void RequestAbort() {abortRequested = true;}   // asynchronous stop

    void FinalizeDataTransfer();
    void UnlockCycleMutex();

    icy::SimParams prms;
    GPU_Implementation3 gpu;
    float compute_time_per_cycle;

    std::vector<Point> points;

    std::mutex hostside_data_update_mutex; // locks "points" and "grid" vectors
    std::mutex processing_current_cycle_data; // locked until the current cycle results' are copied to host and processed

private:
    void ResetGrid();
    void P2G();
    void UpdateNodes();
    void G2P();

    bool abortRequested;
};

#endif
