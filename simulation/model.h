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

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include "poisson_disk_sampling.h"


void test_cuda();
void cuda_update_constants(const icy::SimParams &prms);
void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
void transfer_ponts_to_device(size_t nPoints, void* hostSource);
void cuda_reset_grid(size_t nGridNodes);
void cuda_transfer_from_device(size_t nPoints, void *hostArray);
void cuda_p2g(const int nPoints);
void cuda_g2p(const int nPoints);
void cuda_update_nodes(const int nGridNodes,float indenter_x, float indenter_y);

namespace icy { class Model; }

class icy::Model
{
    // ModelController
public:
    Model();
    void Reset();
    void Prepare();        // invoked once, at simulation start
    bool Step();           // either invoked by Worker or via GUI
    void RequestAbort() {abortRequested = true;}   // asynchronous stop

private:

    // Model
public:
    icy::SimParams prms;

    float indenter_x, indenter_x_initial, indenter_y;

    std::vector<Point> points;
    std::vector<GridNode> grid;

    std::mutex visual_update_mutex; // to prevent modifying mesh data while updating VTK representation
    bool visual_update_requested = false;  // true when signal has been already emitted to update vtk geometry

    bool isTimeToUpdate() { return prms.SimulationStep % prms.UpdateEveryNthStep == 0;}
private:
    void ResetGrid();
    void P2G();
    void UpdateNodes();
    void G2P();


    bool abortRequested;
};

#endif
