#ifndef GPU_IMPLEMENTATION1_H
#define GPU_IMPLEMENTATION1_H


#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"

#include <Eigen/Core>

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void v1_kernel_p2g(const int nPoints);
__global__ void v1_kernel_g2p(const int nPoints);
__global__ void v1_kernel_update_nodes(const int nGridNodes, float indenter_x, float indenter_y);



// Naive GPU Implementation

class GPU_Implementation1
{
public:
    GPU_Implementation1();

    void cuda_update_constants(const icy::SimParams &prms);
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void cuda_reset_grid(size_t nGridNodes);
    void transfer_ponts_to_device(const std::vector<icy::Point> &points);
    void cuda_transfer_from_device(std::vector<icy::Point> &points);
    void cuda_p2g(const int nPoints);
    void cuda_g2p(const int nPoints);
    void cuda_update_nodes(const int nGridNodes,float indenter_x, float indenter_y);
    void cuda_device_synchronize();

    void start_timing();
    float end_timing();

private:
    constexpr static int threadsPerBlock = 256;

    Eigen::Vector2f *_gpu_pts_pos, *_gpu_pts_velocity;
    float *_gpu_pts_Bp[4], *_gpu_pts_Fe[4];
    float *_gpu_pts_NACC_alpha_p;
    Eigen::Vector2f *_gpu_grid_momentum, *_gpu_grid_velocity, *_gpu_grid_force;
    float *_gpu_grid_mass;

    cudaEvent_t start, stop;

    std::vector<Eigen::Vector2f> tmp1;
    std::vector<float> tmp2;


};

#endif // GPU_IMPLEMENTATION0_H
