#ifndef GPU_IMPLEMENTATION2_H
#define GPU_IMPLEMENTATION2_H


#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"

#include <Eigen/Core>

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void v2_kernel_p2g(const int nPoints);
__global__ void v2_kernel_g2p(const int nPoints);
__global__ void v2_kernel_update_nodes(const int nGridNodes, real indenter_x, real indenter_y);


__device__ Matrix2r polar_decomp_R(const Matrix2r &val);
__device__ void svd(const real a[4], real u[4], real sigma[2], real v[4]);
__device__ void svd2x2(const Matrix2r &mA, Matrix2r &mU, Matrix2r &mS, Matrix2r &mV);

__device__ void NACCUpdateDeformationGradient(icy::Point &p);

// Naive GPU Implementation with memory coalescing

class GPU_Implementation2
{
public:
    GPU_Implementation2();

    void cuda_update_constants(const icy::SimParams &prms);
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void cuda_reset_grid(size_t nGridNodes);
    void transfer_ponts_to_device(const std::vector<icy::Point> &points);
    void cuda_transfer_from_device(std::vector<icy::Point> &points);
    void cuda_p2g(const int nPoints);
    void cuda_g2p(const int nPoints);
    void cuda_update_nodes(const int nGridNodes, real indenter_x, real indenter_y);
    void cuda_device_synchronize();

    void start_timing();
    float end_timing();

private:
    constexpr static int threadsPerBlock = 128;

    Vector2r *_gpu_pts_pos, *_gpu_pts_velocity;
    real *_gpu_pts_Bp[4], *_gpu_pts_Fe[4];
    real *_gpu_pts_NACC_alpha_p;
    Vector2r *_gpu_grid_momentum, *_gpu_grid_velocity;
    real *_gpu_grid_mass;

    cudaEvent_t start, stop;

    std::vector<Vector2r> tmp1;
    std::vector<real> tmp2;
};

#endif // GPU_IMPLEMENTATION0_H
