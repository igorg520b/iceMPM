#ifndef GPU_IMPLEMENTATION0_H
#define GPU_IMPLEMENTATION0_H


#include "helper_math.cuh"
#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"

#include <Eigen/Core>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ Eigen::Matrix2f polar_decomp_R(const Eigen::Matrix2f &val);
__device__ float wqs(float x);
__device__ float dwqs(float x);
__device__ float wq(Eigen::Vector2f dx, double h);
__device__ Eigen::Vector2f gradwq(Eigen::Vector2f dx, double h);
__global__ void kernel_p2g(const int nPoints);
__device__ void NACCUpdateDeformationGradient(icy::Point &p, Eigen::Matrix2f &FModifier);
__global__ void kernel_g2p(const int nPoints);
__global__ void kernel_update_nodes(const int nGridNodes, float indenter_x, float indenter_y);
__global__ void cuda_hello(Eigen::Matrix2f A, Eigen::Matrix2f *result);

__device__ void svd(const float a[4], float u[4], float sigma[2], float v[4]);
__device__ void svd2x2(const Eigen::Matrix2f &mA, Eigen::Matrix2f &mU, Eigen::Matrix2f &mS, Eigen::Matrix2f &mV);


// Naive GPU Implementation

class GPU_Implementation0
{
public:
    GPU_Implementation0();

    void test_cuda();
    void cuda_update_constants(const icy::SimParams &prms);
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void transfer_ponts_to_device(size_t nPoints, void* hostSource);
    void cuda_reset_grid(size_t nGridNodes);
    void cuda_transfer_from_device(size_t nPoints, void *hostArray);
    void cuda_p2g(const int nPoints);
    void cuda_g2p(const int nPoints);
    void cuda_update_nodes(const int nGridNodes,float indenter_x, float indenter_y);
    void cuda_device_synchronize();

    void start_timing();
    float end_timing();

private:
    constexpr static int threadsPerBlock = 256;
    icy::Point *gpu_points_ = nullptr;
    icy::GridNode *gpu_nodes_ = nullptr;
    cudaEvent_t start, stop;

};

#endif // GPU_IMPLEMENTATION0_H
