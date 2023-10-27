#ifndef GPU_IMPLEMENTATION2_H
#define GPU_IMPLEMENTATION2_H


#include "parameters_sim.h"
#include "point.h"

#include <Eigen/Core>

#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>


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
    constexpr static int nGridArrays = 3, nPtsArrays = 16;
    int error_code;
    std::function<void()> transfer_completion_callback;

    void test();
    void synchronize(); // call before terminating the main thread
    void cuda_update_constants(const icy::SimParams &prms);
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void cuda_reset_grid(size_t nGridNodes);
    void transfer_ponts_to_device(const std::vector<icy::Point> &points);
    void cuda_p2g(const int nPoints);
    void cuda_g2p(const int nPoints);
    void cuda_update_nodes(const int nGridNodes, real indenter_x, real indenter_y);
    void backup_point_positions(const int nPoints);

    void cuda_transfer_from_device(std::vector<icy::Point> &points);
    void transfer_ponts_to_host_finalize(std::vector<icy::Point> &points);

    cudaEvent_t eventTimingStart, eventTimingStop, eventCycleComplete, eventDataCopiedToHost;

    void *tmp_transfer_buffer = nullptr; // nPoints*sizeof(real)
private:
    constexpr static int threadsPerBlock = 512;
    real *grid_arrays[nGridArrays], *pts_arrays[nPtsArrays];

    cudaStream_t streamCompute, streamTransfer;

    static void CUDART_CB callback_transfer_from_device_completion(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif // GPU_IMPLEMENTATION0_H
