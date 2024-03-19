#ifndef GPU_IMPLEMENTATION5_H
#define GPU_IMPLEMENTATION5_H


#include "parameters_sim.h"
#include "point.h"

#include <Eigen/Core>
#include <Eigen/LU>


#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>


__global__ void v2_kernel_p2g();
__global__ void v2_kernel_g2p(bool recordPQ);
__global__ void v2_kernel_update_nodes(double indenter_x, double indenter_y);


__device__ Eigen::Matrix2d polar_decomp_R(const Eigen::Matrix2d &val);
__device__ void svd(const double a[4], double u[4], double sigma[2], double v[4]);
__device__ void svd2x2_modified(const Eigen::Matrix2d &mA, Eigen::Matrix2d &mU, Eigen::Vector2d &mS, Eigen::Matrix2d &mV);

__device__ void Wolper_Drucker_Prager(icy::Point &p);
__device__ void CheckIfPointIsInsideFailureSurface(icy::Point &p);
__device__ Eigen::Matrix2d KirchhoffStress_Wolper(const Eigen::Matrix2d &F);

__device__ void ComputePQ(icy::Point &p, const double &kappa, const double &mu);

__device__ Eigen::Vector2d dev_d(Eigen::Vector2d Adiag);

// Naive GPU Implementation with memory coalescing
namespace icy { class Model; }

class GPU_Implementation5
{
public:
    icy::Model *model;
    int error_code;
    std::function<void()> transfer_completion_callback;

    void initialize();
    void test();
    void synchronize(); // call before terminating the main thread
    void cuda_update_constants();
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void cuda_reset_grid();
    void transfer_ponts_to_device();
    void cuda_p2g();
    void cuda_g2p(bool recordPQ);
    void cuda_update_nodes(double indenter_x, double indenter_y);
    void cuda_reset_indenter_force_accumulator();

    void cuda_transfer_from_device();

    cudaEvent_t eventCycleStart, eventCycleStop;

    double *tmp_transfer_buffer = nullptr; // buffer in page-locked memory for transferring the data between device and host
    double *host_side_indenter_force_accumulator = nullptr;

private:

    cudaStream_t streamCompute;
    bool initialized = false;

    static void CUDART_CB callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif
