#ifndef GPU_IMPLEMENTATION3_H
#define GPU_IMPLEMENTATION3_H


#include "parameters_sim.h"
#include "point.h"

#include <Eigen/Core>
#include <Eigen/LU>


#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>


__global__ void v2_kernel_p2g();
__global__ void v2_kernel_g2p();
__global__ void v2_kernel_update_nodes(real indenter_x, real indenter_y);


__device__ Matrix2r polar_decomp_R(const Matrix2r &val);
__device__ void svd(const real a[4], real u[4], real sigma[2], real v[4]);
__device__ void svd2x2(const Matrix2r &mA, Matrix2r &mU, Matrix2r &mS, Matrix2r &mV);


__device__ void NACCUpdateDeformationGradient_q_hardening_2(icy::Point &p);
__device__ void NACCUpdateDeformationGradient_q_hardening(icy::Point &p);
//__device__ void NACCUpdateDeformationGradient_Alt(icy::Point &p);

//__device__ void NACCUpdateDeformationGradient(icy::Point &p);
//__device__ void DruckerPragerUpdateDeformationGradient(icy::Point &p);
//__device__ void SnowUpdateDeformationGradient(icy::Point &p);

__device__ Matrix2r dev(Matrix2r A);
__device__ Matrix2r KirchhoffStress_Wolper(const Matrix2r &F);


__device__ double clamp(double x, double a, double b);

// Naive GPU Implementation with memory coalescing

class GPU_Implementation3
{
public:
    icy::SimParams *prms;
    int error_code;
    std::function<void()> transfer_completion_callback;

    void initialize();
    void test();
    void synchronize(); // call before terminating the main thread
    void cuda_update_constants();
    void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints);
    void cuda_reset_grid();
    void transfer_ponts_to_device(const std::vector<icy::Point> &points);
    void cuda_p2g();
    void cuda_g2p();
    void cuda_update_nodes(real indenter_x, real indenter_y);

    void cuda_transfer_from_device();
    void transfer_ponts_to_host_finalize(std::vector<icy::Point> &points);

    cudaEvent_t eventCycleStart, eventCycleStop;

    real *tmp_transfer_buffer = nullptr; // buffer in page-locked memory for transferring the data between device and host

private:

    cudaStream_t streamCompute;
    bool initialized = false;

    static void CUDART_CB callback_transfer_from_device_completion(cudaStream_t stream, cudaError_t status, void *userData);
};

#endif // GPU_IMPLEMENTATION0_H
