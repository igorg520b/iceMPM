#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/LU>


#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"
#include "helper_math.cuh"

__constant__ float mu, lambda, kappa, xi, beta, M_sq, particle_volume, cellsize, Dp_inv, particle_mass, dt;
__constant__ int gridX, gridY;
__constant__ bool hardening;

__device__ icy::Point *gpu_points;
__device__ icy::GridNode *gpu_nodes;
icy::Point *gpu_points_;
icy::GridNode *gpu_nodes_;
__device__ int gpu_error_indicator;

void cuda_update_constants(const icy::SimParams &prms)
{
    cudaMemcpyToSymbol(&mu, &prms.mu, sizeof(float));
    cudaMemcpyToSymbol(&lambda, &prms.lambda, sizeof(float));
    cudaMemcpyToSymbol(&kappa, &prms.kappa, sizeof(float));
    cudaMemcpyToSymbol(&xi, &prms.NACC_xi, sizeof(float));
    cudaMemcpyToSymbol(&beta, &prms.NACC_beta, sizeof(float));
    cudaMemcpyToSymbol(&M_sq, &prms.NACC_M_sq, sizeof(float));
    cudaMemcpyToSymbol(&particle_volume, &prms.ParticleVolume, sizeof(float));
    cudaMemcpyToSymbol(&cellsize, &prms.cellsize, sizeof(float));
    cudaMemcpyToSymbol(&particle_mass, &prms.ParticleMass, sizeof(float));
    cudaMemcpyToSymbol(&dt, &prms.InitialTimeStep, sizeof(float));

    cudaMemcpyToSymbol(&gridX, &prms.GridX, sizeof(int));
    cudaMemcpyToSymbol(&gridY, &prms.GridY, sizeof(int));

    const float host_Dp_inv = 4.f/(prms.cellsize*prms.cellsize); // quadratic
    cudaMemcpyToSymbol(&Dp_inv, &host_Dp_inv, sizeof(float));
    cudaMemcpyToSymbol(&hardening, &prms.NACC_hardening, sizeof(bool));



    spdlog::info("CUDA constants copied to device");
}

void cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
//    cudaFree((void)gpu_error_indicator);

    // TODO: free when needed
    //cudaFree((void*)gpu_points);
    //cudaFree((void*)gpu_nodes);

    cudaError_t err;

//    cudaMallocManaged((void**)&gpu_error_indicator,sizeof(int));

    err = cudaMalloc(&gpu_points_, sizeof(icy::Point)*nPoints);
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_allocate_arrays can't allocate");
        throw std::runtime_error("cuda_allocate_arrays");
    }

    err = cudaMemcpyToSymbol(gpu_points, &gpu_points_, sizeof(gpu_points_));
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_allocate_arrays cudaMemcpyToSymbol error");
        throw std::runtime_error("cuda_allocate_arrays");
    }

    err = cudaMalloc(&gpu_nodes_, sizeof(icy::GridNode)*nGridNodes);
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_allocate_arrays can't allocate");
        throw std::runtime_error("cuda_allocate_arrays");
    }
    err = cudaMemcpyToSymbol(gpu_nodes, &gpu_nodes_, sizeof(gpu_nodes_));
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_allocate_arrays cudaMemcpyToSymbol error");
        throw std::runtime_error("cuda_allocate_arrays");
    }
    spdlog::info("cuda_allocate_arrays done");

}

void transfer_ponts_to_device(size_t nPoints, void* hostSource)
{
    cudaError_t err;
    err = cudaMemcpy(gpu_points_, hostSource, nPoints*sizeof(icy::Point), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        spdlog::critical("transfer_ponts_to_device failed with code {}",err);
        spdlog::critical("gpu_points_ {}",(void*)gpu_points_);
        spdlog::critical("hostsource {}", (void*)hostSource);
        spdlog::critical("size {}",nPoints*sizeof(icy::Point));
        throw std::runtime_error("transfer_ponts_to_device");
    }
    spdlog::info("transfer_ponts_to_device done");
}

void cuda_transfer_from_device(size_t nPoints, void *hostArray)
{
    cudaError_t err;
    err = cudaMemcpy(hostArray, gpu_points_, nPoints*sizeof(icy::Point), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_transfer_from_device failed");
        throw std::runtime_error("cuda_transfer_from_device");
    }
    spdlog::info("cuda_transfer_from_device");
}

void cuda_reset_grid(size_t nGridNodes)
{
    cudaError_t err = cudaMemset(gpu_nodes, 0, sizeof(icy::GridNode)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid memset error");
    spdlog::info("cuda_reset_grid done");
}


__device__ Eigen::Matrix2f polar_decomp_R(const Eigen::Matrix2f &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    float th = atan2f(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Eigen::Matrix2f result;
    result << cosf(th), -sinf(th), sinf(th), cosf(th);
    return result;
}



__device__ float wqs(float x)
{
    x = fabsf(x);
    if (x < 0.5f) return -x * x + 3 / 4.0f;
    else if (x < 1.5f) return x * x / 2.0f - 3 * x / 2.0f + 9 / 8.0f;
    return 0;
}

__device__ float dwqs(float x)
{
    float x_abs = fabsf(x);
    if (x_abs < 0.5f) return -2.0f * x;
    else if (x_abs < 1.5f) return x - 3 / 2.0f * x / x_abs;
    return 0;
}

__device__ float wq(Eigen::Vector2f dx, double h)
{
    return wqs(dx[0]/h)*wqs(dx[1]/h);
}

__device__ Eigen::Vector2f gradwq(Eigen::Vector2f dx, double h)
{
    Eigen::Vector2f result;
    result[0] = dwqs(dx[0]/h)*wqs(dx[1]/h)/h;
    result[1] = wqs(dx[0]/h)*dwqs(dx[1]/h)/h;
    return result;
}


__global__ void kernel_p2g(const int nPoints)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPoints) return;

    icy::Point &p = gpu_points[pt_idx];
//    const float &h = cellsize;

    // NACC constitutive model
    Eigen::Matrix2f Re = polar_decomp_R(p.Fe);
    float Je = p.Fe.determinant();
    Eigen::Matrix2f dFe = 2.f * mu*(p.Fe - Re)* p.Fe.transpose() +
            lambda * (Je - 1.f) * Je * Eigen::Matrix2f::Identity();
    Eigen::Matrix2f Ap = dFe * particle_volume;

    // continue with distributing to the grid
    constexpr float offset = 0.5f;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]/cellsize - offset);
    const int j0 = (int)(p.pos[1]/cellsize - offset);

    for (int i = i0; i < i0+3; i++)
        for (int j = 0; j < j0+3; j++)
        {
            int idx_gridnode = i + j*gridX;
            if(i < 0 || j < 0 || i >=gridX || j>=gridY || idx_gridnode < 0)
                gpu_error_indicator = 1;

            Eigen::Vector2f pos_node(i*cellsize, j*cellsize);
            Eigen::Vector2f d = p.pos - pos_node;
            float Wip = wq(d, cellsize);   // weight
            Eigen::Vector2f dWip = gradwq(d, cellsize);    // weight gradient

            // APIC increments
            float incM = Wip * particle_mass;
            Eigen::Vector2f incV = incM * (p.velocity + Dp_inv * p.Bp * (-d));
            Eigen::Vector2f incFi = Ap * dWip;

            // Udpate mass, velocity and force
            icy::GridNode &gn = gpu_nodes[idx_gridnode];
            atomicAdd(&gn.mass, incM);
            atomicAdd(&gn.velocity[0], incV[0]);
            atomicAdd(&gn.velocity[1], incV[1]);
            atomicAdd(&gn.force[0], incFi[0]);
            atomicAdd(&gn.force[1], incFi[1]);
        }
}

void cuda_p2g(const int nPoints)
{
    int error_code = 0;
    cudaError_t err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_p2g error when resetting error_code");
        throw std::runtime_error("cuda_p2g");
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    kernel_p2g<<<blocksPerGrid, threadsPerBlock>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_p2g error executing kernel_p2g");
        throw std::runtime_error("cuda_p2g");
    }

    err = cudaMemcpyFromSymbol(&error_code, gpu_error_indicator, sizeof(int));
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_p2g cudaMemcpyFromSymbol error");
        throw std::runtime_error("cuda_p2g");
    }
    if(error_code)
    {
        spdlog::critical("point is out of bounds");
        throw std::runtime_error("cuda_p2g");
    }
}




__device__ void NACCUpdateDeformationGradient(icy::Point &p, Eigen::Matrix2f &FModifier)
{
    /*
    constexpr float magic_epsilon = 1e-5f;
    constexpr int d = 2; // dimensions
    float &alpha = p.NACC_alpha_p;

    Eigen::Matrix2f FeTr = (Eigen::Matrix2f::Identity() + dt * FModifier) * p.Fe;

    Eigen::Matrix2f U, V, Sigma;

    svd2x2(FeTr, U, Sigma, V);

    // line 4
    float p0 = kappa * (magic_epsilon + sinhf(xi * fmaxf(-alpha, 0.f)));

    // line 5
    float Je_tr = Sigma[0]*Sigma[1];    // this is for 2D

    // line 6
    Eigen::Matrix2f SigmaMatrix = Sigma.asDiagonal();
    Eigen::Matrix2f SigmaSquared = SigmaMatrix*SigmaMatrix;
    Eigen::Matrix2f SigmaSquaredDev = SigmaSquared - SigmaSquared.trace()/2.f*Eigen::Matrix2f::Identity();
    float J_power_neg_2_d_mulmu = mu * powf(Je_tr, -2.f / (float)d);///< J^(-2/dim) * mu
    Eigen::Matrix2f s_hat_tr = J_power_neg_2_d_mulmu * SigmaSquaredDev;

    // line 7
    float psi_kappa_partial_J = (kappa/2.f) * (Je_tr - 1.f / Je_tr);

    // line 8
    float p_trial = -psi_kappa_partial_J * Je_tr;

    // line 9 (case 1)
    float y = (1.f + 2.f*beta)*(3.f-(float)d/2.f)*s_hat_tr.norm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(p_trial > p0)
    {
        float Je_new = sqrtf(-2.f*p0 / kappa + 1.f);
        Eigen::Matrix2f Sigma_new = Eigen::Matrix2f::Identity() * powf(Je_new, 1.f/(float)d);
        p.Fe = U*Sigma_new*V.transpose();
        if(hardening) alpha += logf(Je_tr / Je_new);
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        float Je_new = sqrtf(2.f*beta*p0/kappa + 1.f);
        Eigen::Matrix2f Sigma_new = Eigen::Matrix2f::Identity() * pow(Je_new, 1.f/(float)d);
        p.Fe = U*Sigma_new*V.transpose();
        if(hardening) alpha += logf(Je_tr / Je_new);
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon*10)
    {
        if(hardening && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
        {
            float p_c = (1.f-beta)*p0/2.f;  // line 23
            float q_tr = sqrtf(3.f-d/2.f)*s_hat_tr.norm();   // line 24
            Eigen::Vector2f direction(p_c-p_trial, -q_tr);  // line 25
            direction.normalize();
            float C = M_sq*(p_c-beta*p0)*(p_c-p0);
            float B = M_sq*direction[0]*(2.f*p_c-p0+beta*p0);
            float A = M_sq*direction[0]*direction[0]+(1.f+2.f*beta)*direction[1]*direction[1];  // line 30
            float l1 = (-B+sqrtf(B*B-4.f*A*C))/(2.f*A);
            float l2 = (-B-sqrtf(B*B-4.f*A*C))/(2.f*A);
            float p1 = p_c + l1*direction[0];
            float p2 = p_c + l2*direction[0];
            float p_x = (p_trial-p_c)*(p1-p_c) > 0 ? p1 : p2;
            float Je_x = sqrtf(fabsf(-2.f*p_x/kappa + 1.f));
            if(Je_x > magic_epsilon*10) alpha += logf(Je_tr / Je_x);
        }

        float expr_under_root = (-M_sq*(p_trial+beta*p0)*(p_trial-p0))/((1+2.f*beta)*(3.f-d/2.));
        Eigen::Matrix2f B_hat_E_new = sqrtf(expr_under_root)*(powf(Je_tr,2.f/d)/mu)*s_hat_tr.normalized() +
                Eigen::Matrix2f::Identity()*SigmaSquared.trace()/(float)d;
        Eigen::Matrix2f Sigma_new;
        Sigma_new << sqrt(B_hat_E_new(0,0)), 0,
                0, sqrt(B_hat_E_new(1,1));
        p.Fe = U*Sigma_new*V.transpose();
    }
    else
    {
        p.Fe = FeTr;
    }
    p.visualized_value = alpha;
    */
}



__global__ void kernel_g2p(const int nPoints)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPoints) return;

    icy::Point &p = gpu_points[pt_idx];

    p.velocity.setZero();
    p.Bp.setZero();

    constexpr float offset = 0.5f;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)((p.pos[0])/cellsize - offset);
    const int j0 = (int)((p.pos[1])/cellsize - offset);
    const Eigen::Vector2f pointPos_copy = p.pos;
    p.pos.setZero();

    Eigen::Matrix2f T;
    T.setZero();

    for (int i = i0; i < i0+3; i++)
        for (int j = j0; j < j0+3; j++)
        {
            const int idx_gridnode = i + j*gridX;
            const icy::GridNode &node = gpu_nodes[idx_gridnode];

            Eigen::Vector2f pos_node(i*cellsize, j*cellsize);
            Eigen::Vector2f d = pointPos_copy - pos_node;   // dist
            float Wip = wq(d, cellsize);   // weight
            Eigen::Vector2f dWip = gradwq(d, cellsize);    // weight gradient

            p.velocity += Wip * node.velocity;
            p.Bp += Wip *(node.velocity*(-d).transpose());
            // Update position and nodal deformation
            p.pos += Wip * (pos_node + dt * node.velocity);
            T += node.velocity * dWip.transpose();
        }
    NACCUpdateDeformationGradient(p, T);
}


void cuda_g2p(const int nPoints)
{
    cudaError_t err;
    int threadsPerBlock = 256;
    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g2p<<<blocksPerGrid, threadsPerBlock>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        spdlog::critical("cuda_g2p");
        throw std::runtime_error("cuda_g2p");
    }
}




__global__ void cuda_hello(Eigen::Matrix2f A, Eigen::Matrix2f *result){

    Eigen::Matrix2f &U = result[0];
    Eigen::Matrix2f &Sigma = result[1];
    Eigen::Matrix2f &V = result[2];

    svd2x2(A, U, Sigma, V);

    printf("Hello World from GPU!\n\n");
}

void test_cuda()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if(error_id) std::cout << "cudaGetDeviceCount returs error " << error_id << '\n';
    std::cout << "CUDA devices " << deviceCount << '\n';

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device \"%s\"\n", deviceProp.name);
    printf("Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);


    Eigen::Matrix2f *results;
    cudaMallocManaged(&results, sizeof(Eigen::Matrix2f)*3);
    Eigen::Matrix2f A;
    A << 1,7,-3,4;
    cuda_hello<<<1,1>>>(A, results);
    cudaDeviceSynchronize();
    Eigen::Matrix2f U = results[0];
    Eigen::Matrix2f S = results[1];
    Eigen::Matrix2f V = results[2];

    std::cout << "A=\n" << A << '\n';
    std::cout << "U=\n" << U << '\n';
    std::cout << "S=\n" << S << '\n';
    std::cout << "V=\n" << V << '\n';
    std::cout << "USV^T=\n" << U*S*V.transpose() << '\n';
    cudaFree(results);

}
