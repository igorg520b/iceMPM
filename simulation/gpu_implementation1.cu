#include "gpu_implementation1.h"
//#include "gpu_implementation0.h"
#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "helper_math.cuh"


__device__ Eigen::Vector2f *gpu_pts_pos, *gpu_pts_velocity;
__device__ float *gpu_pts_Bp[4], *gpu_pts_Fe[4];
__device__ float *gpu_pts_NACC_alpha_p;

__device__ Eigen::Vector2f *gpu_grid_momentum, *gpu_grid_velocity, *gpu_grid_force;
__device__ float *gpu_grid_mass;


GPU_Implementation1::GPU_Implementation1()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}


void GPU_Implementation1::start_timing()
{
    cudaEventRecord(start);
}

float GPU_Implementation1::end_timing()
{
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}


void GPU_Implementation1::cuda_update_constants(const icy::SimParams &prms)
{
    cudaError_t err;
    int error_code = 0;
    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("gpu_error_indicator initialization");

    err = cudaMemcpyToSymbol(gprms, &prms, sizeof(icy::SimParams));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");

    std::cout << "CUDA constants copied to device\n";
}

void GPU_Implementation1::cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
    cudaError_t err;

    err = cudaMalloc(&_gpu_pts_pos, sizeof(Eigen::Vector2f)*nPoints);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMalloc(&_gpu_pts_velocity, sizeof(Eigen::Vector2f)*nPoints);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    for(int k=0;k<4;k++)
    {
        err = cudaMalloc(&_gpu_pts_Bp[k], sizeof(float)*nPoints);
        if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
        err = cudaMalloc(&_gpu_pts_Fe[k], sizeof(float)*nPoints);
        if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    }
    err = cudaMalloc(&_gpu_pts_NACC_alpha_p, sizeof(float)*nPoints);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    err = cudaMalloc(&_gpu_grid_momentum, sizeof(Eigen::Vector2f)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMalloc(&_gpu_grid_velocity, sizeof(Eigen::Vector2f)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMalloc(&_gpu_grid_force, sizeof(Eigen::Vector2f)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMalloc(&_gpu_grid_mass, sizeof(float)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");



    err = cudaMemcpyToSymbol(gpu_pts_pos, &_gpu_pts_pos, sizeof(void*));
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol gpu_pts_pos");
    err = cudaMemcpyToSymbol(gpu_pts_velocity, &_gpu_pts_velocity, sizeof(void*));
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol");
    err = cudaMemcpyToSymbol(gpu_pts_Bp, _gpu_pts_Bp, sizeof(void*)*4);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol gpu_pts_Bp");
    err = cudaMemcpyToSymbol(gpu_pts_Fe, _gpu_pts_Fe, sizeof(void*)*4);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol gpu_pts_Fe");
    err = cudaMemcpyToSymbol(gpu_pts_NACC_alpha_p, &_gpu_pts_NACC_alpha_p, sizeof(void*));
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol");

    err = cudaMemcpyToSymbol(gpu_grid_momentum, &_gpu_grid_momentum, sizeof(void*));
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol");
    err = cudaMemcpyToSymbol(gpu_grid_velocity, &_gpu_grid_velocity, sizeof(void*));
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol gpu_grid_velocity");
    err = cudaMemcpyToSymbol(gpu_grid_force, &_gpu_grid_force, sizeof(void*));
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol");
    err = cudaMemcpyToSymbol(gpu_grid_mass, &_gpu_grid_mass, sizeof(void*));
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol gpu_grid_mass");


    std::cout << "cuda_allocate_arrays done\n";

}

void GPU_Implementation1::transfer_ponts_to_device(const std::vector<icy::Point> &points)
{
    cudaError_t err;
    int n = points.size();
    tmp1.resize(n);
    tmp2.resize(n);

    for(int k=0;k<n;k++) tmp1[k]=points[k].pos;
    err = cudaMemcpy(_gpu_pts_pos, (void*)tmp1.data(), sizeof(Eigen::Vector2f)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp1[k]=points[k].velocity;
    err = cudaMemcpy(_gpu_pts_velocity, (void*)tmp1.data(), sizeof(Eigen::Vector2f)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Bp(0,0);
    err = cudaMemcpy(_gpu_pts_Bp[0], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Bp(0,1);
    err = cudaMemcpy(_gpu_pts_Bp[1], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Bp(1,0);
    err = cudaMemcpy(_gpu_pts_Bp[2], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Bp(1,1);
    err = cudaMemcpy(_gpu_pts_Bp[3], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Fe(0,0);
    err = cudaMemcpy(_gpu_pts_Fe[0], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Fe(0,1);
    err = cudaMemcpy(_gpu_pts_Fe[1], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Fe(1,0);
    err = cudaMemcpy(_gpu_pts_Fe[2], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].Fe(1,1);
    err = cudaMemcpy(_gpu_pts_Fe[3], (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
    for(int k=0;k<n;k++) tmp2[k]=points[k].NACC_alpha_p;
    err = cudaMemcpy(_gpu_pts_NACC_alpha_p, (void*)tmp2.data(), sizeof(float)*n, cudaMemcpyHostToDevice);

    if(err != cudaSuccess) throw std::runtime_error("transfer_ponts_to_device");
}

void GPU_Implementation1::cuda_transfer_from_device(std::vector<icy::Point> &points)
{
    int n = points.size();
    cudaError_t err;
    err = cudaMemcpy(tmp1.data(), _gpu_pts_pos, sizeof(Eigen::Vector2f)*n, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");
    for(int k=0;k<n;k++)points[k].pos = tmp1[k];

    err = cudaMemcpy(tmp2.data(), _gpu_pts_NACC_alpha_p, sizeof(float)*n, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");
    for(int k=0;k<n;k++)points[k].NACC_alpha_p = tmp2[k];

    int error_code = 0;
    err = cudaMemcpyFromSymbol(&error_code, gpu_error_indicator, sizeof(int));
    if(err != cudaSuccess)
    {
        std::cout << "cuda_p2g cudaMemcpyFromSymbol error\n";
        throw std::runtime_error("cuda_p2g");
    }
    if(error_code)
    {
        std::cout << "point is out of bounds\n";
        throw std::runtime_error("cuda_p2g");
    }
}

void GPU_Implementation1::cuda_device_synchronize()
{
    cudaError_t err;
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_device_synchronize failed\n";
        throw std::runtime_error("cuda_device_synchronize");
    }
}

void GPU_Implementation1::cuda_reset_grid(size_t nGridNodes)
{
    cudaError_t err = cudaMemsetAsync(_gpu_grid_momentum, 0, sizeof(Eigen::Vector2f)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid memset error");
    err = cudaMemsetAsync(_gpu_grid_velocity, 0, sizeof(Eigen::Vector2f)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid memset error");
    err = cudaMemsetAsync(_gpu_grid_force, 0, sizeof(Eigen::Vector2f)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid memset error");
    err = cudaMemsetAsync(_gpu_grid_mass, 0, sizeof(float)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid memset error");
}



void GPU_Implementation1::cuda_p2g(const int nPoints)
{
    cudaError_t err;

    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    v1_kernel_p2g<<<blocksPerGrid, threadsPerBlock>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_p2g error executing kernel_p2g\n";
        throw std::runtime_error("cuda_p2g");
    }
}


void GPU_Implementation1::cuda_g2p(const int nPoints)
{
    cudaError_t err;
    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    v1_kernel_g2p<<<blocksPerGrid, threadsPerBlock>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_g2p error\n";
        throw std::runtime_error("cuda_g2p");
    }
}


void GPU_Implementation1::cuda_update_nodes(const int nGridNodes,float indenter_x, float indenter_y)
{
    cudaError_t err;
    int blocksPerGrid = (nGridNodes + threadsPerBlock - 1) / threadsPerBlock;
    v1_kernel_update_nodes<<<blocksPerGrid, threadsPerBlock>>>(nGridNodes, indenter_x, indenter_y);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_update_nodes\n";
        throw std::runtime_error("cuda_update_nodes");
    }
}






// ==============================  kernels  ====================================


__global__ void v1_kernel_p2g(const int nPoints)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPoints) return;

    const float &particle_volume = gprms.ParticleVolume;
    const float &cellsize = gprms.cellsize;
    const float &Dp_inv = gprms.Dp_inv;
    const float &lambda = gprms.lambda;
    const float &mu = gprms.mu;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const float &particle_mass = gprms.ParticleMass;

    icy::Point p;
    p.pos = gpu_pts_pos[pt_idx];
    p.velocity = gpu_pts_velocity[pt_idx];
    p.Bp(0,0) = gpu_pts_Bp[0][pt_idx];
    p.Bp(0,1) = gpu_pts_Bp[1][pt_idx];
    p.Bp(1,0) = gpu_pts_Bp[2][pt_idx];
    p.Bp(1,1) = gpu_pts_Bp[3][pt_idx];
    p.Fe(0,0) = gpu_pts_Fe[0][pt_idx];
    p.Fe(0,1) = gpu_pts_Fe[1][pt_idx];
    p.Fe(1,0) = gpu_pts_Fe[2][pt_idx];
    p.Fe(1,1) = gpu_pts_Fe[3][pt_idx];
    p.NACC_alpha_p = gpu_pts_NACC_alpha_p[pt_idx];

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
            atomicAdd(&gpu_grid_mass[idx_gridnode], incM);
            atomicAdd(&gpu_grid_velocity[idx_gridnode][0], incV[0]);
            atomicAdd(&gpu_grid_velocity[idx_gridnode][1], incV[1]);
            atomicAdd(&gpu_grid_force[idx_gridnode][0], incFi[0]);
            atomicAdd(&gpu_grid_force[idx_gridnode][1], incFi[1]);
        }
}

__global__ void v1_kernel_update_nodes(const int nGridNodes, float indenter_x, float indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nGridNodes) return;

    icy::GridNode gn;
    gn.momentum = gpu_grid_momentum[idx];
    gn.velocity = gpu_grid_velocity[idx];
    gn.force = gpu_grid_force[idx];
    gn.mass = gpu_grid_mass[idx];

    if(gn.mass == 0) return;

    const float &gravity = gprms.Gravity;
    const float &indRsq = gprms.IndRSq;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const float &dt = gprms.InitialTimeStep;
    const float &ind_velocity = gprms.IndVelocity;
    const float &cellsize = gprms.cellsize;
    const float &ice_friction_coeff = gprms.IceFrictionCoefficient;

    const Eigen::Vector2f gravity_(0,-gravity);
    const Eigen::Vector2f vco(ind_velocity,0);  // velocity of the collision object (indenter)
    const Eigen::Vector2f indCenter(indenter_x, indenter_y);

    gn.velocity = gn.velocity/gn.mass + dt*(-gn.force/gn.mass + gravity_);

    int idx_x = idx % gridX;
    int idx_y = idx / gridX;

    // indenter
    Eigen::Vector2f gnpos(idx_x*cellsize, idx_y*cellsize);
    Eigen::Vector2f n = gnpos - indCenter;
    if(n.squaredNorm() < indRsq)
    {
        // grid node is inside the indenter
        Eigen::Vector2f vrel = gn.velocity - vco;
        n.normalize();
        float vn = vrel.dot(n);   // normal component of the velocity
        if(vn < 0)
        {
            Eigen::Vector2f vt = vrel - n*vn;   // tangential portion of relative velocity
            gn.velocity = vco + vt + ice_friction_coeff*vn*vt.normalized();
        }
    }

    // attached bottom layer
    if(idx_y <= 3) gn.velocity.setZero();
    else if(idx_y >= gridY-4 && gn.velocity[1]>0) gn.velocity[1] = 0;
    if(idx_x <= 3 && gn.velocity.x()<0) gn.velocity[0] = 0;
    else if(idx_x >= gridX-5) gn.velocity[0] = 0;

    gpu_grid_velocity[idx] = gn.velocity;
}

__global__ void v1_kernel_g2p(const int nPoints)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPoints) return;

    icy::Point p;
    p.pos = gpu_pts_pos[pt_idx];
//    p.velocity = gpu_pts_velocity[pt_idx];
//    p.Bp(0,0) = gpu_pts_Bp[0][pt_idx];
//    p.Bp(0,1) = gpu_pts_Bp[1][pt_idx];
//    p.Bp(1,0) = gpu_pts_Bp[2][pt_idx];
//    p.Bp(1,1) = gpu_pts_Bp[3][pt_idx];
    p.Fe(0,0) = gpu_pts_Fe[0][pt_idx];
    p.Fe(0,1) = gpu_pts_Fe[1][pt_idx];
    p.Fe(1,0) = gpu_pts_Fe[2][pt_idx];
    p.Fe(1,1) = gpu_pts_Fe[3][pt_idx];
    p.NACC_alpha_p = gpu_pts_NACC_alpha_p[pt_idx];

    const float &cellsize = gprms.cellsize;
    const float &dt = gprms.InitialTimeStep;
    const int &gridX = gprms.GridX;

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
            icy::GridNode node;
            node.momentum = gpu_grid_momentum[idx_gridnode];
            node.velocity = gpu_grid_velocity[idx_gridnode];
            node.force = gpu_grid_force[idx_gridnode];
            node.mass = gpu_grid_mass[idx_gridnode];

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

    gpu_pts_pos[pt_idx] = p.pos;
    gpu_pts_velocity[pt_idx] = p.velocity;
    gpu_pts_Bp[0][pt_idx] = p.Bp(0,0);
    gpu_pts_Bp[1][pt_idx] = p.Bp(0,1);
    gpu_pts_Bp[2][pt_idx] = p.Bp(1,0);
    gpu_pts_Bp[3][pt_idx] = p.Bp(1,1);
    gpu_pts_Fe[0][pt_idx] = p.Fe(0,0);
    gpu_pts_Fe[1][pt_idx] = p.Fe(0,1);
    gpu_pts_Fe[2][pt_idx] = p.Fe(1,0);
    gpu_pts_Fe[3][pt_idx] = p.Fe(1,1);
    gpu_pts_NACC_alpha_p[pt_idx] = p.NACC_alpha_p;
}

