#include "gpu_implementation3_sand.h"
#include "parameters_sim.h"
#include "point.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include <spdlog/spdlog.h>

#include "helper_math.cuh"

__device__ int gpu_error_indicator;
__constant__ icy::SimParams gprms;




void GPU_Implementation3::initialize()
{
    if(initialized) return;
    cudaError_t err;

    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("GPU_Implementation3::initialize() cuda error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    spdlog::info("Compute capability {}.{}",deviceProp.major, deviceProp.minor);

    cudaEventCreate(&eventCycleStart);
    cudaEventCreate(&eventCycleStop);

    err = cudaStreamCreate(&streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Implementation3::initialize() cudaEventCreate");

    initialized = true;
}

void GPU_Implementation3::cuda_update_constants()
{
    cudaError_t err;
    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess)
    {
        spdlog::critical("cudaMemcpyToSymbol error code {}",err);
        throw std::runtime_error("gpu_error_indicator initialization");
    }
    err = cudaMemcpyToSymbol(gprms, prms, sizeof(icy::SimParams));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");
    std::cout << "CUDA constants copied to device\n";
}

void GPU_Implementation3::cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
    if(!initialized) initialize();
//    size_t nGridNodes = prms->GridX*prms->GridY;
//    size_t nPoints = prms->nPts;
    cudaError_t err;

    // device memory for grid
    cudaFree(prms->grid_array);
    cudaFree(prms->pts_array);
    cudaFreeHost(tmp_transfer_buffer);

    err = cudaMallocPitch (&prms->grid_array, &prms->nGridPitch, sizeof(real)*nGridNodes, icy::SimParams::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    spdlog::info("Grid: requested {}B, pitched width is {} B", sizeof(real)*nGridNodes, prms->nGridPitch);

    // device memory for points
    err = cudaMallocPitch (&prms->pts_array, &prms->nPtsPitch, sizeof(real)*nPoints, icy::SimParams::nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    spdlog::info("Points: requested {} B, pitched width is {} B", sizeof(real)*nPoints, prms->nPtsPitch);
    spdlog::info("cuda_allocate_arrays done");

    // pinned host memory
    err = cudaMallocHost(&tmp_transfer_buffer, prms->nPtsPitch*icy::SimParams::nPtsArrays);
    if(err!=cudaSuccess) throw std::runtime_error("GPU_Implementation3::Prepare(int nPoints)");

    double MemAllocGrid = (double)prms->nGridPitch*icy::SimParams::nGridArrays/(1024*1024);
    double MemAllocPoints = (double)prms->nPtsPitch*icy::SimParams::nPtsArrays/(1024*1024);
    double MemAllocTotal = MemAllocGrid + MemAllocPoints;
    spdlog::info("memory use: grid {:03.2f} Mb; points {:03.2f} Mb ; total {:03.2f} Mb",
                 MemAllocGrid, MemAllocPoints, MemAllocTotal);
    error_code = 0;
}

void GPU_Implementation3::transfer_ponts_to_device(const std::vector<icy::Point> &points)
{
    int n = prms->nPtsPitch/sizeof(real);

    for(int i=0;i<prms->nPts;i++)
    {
        tmp_transfer_buffer[i + n*icy::SimParams::posx] = points[i].pos[0];
        tmp_transfer_buffer[i + n*icy::SimParams::posy] = points[i].pos[1];
        tmp_transfer_buffer[i + n*icy::SimParams::velx] = points[i].velocity[0];
        tmp_transfer_buffer[i + n*icy::SimParams::vely] = points[i].velocity[1];
        tmp_transfer_buffer[i + n*icy::SimParams::Bp00] = points[i].Bp(0,0);
        tmp_transfer_buffer[i + n*icy::SimParams::Bp01] = points[i].Bp(0,1);
        tmp_transfer_buffer[i + n*icy::SimParams::Bp10] = points[i].Bp(1,0);
        tmp_transfer_buffer[i + n*icy::SimParams::Bp11] = points[i].Bp(1,1);
        tmp_transfer_buffer[i + n*icy::SimParams::Fe00] = points[i].Fe(0,0);
        tmp_transfer_buffer[i + n*icy::SimParams::Fe01] = points[i].Fe(0,1);
        tmp_transfer_buffer[i + n*icy::SimParams::Fe10] = points[i].Fe(1,0);
        tmp_transfer_buffer[i + n*icy::SimParams::Fe11] = points[i].Fe(1,1);
        tmp_transfer_buffer[i + n*icy::SimParams::idx_case] = points[i].q;
        tmp_transfer_buffer[i + n*icy::SimParams::idx_Jp] = points[i].Jp_inv;
        tmp_transfer_buffer[i + n*icy::SimParams::idx_zeta] = points[i].zeta;
    }

    // transfer point data to device
    cudaError_t err;
    err = cudaMemcpy(prms->pts_array, tmp_transfer_buffer, prms->nPtsPitch*icy::SimParams::nPtsArrays, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");
}

void GPU_Implementation3::cuda_transfer_from_device()
{
    cudaError_t err;

    err = cudaMemcpyAsync(tmp_transfer_buffer, prms->pts_array, prms->nPtsPitch*icy::SimParams::nPtsArrays,
                          cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(int), 0, cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess)
    {
        std::cout << "cuda_p2g cudaMemcpyFromSymbol error\n";
        throw std::runtime_error("cuda_transfer_from_device");
    }

    void* userData = reinterpret_cast<void*>(this);
    cudaStreamAddCallback(streamCompute, GPU_Implementation3::callback_transfer_from_device_completion, userData, 0);
}

void CUDART_CB GPU_Implementation3::callback_transfer_from_device_completion(cudaStream_t stream, cudaError_t status, void *userData)
{
    // simulation data was copied to host memory -> proceed with processing of this data
    GPU_Implementation3 *m = reinterpret_cast<GPU_Implementation3*>(userData);
    if(m->transfer_completion_callback) m->transfer_completion_callback();
}

void GPU_Implementation3::transfer_ponts_to_host_finalize(std::vector<icy::Point> &points)
{
    int n = prms->nPtsPitch/sizeof(real);
    if(points.size() != prms->nPts) points.resize(prms->nPts);
    for(int i=0;i<prms->nPts;i++)
    {
        points[i].pos[0] = tmp_transfer_buffer[i + n*icy::SimParams::posx];
        points[i].pos[1] = tmp_transfer_buffer[i + n*icy::SimParams::posy];
        points[i].velocity[0] = tmp_transfer_buffer[i + n*icy::SimParams::velx];
        points[i].velocity[1] = tmp_transfer_buffer[i + n*icy::SimParams::vely];
        points[i].Bp(0,0) = tmp_transfer_buffer[i + n*icy::SimParams::Bp00];
        points[i].Bp(0,1) = tmp_transfer_buffer[i + n*icy::SimParams::Bp01];
        points[i].Bp(1,0) = tmp_transfer_buffer[i + n*icy::SimParams::Bp10];
        points[i].Bp(1,1) = tmp_transfer_buffer[i + n*icy::SimParams::Bp11];
        points[i].Fe(0,0) = tmp_transfer_buffer[i + n*icy::SimParams::Fe00];
        points[i].Fe(0,1) = tmp_transfer_buffer[i + n*icy::SimParams::Fe01];
        points[i].Fe(1,0) = tmp_transfer_buffer[i + n*icy::SimParams::Fe10];
        points[i].Fe(1,1) = tmp_transfer_buffer[i + n*icy::SimParams::Fe11];
        points[i].Jp_inv = tmp_transfer_buffer[i + n*icy::SimParams::idx_Jp];
        points[i].zeta = tmp_transfer_buffer[i + n*icy::SimParams::idx_zeta];

        points[i].visualize_p = tmp_transfer_buffer[i + n*icy::SimParams::idx_p];
        points[i].visualize_p0 = tmp_transfer_buffer[i + n*icy::SimParams::idx_p0];
        points[i].visualize_q = tmp_transfer_buffer[i + n*icy::SimParams::idx_q];
        points[i].visualize_psi = tmp_transfer_buffer[i + n*icy::SimParams::idx_psi];
        points[i].q = tmp_transfer_buffer[i + n*icy::SimParams::idx_case];
    }
}


void GPU_Implementation3::cuda_reset_grid()
{
    cudaError_t err = cudaMemsetAsync(prms->grid_array, 0, prms->nGridPitch*icy::SimParams::nGridArrays, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Implementation3::cuda_p2g()
{
    const int nPoints = prms->nPts;
    cudaError_t err;

    int tpb = prms->tpb_P2G;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    v2_kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_p2g error executing kernel_p2g " << err << std::endl;
        throw std::runtime_error("cuda_p2g");
    }
}

void GPU_Implementation3::cuda_update_nodes(real indenter_x, real indenter_y)
{
    const int nGridNodes = prms->GridX*prms->GridY;
    cudaError_t err;
    int tpb = prms->tpb_Upd;
    int blocksPerGrid = (nGridNodes + tpb - 1) / tpb;
    v2_kernel_update_nodes<<<blocksPerGrid, tpb, 0, streamCompute>>>(indenter_x, indenter_y);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_update_nodes\n";
        throw std::runtime_error("cuda_update_nodes");
    }
}

void GPU_Implementation3::cuda_g2p()
{
    const int nPoints = prms->nPts;
    cudaError_t err;
    int tpb = prms->tpb_G2P;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    v2_kernel_g2p<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_g2p error " << err << '\n';
        throw std::runtime_error("cuda_g2p");
    }
}

// ==============================  Functions that compute Kirchhoff stress via Strain Energy Density ========




// ==============================  kernels  ====================================

__global__ void v2_kernel_p2g()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const real &dt = gprms.InitialTimeStep;
    const real &vol = gprms.ParticleVolume;
    const real &h = gprms.cellsize;
    const real &h_inv = gprms.cellsize_inv;
    const real &Dinv = gprms.Dp_inv;
//    real lambda = gprms.lambda;
//    real mu = gprms.mu;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const real &particle_mass = gprms.ParticleMass;
    const int &nGridPitch = gprms.nGridPitch/sizeof(real);
    const int nPtsPitch = gprms.nPtsPitch/sizeof(real);

    icy::Point p;
    p.pos[0] = gprms.pts_array[icy::SimParams::posx*nPtsPitch + pt_idx];
    p.pos[1] = gprms.pts_array[icy::SimParams::posy*nPtsPitch + pt_idx];
    p.velocity[0] = gprms.pts_array[icy::SimParams::velx*nPtsPitch + pt_idx];
    p.velocity[1] = gprms.pts_array[icy::SimParams::vely*nPtsPitch + pt_idx];
    p.Bp(0,0) = gprms.pts_array[icy::SimParams::Bp00*nPtsPitch + pt_idx];
    p.Bp(0,1) = gprms.pts_array[icy::SimParams::Bp01*nPtsPitch + pt_idx];
    p.Bp(1,0) = gprms.pts_array[icy::SimParams::Bp10*nPtsPitch + pt_idx];
    p.Bp(1,1) = gprms.pts_array[icy::SimParams::Bp11*nPtsPitch + pt_idx];
    p.Fe(0,0) = gprms.pts_array[icy::SimParams::Fe00*nPtsPitch + pt_idx];
    p.Fe(0,1) = gprms.pts_array[icy::SimParams::Fe01*nPtsPitch + pt_idx];
    p.Fe(1,0) = gprms.pts_array[icy::SimParams::Fe10*nPtsPitch + pt_idx];
    p.Fe(1,1) = gprms.pts_array[icy::SimParams::Fe11*nPtsPitch + pt_idx];
    p.Jp_inv = gprms.pts_array[icy::SimParams::idx_Jp*nPtsPitch + pt_idx];
    p.zeta = gprms.pts_array[icy::SimParams::idx_zeta*nPtsPitch + pt_idx];



    Matrix2r PFt = KirchhoffStress_Wolper(p.Fe, p.zeta, p.Jp_inv);

    Matrix2r subterm2 = particle_mass*p.Bp - (dt*vol*Dinv)*PFt;

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]*h_inv - offset);
    const int j0 = (int)(p.pos[1]*h_inv - offset);

    Vector2r base_coord(i0,j0);
    Vector2r fx = p.pos*h_inv - base_coord;

    real v0[2] {1.5-fx[0], 1.5-fx[1]};
    real v1[2] {fx[0]-1.,  fx[1]-1.};
    real v2[2] {fx[0]-.5,  fx[1]-.5};

    real w[3][2] = {{.5*v0[0]*v0[0],  .5*v0[1]*v0[1]},
                    {.75-v1[0]*v1[0], .75-v1[1]*v1[1]},
                    {.5*v2[0]*v2[0],  .5*v2[1]*v2[1]}};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            real Wip = w[i][0]*w[j][1];
            Vector2r dpos((i-fx[0])*h, (j-fx[1])*h);
            Vector2r incV = Wip*(p.velocity*particle_mass + subterm2*dpos);
            real incM = Wip*particle_mass;

            int idx_gridnode = (i+i0) + (j+j0)*gridX;
            if((i+i0) < 0 || (j+j0) < 0 || (i+i0) >=gridX || (j+j0)>=gridY) gpu_error_indicator = 1;

            // Udpate mass, velocity and force
            atomicAdd(&gprms.grid_array[0*nGridPitch + idx_gridnode], incM);
            atomicAdd(&gprms.grid_array[1*nGridPitch + idx_gridnode], incV[0]);
            atomicAdd(&gprms.grid_array[2*nGridPitch + idx_gridnode], incV[1]);
        }
}

__global__ void v2_kernel_update_nodes(real indenter_x, real indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nGridNodes = gprms.GridX*gprms.GridY;
    if(idx >= nGridNodes) return;

    const int &nGridPitch = gprms.nGridPitch/sizeof(real);
    real mass = gprms.grid_array[0*nGridPitch + idx];
    if(mass == 0) return;

    Vector2r velocity;
    velocity[0] = gprms.grid_array[1*nGridPitch + idx];
    velocity[1] = gprms.grid_array[2*nGridPitch + idx];

    const real &gravity = gprms.Gravity;
    const real &indRsq = gprms.IndRSq;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const real &dt = gprms.InitialTimeStep;
    const real &ind_velocity = gprms.IndVelocity;
    const real &cellsize = gprms.cellsize;
    const real &ice_friction_coeff = gprms.IceFrictionCoefficient;

    const Vector2r vco(ind_velocity,0);  // velocity of the collision object (indenter)
    const Vector2r indCenter(indenter_x, indenter_y);

    velocity /= mass;
    velocity[1] -= dt*gravity;
    real vmax = 0.5*cellsize/dt;
    if(velocity.norm() > vmax) velocity = velocity/velocity.norm()*vmax;

    int idx_x = idx % gridX;
    int idx_y = idx / gridX;

    // indenter
    Vector2r gnpos(idx_x*cellsize, idx_y*cellsize);
    Vector2r n = gnpos - indCenter;
    if(n.squaredNorm() < indRsq)
    {
        // grid node is inside the indenter
        Vector2r vrel = velocity - vco;
        n.normalize();
        real vn = vrel.dot(n);   // normal component of the velocity
        if(vn < 0)
        {
            Vector2r vt = vrel - n*vn;   // tangential portion of relative velocity
            velocity = vco + vt + ice_friction_coeff*vn*vt.normalized();
        }
    }

    // attached bottom layer
    if(idx_y <= 3) velocity.setZero();
    else if(idx_y >= gridY-4 && velocity[1]>0) velocity[1] = 0;
    if(idx_x <= 3 && velocity.x()<0) velocity[0] = 0;
    else if(idx_x >= gridX-5) velocity[0] = 0;
    if(gprms.HoldBlockOnTheRight==1)
    {
        int blocksGridX = gprms.BlockLength*gprms.cellsize_inv+5-2;
        if(idx_x >= blocksGridX) velocity.setZero();
    }
    else if(gprms.HoldBlockOnTheRight==2)
    {
        int blocksGridX = gprms.BlockLength*gprms.cellsize_inv+5-2;
        int blocksGridY = gprms.BlockHeight/2*gprms.cellsize_inv+2;
        if(idx_x >= blocksGridX && idx_x <= blocksGridX + 2 && idx_y < blocksGridY) velocity.setZero();
        if(idx_x <= 7 && idx_x > 4 && idx_y < blocksGridY) velocity.setZero();
    }


    // write the updated grid velocity back to memory
    gprms.grid_array[1*nGridPitch + idx] = velocity[0];
    gprms.grid_array[2*nGridPitch + idx] = velocity[1];
}

__global__ void v2_kernel_g2p()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const int nPtsPitched = gprms.nPtsPitch/sizeof(real);
    const int nGridPitched = gprms.nGridPitch/sizeof(real);
    const real &h_inv = gprms.cellsize_inv;
    const real &dt = gprms.InitialTimeStep;
    const int &gridX = gprms.GridX;

    icy::Point p;
    p.pos[0] = gprms.pts_array[icy::SimParams::posx*nPtsPitched + pt_idx];
    p.pos[1] = gprms.pts_array[icy::SimParams::posy*nPtsPitched + pt_idx];
    p.Fe(0,0) = gprms.pts_array[icy::SimParams::Fe00*nPtsPitched + pt_idx];
    p.Fe(0,1) = gprms.pts_array[icy::SimParams::Fe01*nPtsPitched + pt_idx];
    p.Fe(1,0) = gprms.pts_array[icy::SimParams::Fe10*nPtsPitched + pt_idx];
    p.Fe(1,1) = gprms.pts_array[icy::SimParams::Fe11*nPtsPitched + pt_idx];

    p.q = gprms.pts_array[icy::SimParams::idx_case*nPtsPitched + pt_idx];
    p.Jp_inv = gprms.pts_array[icy::SimParams::idx_Jp*nPtsPitched + pt_idx];
    p.zeta = gprms.pts_array[icy::SimParams::idx_zeta*nPtsPitched + pt_idx];

    p.velocity.setZero();
    p.Bp.setZero();

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]*h_inv - offset);
    const int j0 = (int)(p.pos[1]*h_inv - offset);

    Vector2r base_coord(i0,j0);
    Vector2r fx = p.pos*h_inv - base_coord;

    real v0[2] {1.5-fx[0], 1.5-fx[1]};
    real v1[2] {fx[0]-1.,  fx[1]-1.};
    real v2[2] {fx[0]-.5,  fx[1]-.5};

    real w[3][2] = {{.5*v0[0]*v0[0],  .5*v0[1]*v0[1]},
                    {.75-v1[0]*v1[0], .75-v1[1]*v1[1]},
                    {.5*v2[0]*v2[0],  .5*v2[1]*v2[1]}};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            Vector2r dpos = Vector2r(i, j) - fx;
            real weight = w[i][0]*w[j][1];

            int idx_gridnode = i+i0 + (j+j0)*gridX;
            Vector2r node_velocity;
            node_velocity[0] = gprms.grid_array[1*nGridPitched + idx_gridnode];
            node_velocity[1] = gprms.grid_array[2*nGridPitched + idx_gridnode];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection
    p.pos += dt * p.velocity;

//    NACCUpdateDeformationGradient_q_hardening_2(p);
    NACCUpdateDeformationGradient_q_hardening(p);

    gprms.pts_array[icy::SimParams::posx*nPtsPitched + pt_idx] = p.pos[0];
    gprms.pts_array[icy::SimParams::posy*nPtsPitched + pt_idx] = p.pos[1];
    gprms.pts_array[icy::SimParams::velx*nPtsPitched + pt_idx] = p.velocity[0];
    gprms.pts_array[icy::SimParams::vely*nPtsPitched + pt_idx] = p.velocity[1];
    gprms.pts_array[icy::SimParams::Bp00*nPtsPitched + pt_idx] = p.Bp(0,0);
    gprms.pts_array[icy::SimParams::Bp01*nPtsPitched + pt_idx] = p.Bp(0,1);
    gprms.pts_array[icy::SimParams::Bp10*nPtsPitched + pt_idx] = p.Bp(1,0);
    gprms.pts_array[icy::SimParams::Bp11*nPtsPitched + pt_idx] = p.Bp(1,1);
    gprms.pts_array[icy::SimParams::Fe00*nPtsPitched + pt_idx] = p.Fe(0,0);
    gprms.pts_array[icy::SimParams::Fe01*nPtsPitched + pt_idx] = p.Fe(0,1);
    gprms.pts_array[icy::SimParams::Fe10*nPtsPitched + pt_idx] = p.Fe(1,0);
    gprms.pts_array[icy::SimParams::Fe11*nPtsPitched + pt_idx] = p.Fe(1,1);

    gprms.pts_array[icy::SimParams::idx_Jp*nPtsPitched + pt_idx] = p.Jp_inv;
    gprms.pts_array[icy::SimParams::idx_zeta*nPtsPitched + pt_idx] = p.zeta;

    // visualized variables
    gprms.pts_array[icy::SimParams::idx_p*nPtsPitched + pt_idx] = p.visualize_p;
    gprms.pts_array[icy::SimParams::idx_p0*nPtsPitched + pt_idx] = p.visualize_p0;
    gprms.pts_array[icy::SimParams::idx_q*nPtsPitched + pt_idx] = p.visualize_q;
    gprms.pts_array[icy::SimParams::idx_psi*nPtsPitched + pt_idx] = p.visualize_psi;
    gprms.pts_array[icy::SimParams::idx_case*nPtsPitched + pt_idx] = p.q;
}

//===========================================================================





__device__ Matrix2r dev(Matrix2r A)
{
    return A - A.trace()/2*Matrix2r::Identity();
}




// clamp x to range [a, b]
__device__ double clamp(double x, double a, double b)
{
    return max(a, min(b, x));
}


//===========================================================================

//===========================================================================

__device__ void svd(const real a[4], real u[4], real sigma[2], real v[4])
{
    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(a, gu, sigma, gv);
    gu.template fill<2, real>(u);
    gv.template fill<2, real>(v);
}

__device__ void svd2x2(const Matrix2r &mA, Matrix2r &mU, Matrix2r &mS, Matrix2r &mV)
{
    real U[4], V[4], S[2];
    real a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);
    mU << U[0],U[1],U[2],U[3];
    mS << S[0],0,0,S[1];
    mV << V[0],V[1],V[2],V[3];
}


__device__ Matrix2r polar_decomp_R(const Matrix2r &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    real th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Matrix2r result;
    result << cosf(th), -sinf(th), sinf(th), cosf(th);
    return result;
}

__global__ void kernel_hello()
{
    printf("hello from CUDA\n");
}


void GPU_Implementation3::test()
{
    cudaError_t err;
    kernel_hello<<<1,1,0,streamCompute>>>();
    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        std::cout << "cuda test error " << err << '\n';
        throw std::runtime_error("cuda test");
    }
    else
    {
        std::cout << "hello kernel executed successfully\n";
    }
    cudaDeviceSynchronize();
}

void GPU_Implementation3::synchronize()
{
    if(!initialized) return;
    cudaDeviceSynchronize();
}

