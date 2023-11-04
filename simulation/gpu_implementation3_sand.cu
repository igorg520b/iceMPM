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

    err = cudaEventCreate(&eventP2GStart);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Implementation3::initialize() cudaEventCreate");

    cudaEventCreate(&eventP2GStop);
    cudaEventCreate(&eventUpdGridStart);
    cudaEventCreate(&eventUpdGridStop);
    cudaEventCreate(&eventG2PStart);
    cudaEventCreate(&eventG2PStop);
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
        tmp_transfer_buffer[i + n*icy::SimParams::idx_NACCAlphaP] = points[i].NACC_alpha_p;
        tmp_transfer_buffer[i + n*icy::SimParams::idx_q] = points[i].q;
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
        points[i].NACC_alpha_p = tmp_transfer_buffer[i + n*icy::SimParams::idx_NACCAlphaP];
        points[i].q = tmp_transfer_buffer[i + n*icy::SimParams::idx_q];
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

    int blocksPerGrid = (nPoints + threadsPerBlock2 - 1) / threadsPerBlock2;
    v2_kernel_p2g<<<blocksPerGrid, threadsPerBlock2, 0, streamCompute>>>();
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
    int blocksPerGrid = (nGridNodes + threadsPerBlock1 - 1) / threadsPerBlock1;
    v2_kernel_update_nodes<<<blocksPerGrid, threadsPerBlock1, 0, streamCompute>>>(indenter_x, indenter_y);
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
    int blocksPerGrid = (nPoints + threadsPerBlock2 - 1) / threadsPerBlock2;
    v2_kernel_g2p<<<blocksPerGrid, threadsPerBlock2, 0, streamCompute>>>();
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_g2p error " << err << '\n';
        throw std::runtime_error("cuda_g2p");
    }
}


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
    const real &lambda = gprms.lambda;
    const real &mu = gprms.mu;
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

    Matrix2r Re = polar_decomp_R(p.Fe);
    real Je = p.Fe.determinant();
    Matrix2r &F = p.Fe;
//    Matrix2r PFt = 2.*mu*(p.Fe - Re)* p.Fe.transpose() + lambda * (Je - 1.) * Je * Matrix2r::Identity();
    Matrix2r PFt = mu*F*F.transpose() + (-mu+lambda*log(Je))* Matrix2r::Identity();     // Neo-Hookean; Sifakis
/*    Matrix2r U, V, Sigma;
    svd2x2(F, U, Sigma, V);
    Matrix2r lnSigma,invSigma;
    lnSigma << log(Sigma(0,0)),0,0,log(Sigma(1,1));
    invSigma = Sigma.inverse();
    Matrix2r PFt = U*(2*mu*invSigma*lnSigma + lambda*lnSigma.trace()*invSigma)*V.transpose()*F.transpose();
*/

    Matrix2r subterm1 = -(dt*vol*Dinv) * PFt;
    Matrix2r subterm2 = subterm1 + particle_mass * p.Bp;

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]*h_inv - offset);
    const int j0 = (int)(p.pos[1]*h_inv - offset);

    Vector2r base_coord(i0,j0);
    Vector2r fx = p.pos*h_inv - base_coord;

    Vector2r v0(1.5-fx[0],1.5-fx[1]);
    Vector2r v1(fx[0]-1.,fx[1]-1.);
    Vector2r v2(fx[0]-.5,fx[1]-.5);

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
            if((i+i0) < 0 || (j+j0) < 0 || (i+i0) >=gridX || (j+j0)>=gridY || idx_gridnode < 0)
                gpu_error_indicator = 1;

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
//    p.NACC_alpha_p = gprms.pts_array[icy::SimParams::idx_NACCAlphaP*nPtsPitched + pt_idx];
    p.q = gprms.pts_array[icy::SimParams::idx_q*nPtsPitched + pt_idx];
    p.velocity.setZero();
    p.Bp.setZero();

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]*h_inv - offset);
    const int j0 = (int)(p.pos[1]*h_inv - offset);

    Vector2r base_coord(i0,j0);
    Vector2r fx = p.pos*h_inv - base_coord;

    Vector2r v0(1.5-fx[0],1.5-fx[1]);
    Vector2r v1(fx[0]-1.,fx[1]-1.);
    Vector2r v2(fx[0]-.5,fx[1]-.5);

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

//    NACCUpdateDeformationGradient(p);
    DruckerPragerUpdateDeformationGradient(p);

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
//    gprms.pts_array[icy::SimParams::idx_NACCAlphaP*nPtsPitched + pt_idx] = p.NACC_alpha_p;
    gprms.pts_array[icy::SimParams::idx_q*nPtsPitched + pt_idx] = p.q;
}


__device__ void DruckerPragerUpdateDeformationGradient(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
//    constexpr real magic_epsilon = 1.e-15;
    const real &mu = gprms.mu;
    const real &lambda = gprms.lambda;
    const real &dt = gprms.InitialTimeStep;
    const real &H0 = gprms.H0;
    const real &H1 = gprms.H1;
    const real &H2 = gprms.H2;
    const real &H3 = gprms.H3;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;

    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    Vector2r T;
    Matrix2r Tmatrix; // from DrySand::Plasticity()

    // Projection
    double dq = 0;
    Matrix2r lnSigma, e_c;
    lnSigma << log(Sigma(0,0)),0,0,log(Sigma(1,1));
    e_c = lnSigma - lnSigma.trace()/2.0 * Matrix2r::Identity();   // deviatoric part

//    if(e_c.norm() < magic_epsilon || lnSigma.trace()>0)
    if(e_c.norm() ==0 || lnSigma.trace()>0)
    {
        // Projection to the tip of the cone
        Tmatrix.setIdentity();
        dq = lnSigma.norm();
    }
    else
    {
        double phi = H0 + (H1 *p.q - H3)*exp(-H2 * p.q);
        double alpha = sqrt(2.0 / 3.0) * (2.0 * sin(phi)) / (3.0 - sin(phi));
        double dg = e_c.norm() + (lambda + mu) / mu * lnSigma.trace() * alpha;

        if (dg <= 0)
        {
            Tmatrix = Sigma;
            dq = 0;
        }
        else
        {
            Matrix2r Hm = lnSigma - e_c * (dg / e_c.norm());
            Tmatrix << exp(Hm(0,0)), 0, 0, exp(Hm(1,1));
            dq = dg;
        }
    }

    p.Fe = U*Tmatrix*V.transpose();
    p.q += dq; // hardening
}

//===========================================================================


__device__ void NACCUpdateDeformationGradient(icy::Point &p)
{
    const Matrix2r &FModifier = p.Bp;
    constexpr real magic_epsilon = 1.e-5;
    constexpr int d = 2; // dimensions
    real &alpha = p.NACC_alpha_p;
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &beta = gprms.NACC_beta;
    const real &M_sq = gprms.NACC_M_sq;
    const real &xi = gprms.NACC_xi;
    const real &dt = gprms.InitialTimeStep;

    Matrix2r FeTr = (Matrix2r::Identity() + dt * FModifier) * p.Fe;
    Matrix2r U, V, Sigma;
    svd2x2(FeTr, U, Sigma, V);

    // line 4
    real p0 = kappa * (magic_epsilon + sinh(xi * max(-alpha, 0.)));

    // line 5
    real Je_tr = Sigma(0,0)*Sigma(1,1);    // this is for 2D

    // line 6
    Matrix2r SigmaSquared = Sigma*Sigma;
    Matrix2r SigmaSquaredDev = SigmaSquared - SigmaSquared.trace()/2.*Matrix2r::Identity();
    real J_power_neg_2_d_mulmu = mu * pow(Je_tr, -2. / (real)d);///< J^(-2/dim) * mu
    Matrix2r s_hat_tr = J_power_neg_2_d_mulmu * SigmaSquaredDev;

    // line 7
    real psi_kappa_partial_J = (kappa/2.) * (Je_tr - 1./Je_tr);

    // line 8
    real p_trial = -psi_kappa_partial_J * Je_tr;

    // line 9 (case 1)
    real y = (1. + 2.*beta)*(3.-(real)d/2.)*s_hat_tr.norm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(p_trial > p0)
    {
        real Je_new = sqrt(-2.*p0 / kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        if(true) alpha += log(Je_tr / Je_new);
    }

    // line 14 (case 2)
    else if(p_trial < -beta*p0)
    {
        real Je_new = sqrt(2.*beta*p0/kappa + 1.);
        Matrix2r Sigma_new = Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*Sigma_new*V.transpose();
        if(true) alpha += log(Je_tr / Je_new);
    }

    // line 19 (case 3)
    else if(y >= magic_epsilon*10)
    {
        if(true && p0 > magic_epsilon && p_trial < p0 - magic_epsilon && p_trial > -beta*p0 + magic_epsilon)
        {
            real p_c = (1.-beta)*p0/2.;  // line 23
            real q_tr = sqrt(3.-d/2.)*s_hat_tr.norm();   // line 24
            Vector2r direction(p_c-p_trial, -q_tr);  // line 25
            direction.normalize();
            real C = M_sq*(p_c-beta*p0)*(p_c-p0);
            real B = M_sq*direction[0]*(2.*p_c-p0+beta*p0);
            real A = M_sq*direction[0]*direction[0]+(1.+2.*beta)*direction[1]*direction[1];  // line 30
            real l1 = (-B+sqrt(B*B-4.*A*C))/(2.*A);
            real l2 = (-B-sqrt(B*B-4.*A*C))/(2.*A);
            real p1 = p_c + l1*direction[0];
            real p2 = p_c + l2*direction[0];
            real p_x = (p_trial-p_c)*(p1-p_c) > 0 ? p1 : p2;
            real Je_x = sqrt(abs(-2.*p_x/kappa + 1.));
            if(Je_x > magic_epsilon*10) alpha += log(Je_tr / Je_x);
        }

        real expr_under_root = (-M_sq*(p_trial+beta*p0)*(p_trial-p0))/((1+2.*beta)*(3.-d/2.));
        Matrix2r B_hat_E_new = sqrt(expr_under_root)*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() +
                               Matrix2r::Identity()*SigmaSquared.trace()/(real)d;
        Matrix2r Sigma_new;
        Sigma_new << sqrt(B_hat_E_new(0,0)), 0,
            0, sqrt(B_hat_E_new(1,1));
        p.Fe = U*Sigma_new*V.transpose();
    }
    else
    {
        p.Fe = FeTr;
    }
    //p.visualized_value = alpha;
}



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

