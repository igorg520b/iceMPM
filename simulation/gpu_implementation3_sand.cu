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




void GPU_Implementation2::initialize()
{
    if(initialized) return;
    cudaError_t err;

    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("GPU_Implementation2() constructor cuda error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    spdlog::info("Compute capability {}.{}",deviceProp.major, deviceProp.minor);

    err = cudaEventCreate(&eventTimingStart);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Implementation2::GPU_Implementation2() cudaEventCreate");
    cudaEventCreate(&eventTimingStop);

    cudaEventCreate(&eventCycleComplete);
    cudaEventCreate(&eventDataCopiedToHost);

    cudaStreamCreate(&streamCompute);
    cudaStreamCreate(&streamTransfer);

    initialized = true;


    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess)
    {
        spdlog::critical("cudaMemcpyToSymbol error code {}",err);
        throw std::runtime_error("gpu_error_indicator initialization");
    }
}



void GPU_Implementation2::cuda_update_constants()
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

void GPU_Implementation2::cuda_allocate_arrays()
{
    size_t nGridNodes = prms->GridSize;
    size_t nPoints = prms->PointCountActual;
    cudaError_t err;

    // pinned host memory
    if(tmp_transfer_buffer) cudaFreeHost(tmp_transfer_buffer);
    err = cudaMallocHost(&tmp_transfer_buffer, sizeof(real)*nPoints*3);
    if(err!=cudaSuccess) throw std::runtime_error("GPU_Implementation2::Prepare(int nPoints)");

    // device memory for grid
    for(int k=0;k<icy::SimParams::nGridArrays;k++)
    {
        err = cudaMalloc(&prms->grid_arrays[k], sizeof(real)*nGridNodes);
        if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    }

    // device memory for points
    for(int k=0;k<icy::SimParams::nPtsArrays;k++)
    {
        err = cudaMalloc(&prms->pts_arrays[k], sizeof(real)*nPoints);
        if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    }
    std::cout << "cuda_allocate_arrays done\n";
}

void GPU_Implementation2::transfer_ponts_to_device(const std::vector<icy::Point> &points)
{
    cudaError_t err;
    int n = points.size();
    real *tmp = (real*) tmp_transfer_buffer;

    // transfer point positions
    for(int i=0;i<n;i++) tmp[i] = points[i].pos[0];
    err = cudaMemcpy(prms->pts_arrays[0], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].pos[1];
    err = cudaMemcpy(prms->pts_arrays[1], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer velocity
    for(int i=0;i<n;i++) tmp[i] = points[i].velocity[0];
    err = cudaMemcpy(prms->pts_arrays[2], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].velocity[1];
    err = cudaMemcpy(prms->pts_arrays[3], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer Bp
    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(0,0);
    err = cudaMemcpy(prms->pts_arrays[4], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(0,1);
    err = cudaMemcpy(prms->pts_arrays[5], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(1,0);
    err = cudaMemcpy(prms->pts_arrays[6], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(1,1);
    err = cudaMemcpy(prms->pts_arrays[7], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer Fe
    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(0,0);
    err = cudaMemcpy(prms->pts_arrays[8], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(0,1);
    err = cudaMemcpy(prms->pts_arrays[9], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(1,0);
    err = cudaMemcpy(prms->pts_arrays[10], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(1,1);
    err = cudaMemcpy(prms->pts_arrays[11], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer NACC_alpha
    for(int i=0;i<n;i++) tmp[i] = points[i].NACC_alpha_p;
    err = cudaMemcpy(prms->pts_arrays[12], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");
//13,14,15

    // transfer q
    for(int i=0;i<n;i++) tmp[i] = points[i].q;
    err = cudaMemcpy(prms->pts_arrays[16], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

}

void GPU_Implementation2::backup_point_positions()
{
    const int nPoints = prms->PointCountActual;
    cudaError_t err;
    err = cudaMemcpyAsync(prms->pts_arrays[13], prms->pts_arrays[0], sizeof(real)*nPoints, cudaMemcpyDeviceToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("backup_point_positions");
    err = cudaMemcpyAsync(prms->pts_arrays[14], prms->pts_arrays[1], sizeof(real)*nPoints, cudaMemcpyDeviceToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("backup_point_positions");
    err = cudaMemcpyAsync(prms->pts_arrays[15], prms->pts_arrays[16], sizeof(real)*nPoints, cudaMemcpyDeviceToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("backup_point_positions");
}

void GPU_Implementation2::cuda_transfer_from_device()
{
    int n = prms->PointCountActual;
    cudaError_t err;

    real* tmp2 = (real*)tmp_transfer_buffer + n;
    real* tmp3 = (real*)tmp_transfer_buffer + n*2;

    cudaStreamWaitEvent(streamTransfer, eventCycleComplete);    // wait until streamCompute has the data ready
    err = cudaMemcpyAsync(tmp_transfer_buffer, prms->pts_arrays[13], sizeof(real)*n, cudaMemcpyDeviceToHost, streamTransfer);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");
    err = cudaMemcpyAsync((void*) tmp2, prms->pts_arrays[14], sizeof(real)*n, cudaMemcpyDeviceToHost, streamTransfer);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");
    err = cudaMemcpyAsync((void*) tmp3, prms->pts_arrays[15], sizeof(real)*n, cudaMemcpyDeviceToHost, streamTransfer);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(int), 0, cudaMemcpyDeviceToHost, streamTransfer);
    if(err != cudaSuccess)
    {
        std::cout << "cuda_p2g cudaMemcpyFromSymbol error\n";
        throw std::runtime_error("cuda_transfer_from_device");
    }

    void* userData = reinterpret_cast<void*>(this);
    cudaStreamAddCallback(streamTransfer, GPU_Implementation2::callback_transfer_from_device_completion, userData, 0);
}

void CUDART_CB GPU_Implementation2::callback_transfer_from_device_completion(cudaStream_t stream, cudaError_t status, void *userData)
{
    // simulation data was copied to host memory -> proceed with processing of this data
    GPU_Implementation2 *m = reinterpret_cast<GPU_Implementation2*>(userData);
    if(m->transfer_completion_callback) m->transfer_completion_callback();
}

void GPU_Implementation2::transfer_ponts_to_host_finalize(std::vector<icy::Point> &points)
{
    int n = points.size();
    real* tmp1 = (real*)tmp_transfer_buffer;
    for(int k=0;k<n;k++)
    {
        points[k].pos[0] = tmp1[k];
        points[k].pos[1] = tmp1[k+n];
//        points[k].NACC_alpha_p = tmp1[k+2*n];
        points[k].q = tmp1[k+2*n];
    }
}


void GPU_Implementation2::cuda_reset_grid()
{
    for(int k=0;k<3;k++)
    {
        cudaError_t err = cudaMemsetAsync(prms->grid_arrays[k], 0, sizeof(real)*prms->GridSize, streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
    }
}



void GPU_Implementation2::cuda_p2g()
{
    const int nPoints = prms->PointCountActual;
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

void GPU_Implementation2::cuda_update_nodes(real indenter_x, real indenter_y)
{
    const int nGridNodes = prms->GridSize;
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

void GPU_Implementation2::cuda_g2p()
{
    const int nPoints = prms->PointCountActual;
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
    const int &nPoints = gprms.PointCountActual;
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

    icy::Point p;
    p.pos[0] = gprms.pts_arrays[0][pt_idx];
    p.pos[1] = gprms.pts_arrays[1][pt_idx];
    p.velocity[0] = gprms.pts_arrays[2][pt_idx];
    p.velocity[1] = gprms.pts_arrays[3][pt_idx];
    p.Bp(0,0) = gprms.pts_arrays[4][pt_idx];
    p.Bp(0,1) = gprms.pts_arrays[5][pt_idx];
    p.Bp(1,0) = gprms.pts_arrays[6][pt_idx];
    p.Bp(1,1) = gprms.pts_arrays[7][pt_idx];
    p.Fe(0,0) = gprms.pts_arrays[8][pt_idx];
    p.Fe(0,1) = gprms.pts_arrays[9][pt_idx];
    p.Fe(1,0) = gprms.pts_arrays[10][pt_idx];
    p.Fe(1,1) = gprms.pts_arrays[11][pt_idx];

    Matrix2r Re = polar_decomp_R(p.Fe);
    real Je = p.Fe.determinant();
    Matrix2r &F = p.Fe;
//    Matrix2r PFt = 2.*mu*(p.Fe - Re)* p.Fe.transpose() + lambda * (Je - 1.) * Je * Matrix2r::Identity();
    Matrix2r PFt = mu*F*F.transpose() + (-mu+lambda*log(Je))* Matrix2r::Identity();
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
            atomicAdd(&gprms.grid_arrays[0][idx_gridnode], incM);
            atomicAdd(&gprms.grid_arrays[1][idx_gridnode], incV[0]);
            atomicAdd(&gprms.grid_arrays[2][idx_gridnode], incV[1]);
        }
}

__global__ void v2_kernel_update_nodes(real indenter_x, real indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nGridNodes = gprms.GridSize;
    if(idx >= nGridNodes) return;

    real mass = gprms.grid_arrays[0][idx];
    if(mass == 0) return;

    Vector2r velocity;
    velocity << gprms.grid_arrays[1][idx], gprms.grid_arrays[2][idx];

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
    gprms.grid_arrays[1][idx] = velocity[0];
    gprms.grid_arrays[2][idx] = velocity[1];
}

__global__ void v2_kernel_g2p()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.PointCountActual;
    if(pt_idx >= nPoints) return;

    icy::Point p;
    p.pos[0] = gprms.pts_arrays[0][pt_idx];
    p.pos[1] = gprms.pts_arrays[1][pt_idx];
    p.Fe(0,0) = gprms.pts_arrays[8][pt_idx];
    p.Fe(0,1) = gprms.pts_arrays[9][pt_idx];
    p.Fe(1,0) = gprms.pts_arrays[10][pt_idx];
    p.Fe(1,1) = gprms.pts_arrays[11][pt_idx];
//    p.NACC_alpha_p = gprms.pts_arrays[12][pt_idx];
    p.q = gprms.pts_arrays[16][pt_idx];

    p.velocity.setZero();
    p.Bp.setZero();

    const real &h_inv = gprms.cellsize_inv;
    const real &dt = gprms.InitialTimeStep;
    const int &gridX = gprms.GridX;

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
            node_velocity << gprms.grid_arrays[1][idx_gridnode], gprms.grid_arrays[2][idx_gridnode];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection
    p.pos += dt * p.velocity;

//    NACCUpdateDeformationGradient(p);
    DruckerPragerUpdateDeformationGradient(p);

    gprms.pts_arrays[0][pt_idx] = p.pos[0];
    gprms.pts_arrays[1][pt_idx] = p.pos[1];
    gprms.pts_arrays[2][pt_idx] = p.velocity[0];
    gprms.pts_arrays[3][pt_idx] = p.velocity[1];
    gprms.pts_arrays[4][pt_idx] = p.Bp(0,0);
    gprms.pts_arrays[5][pt_idx] = p.Bp(0,1);
    gprms.pts_arrays[6][pt_idx] = p.Bp(1,0);
    gprms.pts_arrays[7][pt_idx] = p.Bp(1,1);
    gprms.pts_arrays[8][pt_idx] = p.Fe(0,0);
    gprms.pts_arrays[9][pt_idx] = p.Fe(0,1);
    gprms.pts_arrays[10][pt_idx] = p.Fe(1,0);
    gprms.pts_arrays[11][pt_idx] = p.Fe(1,1);
//    gprms.pts_arrays[12][pt_idx] = p.NACC_alpha_p;

    gprms.pts_arrays[16][pt_idx] = p.q;
}


__device__ void DruckerPragerUpdateDeformationGradient(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    constexpr real magic_epsilon = 1.e-15;
    constexpr int d = 2; // dimensions
    const real &mu = gprms.mu;
    const real &lambda = gprms.lambda;
    const real &dt = gprms.InitialTimeStep;

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

//    if(e_c.norm() < magic_epsilon || e_c.trace()>0)
    if(e_c.norm() ==0 || lnSigma.trace()>0)
    {
        // Projection to the tip of the cone
        Tmatrix.setIdentity();
        dq = lnSigma.norm();
    }
    else
    {
        constexpr double PI = 3.1415927;
        constexpr double H0 = 35 * PI / 180.0f;
        constexpr double H1 = 9 * PI / 180.0f;
        constexpr double H2 = 0.2f;
        constexpr double H3 = 10 * PI / 180.0f;

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
//    p.Fp = V*Tmatrix.inverse()*Sigma*V.transpose()*p.Fp;

    // hardening
    p.q += dq;


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


void GPU_Implementation2::test()
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

void GPU_Implementation2::synchronize()
{
    cudaDeviceSynchronize();
}

