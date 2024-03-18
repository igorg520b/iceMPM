#include "gpu_implementation5.h"
#include "parameters_sim.h"
#include "point.h"
#include "model.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include <spdlog/spdlog.h>

#include "helper_math.cuh"

__device__ int gpu_error_indicator;
__constant__ icy::SimParams gprms;

constexpr real d = 2; // dimensions


__forceinline__ __device__ Matrix2r KirchhoffStress_Wolper(const Matrix2r &F)
{
    const real &kappa = gprms.kappa;
    const real &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    real Je = F.determinant();
    Matrix2r b = F*F.transpose();
    Matrix2r PFt = mu*(1/Je)*(b-b.trace()*Matrix2r::Identity()/2) + kappa*(Je*Je-1.)*Matrix2r::Identity();
    return PFt;
}


__forceinline__ __device__ void Wolper_Drucker_Prager(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &dt = gprms.InitialTimeStep;
    const real &tan_phi = gprms.DP_tan_phi;
    const real &DP_threshold_p = gprms.DP_threshold_p;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;
    Matrix2r U, V; //, Sigma;
    Vector2r vSigma;
    svd2x2_modified(FeTr, U, vSigma, V);

    real Je_tr = vSigma.prod();         // product of elements of vSigma (representation of diagonal matrix)
    real p_trial = -(kappa/2.) * (Je_tr*Je_tr - 1.);
    Vector2r vSigmaSquared = vSigma.array().square().matrix();
    Vector2r v_s_hat_tr = mu/Je_tr * dev_d(vSigmaSquared); //mu * pow(Je_tr,-2./d)* dev(SigmaSquared);

    if(p_trial < -DP_threshold_p || p.Jp_inv < 1)
    {
//        if(p_trial < 1)  p.q = 1;
//        else if(p.Jp_inv < 1) p.q = 2;

        // tear in tension or compress until original state
        real p_new = -DP_threshold_p;
        real Je_new = sqrt(-2.*p_new/kappa + 1.);
        Vector2r vSigma_new = Vector2r::Constant(1.)*sqrt(Je_new);  //Matrix2r::Identity() * pow(Je_new, 1./(real)d);
        p.Fe = U*vSigma_new.asDiagonal()*V.transpose();
        p.Jp_inv *= Je_new/Je_tr;
    }
    else
    {
        constexpr real coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);
        real q_tr = coeff1*v_s_hat_tr.norm();
        real q_n_1 = (p_trial+DP_threshold_p)*tan_phi;
        q_n_1 = min(gprms.IceShearStrength, q_n_1);

        if(q_tr < q_n_1)
        {
            // elastic regime
            p.Fe = FeTr;
//            p.q = 4;
        }
        else
        {
            // project onto YS
            real s_hat_n_1_norm = q_n_1/coeff1;
//            Matrix2r B_hat_E_new = s_hat_n_1_norm*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2r::Identity()*(SigmaSquared.trace()/d);
            Vector2r vB_hat_E_new = s_hat_n_1_norm*(Je_tr/mu)*v_s_hat_tr.normalized() + Vector2r::Constant(1.)*(vSigmaSquared.sum()/d);
            Vector2r vSigma_new = vB_hat_E_new.array().sqrt().matrix();
            p.Fe = U*vSigma_new.asDiagonal()*V.transpose();
//            p.q = 3;
        }
    }
}


__forceinline__ __device__ void CheckIfPointIsInsideFailureSurface(icy::Point &p)
{
    const Matrix2r &gradV = p.Bp;
    const real &mu = gprms.mu;
    const real &kappa = gprms.kappa;
    const real &beta = gprms.NACC_beta;
    const real &dt = gprms.InitialTimeStep;
    const real &p0 = gprms.IceCompressiveStrength;
    const real &M_sq = gprms.NACC_Msq;

    Matrix2r FeTr = (Matrix2r::Identity() + dt*gradV) * p.Fe;
    p.Fe = FeTr;
    Matrix2r U, V;
    Vector2r vSigma;
    svd2x2_modified(FeTr, U, vSigma, V);

    real Je_tr = vSigma.prod();         // product of elements of vSigma (representation of diagonal matrix)
    real p_trial = -(kappa/2.) * (Je_tr*Je_tr - 1.);
    Vector2r vSigmaSquared = vSigma.array().square().matrix();
    Vector2r v_s_hat_tr = mu/Je_tr * dev_d(vSigmaSquared); //mu * pow(Je_tr,-2./d)* dev(SigmaSquared);

    real y = (1.+2.*beta)*(3.-d/2.)*v_s_hat_tr.squaredNorm() + M_sq*(p_trial + beta*p0)*(p_trial - p0);
    if(y > 0)
    {
        p.crushed = 1;
        p.crushed_status_modified = true;
    }
}



void GPU_Implementation5::initialize()
{
    if(initialized) return;
    cudaError_t err;
    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) throw std::runtime_error("initialize() cuda error");
    if(deviceCount == 0) throw std::runtime_error("No avaialble CUDA devices");
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    spdlog::info("Compute capability {}.{}", deviceProp.major, deviceProp.minor);
    cudaEventCreate(&eventCycleStart);
    cudaEventCreate(&eventCycleStop);
    err = cudaStreamCreate(&streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("GPU_Implementation3::initialize() cudaEventCreate");
    initialized = true;
    spdlog::info("GPU Implementation: prepared");
}

void GPU_Implementation5::cuda_update_constants()
{
    cudaError_t err;
    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("gpu_error_indicator initialization");
    err = cudaMemcpyToSymbol(gprms, &model->prms, sizeof(icy::SimParams));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");
    std::cout << "CUDA constants copied to device\n";
}

void GPU_Implementation5::cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
    if(!initialized) initialize();
    cudaError_t err;

    // device memory for grid
    cudaFree(model->prms.grid_array);
    cudaFree(model->prms.pts_array);
    cudaFree(model->prms.indenter_force_accumulator);
    cudaFreeHost(tmp_transfer_buffer);
    cudaFreeHost(host_side_indenter_force_accumulator);

    err = cudaMallocPitch (&model->prms.grid_array, &model->prms.nGridPitch, sizeof(real)*nGridNodes, icy::SimParams::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    model->prms.nGridPitch /= sizeof(real); // assume that this divides without remainder

    // device memory for points
    err = cudaMallocPitch (&model->prms.pts_array, &model->prms.nPtsPitch, sizeof(real)*nPoints, icy::SimParams::nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    model->prms.nPtsPitch /= sizeof(real);

    err = cudaMalloc(&model->prms.indenter_force_accumulator, sizeof(real)*icy::SimParams::n_indenter_subdivisions*2);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    // pinned host memory
    err = cudaMallocHost(&tmp_transfer_buffer, sizeof(real)*model->prms.nPtsPitch*icy::SimParams::nPtsArrays);
    if(err!=cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMallocHost(&host_side_indenter_force_accumulator, sizeof(real)*icy::SimParams::n_indenter_subdivisions*2);
    if(err!=cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    error_code = 0;
    spdlog::info("cuda_allocate_arrays done");
}

void GPU_Implementation5::transfer_ponts_to_device()
{
    spdlog::info("transfer_to_device()");
    int pitch = model->prms.nPtsPitch;
    // transfer point data to device
    cudaError_t err = cudaMemcpy(model->prms.pts_array, tmp_transfer_buffer,
                                 pitch*sizeof(real)*icy::SimParams::nPtsArrays, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    memset(host_side_indenter_force_accumulator, 0, sizeof(real)*icy::SimParams::n_indenter_subdivisions*2);
    spdlog::info("transfer_ponts_to_device() done");
}

void GPU_Implementation5::cuda_transfer_from_device()
{
    spdlog::info("cuda_transfer_from_device()");
    cudaError_t err = cudaMemcpyAsync(tmp_transfer_buffer, model->prms.pts_array,
                                      model->prms.nPtsPitch*sizeof(real)*icy::SimParams::nPtsArrays,
                                      cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyAsync(host_side_indenter_force_accumulator, model->prms.indenter_force_accumulator,
                          sizeof(real)*icy::SimParams::n_indenter_subdivisions*2,
                          cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(int), 0, cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    void* userData = reinterpret_cast<void*>(this);
    cudaStreamAddCallback(streamCompute, GPU_Implementation5::callback_from_stream, userData, 0);
}

void CUDART_CB GPU_Implementation5::callback_from_stream(cudaStream_t stream, cudaError_t status, void *userData)
{
    // simulation data was copied to host memory -> proceed with processing of this data
    GPU_Implementation5 *gpu = reinterpret_cast<GPU_Implementation5*>(userData);
    // any additional processing here
    if(gpu->transfer_completion_callback) gpu->transfer_completion_callback();
}

void GPU_Implementation5::cuda_reset_grid()
{
    cudaError_t err = cudaMemsetAsync(model->prms.grid_array, 0,
                                      model->prms.nGridPitch*icy::SimParams::nGridArrays*sizeof(real), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Implementation5::cuda_reset_indenter_force_accumulator()
{
    cudaError_t err = cudaMemsetAsync(model->prms.indenter_force_accumulator, 0,
                                      sizeof(real)*icy::SimParams::n_indenter_subdivisions*2, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}


void GPU_Implementation5::cuda_p2g()
{
    const int nPoints = model->prms.nPts;
    int tpb = model->prms.tpb_P2G;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    v2_kernel_p2g<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("cuda_p2g");
}

void GPU_Implementation5::cuda_update_nodes(real indenter_x, real indenter_y)
{
    const int nGridNodes = model->prms.GridTotal;
    int tpb = model->prms.tpb_Upd;
    int blocksPerGrid = (nGridNodes + tpb - 1) / tpb;
    v2_kernel_update_nodes<<<blocksPerGrid, tpb, 0, streamCompute>>>(indenter_x, indenter_y);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("cuda_update_nodes");
}

void GPU_Implementation5::cuda_g2p()
{
    const int nPoints = model->prms.nPts;
    int tpb = model->prms.tpb_G2P;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    v2_kernel_g2p<<<blocksPerGrid, tpb, 0, streamCompute>>>();
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("cuda_g2p");
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
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const real &particle_mass = gprms.ParticleMass;
    const int &nGridPitch = gprms.nGridPitch;
    const int &pitch = gprms.nPtsPitch;

    // pull point data from SOA
    const real *buffer = gprms.pts_array;

    Vector2r pos, velocity;
    Matrix2r Bp, Fe;

    for(int i=0; i<icy::SimParams::dim; i++)
    {
        pos[i] = buffer[pt_idx + pitch*(icy::SimParams::posx+i)];
        velocity[i] = buffer[pt_idx + pitch*(icy::SimParams::velx+i)];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            Fe(i,j) = buffer[pt_idx + pitch*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)];
            Bp(i,j) = buffer[pt_idx + pitch*(icy::SimParams::Bp00 + i*icy::SimParams::dim + j)];
        }
    }

    Matrix2r PFt = KirchhoffStress_Wolper(Fe);
    Matrix2r subterm2 = particle_mass*Bp - (dt*vol*Dinv)*PFt;

    Eigen::Vector2i base_coord_i = (pos*h_inv - Vector2r::Constant(0.5)).cast<int>(); // coords of base grid node for point
    Vector2r base_coord = base_coord_i.cast<real>();
    Vector2r fx = pos*h_inv - base_coord;

    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    Array2r arr_v0 = 1.5-fx.array();
    Array2r arr_v1 = fx.array() - 1.0;
    Array2r arr_v2 = fx.array() - 0.5;
    Array2r ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            real Wip = ww[i][0]*ww[j][1];
            Vector2r dpos((i-fx[0])*h, (j-fx[1])*h);
            Vector2r incV = Wip*(velocity*particle_mass + subterm2*dpos);
            real incM = Wip*particle_mass;

            int i2 = i+base_coord_i[0];
            int j2 = j+base_coord_i[1];
            int idx_gridnode = (i+base_coord_i[0]) + (j+base_coord_i[1])*gridX;
            if(i2<0 || j2<0 || i2>=gridX || j2>=gridY) gpu_error_indicator = 1;

            // Udpate mass, velocity and force
            atomicAdd(&gprms.grid_array[0*nGridPitch + idx_gridnode], incM);
            atomicAdd(&gprms.grid_array[1*nGridPitch + idx_gridnode], incV[0]);
            atomicAdd(&gprms.grid_array[2*nGridPitch + idx_gridnode], incV[1]);
        }
}

__global__ void v2_kernel_update_nodes(real indenter_x, real indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nGridNodes = gprms.GridTotal;
    if(idx >= nGridNodes) return;

    real mass = gprms.grid_array[idx];
    if(mass == 0) return;

    const int &nGridPitch = gprms.nGridPitch;
    const real &gravity = gprms.Gravity;
    const real &indRsq = gprms.IndRSq;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const real &dt = gprms.InitialTimeStep;
    const real &ind_velocity = gprms.IndVelocity;
    const real &cellsize = gprms.cellsize;

    const Vector2r vco(ind_velocity,0);  // velocity of the collision object (indenter)
    const Vector2r indCenter(indenter_x, indenter_y);

    Vector2r velocity(gprms.grid_array[1*nGridPitch + idx], gprms.grid_array[2*nGridPitch + idx]);
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
            Vector2r prev_velocity = velocity;
            velocity = vco + vt;

            // force on the indenter
            Vector2r force = (prev_velocity-velocity)*mass/dt;
            double angle = atan2(n[0],n[1]);
            angle += icy::SimParams::pi;
            angle *= icy::SimParams::n_indenter_subdivisions/ (2*icy::SimParams::pi);
            int index = (int)angle;
            index = max(index, 0);
            index = min(index, icy::SimParams::n_indenter_subdivisions-1);
            atomicAdd(&gprms.indenter_force_accumulator[0+2*index], force[0]);
            atomicAdd(&gprms.indenter_force_accumulator[1+2*index], force[1]);
        }
    }

    // attached bottom layer
    if(idx_y <= 2) velocity.setZero();
    else if(idx_y >= gridY-3 && velocity[1]>0) velocity[1] = 0;
    if(idx_x <= 2 && velocity[0]<0) velocity[0] = 0;
    else if(idx_x >= gridX-3 && velocity[0]>0) velocity[0] = 0;

    /*
    // side boundary conditions
    int blocksGridX = gprms.BlockLength*gprms.cellsize_inv+5-2;
    int blocksGridY = gprms.BlockHeight/2*gprms.cellsize_inv+2;
    if(idx_x >= blocksGridX && idx_x <= blocksGridX + 2 && idx_y < blocksGridY) velocity.setZero();
    if(idx_x <= 7 && idx_x > 4 && idx_y < blocksGridY) velocity.setZero();
*/

    // write the updated grid velocity back to memory
    gprms.grid_array[1*nGridPitch + idx] = velocity[0];
    gprms.grid_array[2*nGridPitch + idx] = velocity[1];
}

__global__ void v2_kernel_g2p()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const int &pitch_pts = gprms.nPtsPitch;
    const int &pitch_grid = gprms.nGridPitch;
    const real &h_inv = gprms.cellsize_inv;
    const real &dt = gprms.InitialTimeStep;
    const int &gridX = gprms.GridX;

    icy::Point p;
    p.crushed_status_modified = false;
    p.velocity.setZero();
    p.Bp.setZero();
    real *buffer = gprms.pts_array;
    for(int i=0; i<icy::SimParams::dim; i++)
    {
        p.pos[i] = buffer[pt_idx + pitch_pts*(icy::SimParams::posx+i)];
        for(int j=0; j<icy::SimParams::dim; j++)
            p.Fe(i,j) = buffer[pt_idx + pitch_pts*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)];
    }
    char* ptr_intact = (char*)(&buffer[pitch_pts*icy::SimParams::idx_utility_data]);
    p.crushed = ptr_intact[pt_idx];
    short* ptr_grain = (short*)(&ptr_intact[pitch_pts]);
    p.grain = ptr_grain[pt_idx];
    p.Jp_inv = buffer[pt_idx + pitch_pts*icy::SimParams::idx_Jp_inv];

    Eigen::Vector2i base_coord_i = (p.pos*h_inv - Vector2r::Constant(0.5)).cast<int>(); // coords of base grid node for point
    Vector2r base_coord = base_coord_i.cast<real>();
    Vector2r fx = p.pos*h_inv - base_coord;

    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    Array2r arr_v0 = 1.5-fx.array();
    Array2r arr_v1 = fx.array() - 1.0;
    Array2r arr_v2 = fx.array() - 0.5;
    Array2r ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            Vector2r dpos = Vector2r(i, j) - fx;
            real weight = ww[i][0]*ww[j][1];
            int idx_gridnode = i+base_coord_i[0] + (j+base_coord_i[1])*gridX;
            Vector2r node_velocity;
            node_velocity[0] = gprms.grid_array[1*pitch_grid + idx_gridnode];
            node_velocity[1] = gprms.grid_array[2*pitch_grid + idx_gridnode];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection
    p.pos += dt * p.velocity;

    if(p.crushed == 0) CheckIfPointIsInsideFailureSurface(p);
    else Wolper_Drucker_Prager(p);

    // distribute the values of p back into GPU memory: pos, velocity, BP, Fe, Jp_inv, q
    buffer[pt_idx + pitch_pts*icy::SimParams::idx_Jp_inv] = p.Jp_inv;

    for(int i=0; i<icy::SimParams::dim; i++)
    {
        buffer[pt_idx + pitch_pts*(icy::SimParams::posx+i)] = p.pos[i];
        buffer[pt_idx + pitch_pts*(icy::SimParams::velx+i)] = p.velocity[i];
        for(int j=0; j<icy::SimParams::dim; j++)
        {
            buffer[pt_idx + pitch_pts*(icy::SimParams::Fe00 + i*icy::SimParams::dim + j)] = p.Fe(i,j);
            buffer[pt_idx + pitch_pts*(icy::SimParams::Bp00 + i*icy::SimParams::dim + j)] = p.Bp(i,j);
        }
    }

    if(p.crushed_status_modified) ptr_intact[pt_idx] = p.crushed;
}

//===========================================================================



// deviatoric part of a diagonal matrix
__forceinline__ __device__ Vector2r dev_d(Vector2r Adiag)
{
    return Adiag - Adiag.sum()/2*Vector2r::Constant(1.);
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

__device__ void svd2x2_modified(const Matrix2r &mA, Matrix2r &mU, Vector2r &mS, Matrix2r &mV)
{
    real U[4], V[4], S[2];
    real a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);
    mU << U[0],U[1],U[2],U[3];
    mS << S[0],S[1];
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


void GPU_Implementation5::test()
{
    cudaError_t err;
    kernel_hello<<<1,1,0,streamCompute>>>();
    if(cudaGetLastError() != cudaSuccess)
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

void GPU_Implementation5::synchronize()
{
    if(!initialized) return;
    cudaDeviceSynchronize();
}

