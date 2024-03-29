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

using namespace Eigen;

__device__ int gpu_error_indicator;
__constant__ icy::SimParams gprms;

constexpr double d = 2; // dimensions
constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);

__device__ Matrix2d KirchhoffStress_Wolper(const Matrix2d &F)
{
    const double &kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    double Je = F.determinant();
    Matrix2d b = F*F.transpose();
    Matrix2d PFt = mu*(1/Je)*dev(b) + kappa*(Je*Je-1.)*Matrix2d::Identity();
    return PFt;
}

__device__ void ComputePQ(icy::Point &p, const double &kappa, const double &mu)
{
    svd2x2_modified(p.Fe, p.U, p.vSigma, p.V);
    p.Je_tr = p.vSigma.prod();         // product of elements of vSigma (representation of diagonal matrix)
    p.p_tr = -(kappa/2.) * (p.Je_tr*p.Je_tr - 1.);
    p.vSigmaSquared = p.vSigma.array().square().matrix();
    p.v_s_hat_tr = mu/p.Je_tr * dev_d(p.vSigmaSquared); //mu * pow(Je_tr,-2./d)* dev(SigmaSquared);
    p.q_tr = coeff1*p.v_s_hat_tr.norm();
}



__device__ void Wolper_Drucker_Prager(icy::Point &p)
{
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;
    const double &tan_phi = gprms.DP_tan_phi;
    const double &DP_threshold_p = gprms.DP_threshold_p;

//    const double &pmin = -gprms.IceTensileStrength;
    const double &pmax = gprms.IceCompressiveStrength;
    const double &qmax = gprms.IceShearStrength;

    if(p.p_tr < -DP_threshold_p || p.Jp_inv < 1)
    {
        // tear in tension or compress until original state
        double p_new = -DP_threshold_p;
        double Je_new = sqrt(-2.*p_new/kappa + 1.);
        Vector2d vSigma_new = Vector2d::Constant(1.)*sqrt(Je_new);  //Matrix2d::Identity() * pow(Je_new, 1./(double)d);
        p.Fe = p.U*vSigma_new.asDiagonal()*p.V.transpose();
        p.Jp_inv *= Je_new/p.Je_tr;
    }
    else
    {
        double q_n_1;

        if(p.p_tr > pmax)
        {
            q_n_1 = 0;
        }
        else
        {
            double q_from_dp = (p.p_tr+DP_threshold_p)*tan_phi;
            q_n_1 = min(q_from_dp,qmax);
//            double q_from_failure_surface = 2*sqrt((pmax-p.p_tr)*(p.p_tr-pmin))*qmax/(pmax-pmin);
//            q_n_1 = min(q_from_failure_surface, q_from_dp);
        }

        if(p.q_tr >= q_n_1)
        {
            // project onto YS
            double s_hat_n_1_norm = q_n_1/coeff1;
//            Matrix2d B_hat_E_new = s_hat_n_1_norm*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2d::Identity()*(SigmaSquared.trace()/d);
            Vector2d vB_hat_E_new = s_hat_n_1_norm*(p.Je_tr/mu)*p.v_s_hat_tr.normalized() + Vector2d::Constant(1.)*(p.vSigmaSquared.sum()/d);
            Vector2d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
            p.Fe = p.U*vSigma_new.asDiagonal()*p.V.transpose();
        }
    }
}


__device__ void GetParametersForGrain(short grain, double &pmin, double &pmax, double &qmax, double &beta, double &mSq)
{
    double var1 = 1.0 + gprms.GrainVariability*0.05*(-10 + grain%21);
    double var2 = 1.0 + gprms.GrainVariability*0.033*(-15 + (grain+3)%30);
    double var3 = 1.0 + gprms.GrainVariability*0.1*(-10 + (grain+4)%11);

    pmax = gprms.IceCompressiveStrength * var1;
    pmin = -gprms.IceTensileStrength * var2;
    qmax = gprms.IceShearStrength * var3;

    beta = -pmin / pmax;
//    double NACC_M = (2*qmax*sqrt(1+2*beta))/(pmax*(1+beta));
    mSq = (4*qmax*qmax*(1+2*beta))/((pmax*(1+beta))*(pmax*(1+beta)));
//    mSq = NACC_M*NACC_M;
}


__device__ void CheckIfPointIsInsideFailureSurface(icy::Point &p)
{
    /*
    const double &beta = gprms.NACC_beta;
    const double &M_sq = gprms.NACC_Msq;
    const double &pmin = -gprms.IceTensileStrength;
    const double &pmax = gprms.IceCompressiveStrength;;
    const double &qmax = gprms.IceShearStrength;
    */

    double beta, M_sq, pmin, pmax, qmax;
    GetParametersForGrain(p.grain, pmin, pmax, qmax, beta, M_sq);

    const double pmin2 = -3e6;
    if(p.p_tr<0)
    {
        if(p.p_tr<pmin2) {p.crushed = 1; return;}
        double q0 = 2*sqrt(-pmax*pmin)*qmax/(pmax-pmin);
        double k = -q0/pmin2;
        double q_limit = k*(p.p_tr-pmin2);
        if(p.q_tr > q_limit) {p.crushed = 1; return;}
    }
    else
    {
        double y = (1.+2.*beta)*p.q_tr*p.q_tr + M_sq*(p.p_tr + beta*pmax)*(p.p_tr - pmax);
        if(y > 0) p.crushed = 1;
    }
}



// ==============================  kernels  ====================================


__global__ void v2_kernel_p2g()
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const double &dt = gprms.InitialTimeStep;
    const double &vol = gprms.ParticleVolume;
    const double &h = gprms.cellsize;
    const double &h_inv = gprms.cellsize_inv;
    const double &Dinv = gprms.Dp_inv;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const double &particle_mass = gprms.ParticleMass;
    const int &nGridPitch = gprms.nGridPitch;
    const int &pitch = gprms.nPtsPitch;

    // pull point data from SOA
    const double *buffer = gprms.pts_array;

    Vector2d pos, velocity;
    Matrix2d Bp, Fe;

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

    Matrix2d PFt = KirchhoffStress_Wolper(Fe);
    Matrix2d subterm2 = particle_mass*Bp - (dt*vol*Dinv)*PFt;

    Eigen::Vector2i base_coord_i = (pos*h_inv - Vector2d::Constant(0.5)).cast<int>(); // coords of base grid node for point
    Vector2d base_coord = base_coord_i.cast<double>();
    Vector2d fx = pos*h_inv - base_coord;

    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    Array2d arr_v0 = 1.5-fx.array();
    Array2d arr_v1 = fx.array() - 1.0;
    Array2d arr_v2 = fx.array() - 0.5;
    Array2d ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            double Wip = ww[i][0]*ww[j][1];
            Vector2d dpos((i-fx[0])*h, (j-fx[1])*h);
            Vector2d incV = Wip*(velocity*particle_mass + subterm2*dpos);
            double incM = Wip*particle_mass;

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

__global__ void v2_kernel_update_nodes(double indenter_x, double indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nGridNodes = gprms.GridTotal;
    if(idx >= nGridNodes) return;

    double mass = gprms.grid_array[idx];
    if(mass == 0) return;

    const int &nGridPitch = gprms.nGridPitch;
    const double &gravity = gprms.Gravity;
    const double &indRsq = gprms.IndRSq;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const double &dt = gprms.InitialTimeStep;
    const double &ind_velocity = gprms.IndVelocity;
    const double &cellsize = gprms.cellsize;

    const Vector2d vco(ind_velocity,0);  // velocity of the collision object (indenter)
    const Vector2d indCenter(indenter_x, indenter_y);

    Vector2d velocity(gprms.grid_array[1*nGridPitch + idx], gprms.grid_array[2*nGridPitch + idx]);
    velocity /= mass;
    velocity[1] -= dt*gravity;
    double vmax = 0.5*cellsize/dt;
    if(velocity.norm() > vmax) velocity = velocity/velocity.norm()*vmax;

    int idx_x = idx % gridX;
    int idx_y = idx / gridX;

    // indenter
    Vector2d gnpos(idx_x*cellsize, idx_y*cellsize);
    Vector2d n = gnpos - indCenter;
    if(n.squaredNorm() < indRsq)
    {
        // grid node is inside the indenter
        Vector2d vrel = velocity - vco;
        n.normalize();
        double vn = vrel.dot(n);   // normal component of the velocity
        if(vn < 0)
        {
            Vector2d vt = vrel - n*vn;   // tangential portion of relative velocity
            Vector2d prev_velocity = velocity;
            velocity = vco + vt;

            // force on the indenter
            Vector2d force = (prev_velocity-velocity)*mass/dt;
            double angle = atan2(n[0],n[1]);
            angle += icy::SimParams::pi;
            angle *= gprms.n_indenter_subdivisions/ (2*icy::SimParams::pi);
            int index = (int)angle;
            index = max(index, 0);
            index = min(index, gprms.n_indenter_subdivisions-1);
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



__global__ void v2_kernel_g2p(bool recordPQ)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int &nPoints = gprms.nPts;
    if(pt_idx >= nPoints) return;

    const int &pitch_pts = gprms.nPtsPitch;
    const int &pitch_grid = gprms.nGridPitch;
    const double &h_inv = gprms.cellsize_inv;
    const double &dt = gprms.InitialTimeStep;
    const int &gridX = gprms.GridX;
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;

    icy::Point p;
    p.velocity.setZero();
    p.Bp.setZero();
    double *buffer = gprms.pts_array;
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

    // coords of base grid node for point
    Eigen::Vector2i base_coord_i = (p.pos*h_inv - Vector2d::Constant(0.5)).cast<int>();
    Vector2d base_coord = base_coord_i.cast<double>();
    Vector2d fx = p.pos*h_inv - base_coord;

    // optimized method of computing the quadratic weight function without conditional operators
    Array2d arr_v0 = 1.5 - fx.array();
    Array2d arr_v1 = fx.array() - 1.0;
    Array2d arr_v2 = fx.array() - 0.5;
    Array2d ww[3] = {0.5*arr_v0*arr_v0, 0.75-arr_v1*arr_v1, 0.5*arr_v2*arr_v2};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            Vector2d dpos = Vector2d(i, j) - fx;
            double weight = ww[i][0]*ww[j][1];
            int idx_gridnode = i+base_coord_i[0] + (j+base_coord_i[1])*gridX;
            Vector2d node_velocity;
            node_velocity[0] = gprms.grid_array[1*pitch_grid + idx_gridnode];
            node_velocity[1] = gprms.grid_array[2*pitch_grid + idx_gridnode];
            p.velocity += weight * node_velocity;
            p.Bp += (4.*h_inv)*weight *(node_velocity*dpos.transpose());
        }

    // Advection and update of the deformation gradient
    p.pos += p.velocity * dt;
    const Matrix2d &gradV = p.Bp;
    p.Fe = (Matrix2d::Identity() + dt*gradV) * p.Fe;

    ComputePQ(p, kappa, mu);    // pre-computes USV, p, q, etc.

    if(p.crushed == 0) CheckIfPointIsInsideFailureSurface(p);
    if(p.crushed == 1) Wolper_Drucker_Prager(p);

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

    ptr_intact[pt_idx] = p.crushed;

    if(recordPQ)
    {
        buffer[pt_idx + pitch_pts*icy::SimParams::idx_P] = p.p_tr;
        buffer[pt_idx + pitch_pts*icy::SimParams::idx_Q] = p.q_tr;
    }
}

//===========================================================================



// deviatoric part of a diagonal matrix
__device__ Vector2d dev_d(Vector2d Adiag)
{
    return Adiag - Adiag.sum()/2*Vector2d::Constant(1.);
}

__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A)
{
    return A - A.trace()/2*Eigen::Matrix2d::Identity();
}


//===========================================================================

__device__ void svd(const double a[4], double u[4], double sigma[2], double v[4])
{
    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(a, gu, sigma, gv);
    gu.template fill<2, double>(u);
    gv.template fill<2, double>(v);
}

__device__ void svd2x2_modified(const Matrix2d &mA, Matrix2d &mU, Vector2d &mS, Matrix2d &mV)
{
    double U[4], V[4], S[2];
    double a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);
    mU << U[0],U[1],U[2],U[3];
    mS << S[0],S[1];
    mV << V[0],V[1],V[2],V[3];
}

__device__ Matrix2d polar_decomp_R(const Matrix2d &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    double th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Matrix2d result;
    result << cosf(th), -sinf(th), sinf(th), cosf(th);
    return result;
}

__global__ void kernel_hello()
{
    printf("hello from CUDA\n");
}


void GPU_Implementation5::test()
{
    kernel_hello<<<1,1,0,streamCompute>>>();
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("cuda test");
    else std::cout << "hello kernel executed successfully\n";
    cudaDeviceSynchronize();
}

void GPU_Implementation5::synchronize()
{
    if(!initialized) return;
    cudaDeviceSynchronize();
}


// ========================================= initialization and kernel execution


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

    err = cudaMallocPitch (&model->prms.grid_array, &model->prms.nGridPitch, sizeof(double)*nGridNodes, icy::SimParams::nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    model->prms.nGridPitch /= sizeof(double); // assume that this divides without remainder

    // device memory for points
    err = cudaMallocPitch (&model->prms.pts_array, &model->prms.nPtsPitch, sizeof(double)*nPoints, icy::SimParams::nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    model->prms.nPtsPitch /= sizeof(double);

    err = cudaMalloc(&model->prms.indenter_force_accumulator, model->prms.IndenterArraySize());
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");

    // pinned host memory
    err = cudaMallocHost(&tmp_transfer_buffer, sizeof(double)*model->prms.nPtsPitch*icy::SimParams::nPtsArrays);
    if(err!=cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    err = cudaMallocHost(&host_side_indenter_force_accumulator, model->prms.IndenterArraySize());
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
                                 pitch*sizeof(double)*icy::SimParams::nPtsArrays, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    memset(host_side_indenter_force_accumulator, 0, model->prms.IndenterArraySize());
    spdlog::info("transfer_ponts_to_device() done");
}

void GPU_Implementation5::cuda_transfer_from_device()
{
    cudaError_t err = cudaMemcpyAsync(tmp_transfer_buffer, model->prms.pts_array,
                                      model->prms.nPtsPitch*sizeof(double)*icy::SimParams::nPtsArrays,
                                      cudaMemcpyDeviceToHost, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");

    err = cudaMemcpyAsync(host_side_indenter_force_accumulator, model->prms.indenter_force_accumulator,
                          model->prms.IndenterArraySize(), cudaMemcpyDeviceToHost, streamCompute);
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
                                      model->prms.nGridPitch*icy::SimParams::nGridArrays*sizeof(double), streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
}

void GPU_Implementation5::cuda_reset_indenter_force_accumulator()
{
    cudaError_t err = cudaMemsetAsync(model->prms.indenter_force_accumulator, 0, model->prms.IndenterArraySize(), streamCompute);
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

void GPU_Implementation5::cuda_update_nodes(double indenter_x, double indenter_y)
{
    const int nGridNodes = model->prms.GridTotal;
    int tpb = model->prms.tpb_Upd;
    int blocksPerGrid = (nGridNodes + tpb - 1) / tpb;
    v2_kernel_update_nodes<<<blocksPerGrid, tpb, 0, streamCompute>>>(indenter_x, indenter_y);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("cuda_update_nodes");
}

void GPU_Implementation5::cuda_g2p(bool recordPQ)
{
    const int nPoints = model->prms.nPts;
    int tpb = model->prms.tpb_G2P;
    int blocksPerGrid = (nPoints + tpb - 1) / tpb;
    v2_kernel_g2p<<<blocksPerGrid, tpb, 0, streamCompute>>>(recordPQ);
    if(cudaGetLastError() != cudaSuccess) throw std::runtime_error("cuda_g2p");
}


