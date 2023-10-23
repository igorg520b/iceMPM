#include "gpu_implementation2.h"
#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "helper_math.cuh"

__device__ real *device_grid_arrays[GPU_Implementation2::nGridArrays];
__device__ real *device_pts_arrays[GPU_Implementation2::nPtsArrays];

__constant__ icy::SimParams gprms;
__device__ int gpu_error_indicator;


GPU_Implementation2::GPU_Implementation2()
{
    cudaEventCreate(&eventTimingStart);
    cudaEventCreate(&eventTimingStop);

    cudaEventCreate(&eventCycleComplete);
    cudaEventCreate(&eventDataCopiedToHost);

    cudaStreamCreate(&streamCompute);
    cudaStreamCreate(&streamTransfer);
}




void GPU_Implementation2::cuda_update_constants(const icy::SimParams &prms)
{
    cudaError_t err;
    int error_code = 0;
    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("gpu_error_indicator initialization");

    err = cudaMemcpyToSymbol(gprms, &prms, sizeof(icy::SimParams));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");

    std::cout << "CUDA constants copied to device\n";
}

void GPU_Implementation2::cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
    cudaError_t err;

    // pinned host memory
    if(tmp_transfer_buffer) cudaFreeHost(tmp_transfer_buffer);
    err = cudaMallocHost(&tmp_transfer_buffer, sizeof(real)*nPoints*3);
    if(err!=cudaSuccess) throw std::runtime_error("GPU_Implementation2::Prepare(int nPoints)");

    // device memory for grid
    for(int k=0;k<nGridArrays;k++)
    {
        err = cudaMalloc(&grid_arrays[k], sizeof(real)*nGridNodes);
        if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    }

    // device memory for points
    for(int k=0;k<nPtsArrays;k++)
    {
        err = cudaMalloc(&pts_arrays[k], sizeof(real)*nPoints);
        if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays");
    }

    err = cudaMemcpyToSymbol(device_grid_arrays, grid_arrays, sizeof(void*)*nGridArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol device_grid_arrays");

    err = cudaMemcpyToSymbol(device_pts_arrays, pts_arrays, sizeof(void*)*nPtsArrays);
    if(err != cudaSuccess) throw std::runtime_error("cuda_allocate_arrays memcpytosymbol device_pts_arrays");

    std::cout << "cuda_allocate_arrays done\n";
}

void GPU_Implementation2::transfer_ponts_to_device(const std::vector<icy::Point> &points)
{
    cudaError_t err;
    int n = points.size();
    real *tmp = (real*) tmp_transfer_buffer;

    // transfer point positions
    for(int i=0;i<n;i++) tmp[i] = points[i].pos[0];
    err = cudaMemcpy(pts_arrays[0], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].pos[1];
    err = cudaMemcpy(pts_arrays[1], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer velocity
    for(int i=0;i<n;i++) tmp[i] = points[i].velocity[0];
    err = cudaMemcpy(pts_arrays[2], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].velocity[1];
    err = cudaMemcpy(pts_arrays[3], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer Bp
    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(0,0);
    err = cudaMemcpy(pts_arrays[4], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(0,1);
    err = cudaMemcpy(pts_arrays[5], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(1,0);
    err = cudaMemcpy(pts_arrays[6], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Bp(1,1);
    err = cudaMemcpy(pts_arrays[7], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer Fe
    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(0,0);
    err = cudaMemcpy(pts_arrays[8], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(0,1);
    err = cudaMemcpy(pts_arrays[9], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(1,0);
    err = cudaMemcpy(pts_arrays[10], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    for(int i=0;i<n;i++) tmp[i] = points[i].Fe(1,1);
    err = cudaMemcpy(pts_arrays[11], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");

    // transfer NACC_alpha
    for(int i=0;i<n;i++) tmp[i] = points[i].NACC_alpha_p;
    err = cudaMemcpy(pts_arrays[12], tmp_transfer_buffer, sizeof(real)*n, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) throw std::runtime_error("transfer_points_to_device");
}


void GPU_Implementation2::backup_point_positions(const int nPoints)
{
    cudaError_t err;
    err = cudaMemcpyAsync(pts_arrays[13], pts_arrays[0], sizeof(real)*nPoints, cudaMemcpyDeviceToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("backup_point_positions");
    err = cudaMemcpyAsync(pts_arrays[14], pts_arrays[1], sizeof(real)*nPoints, cudaMemcpyDeviceToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("backup_point_positions");
    err = cudaMemcpyAsync(pts_arrays[15], pts_arrays[12], sizeof(real)*nPoints, cudaMemcpyDeviceToDevice, streamCompute);
    if(err != cudaSuccess) throw std::runtime_error("backup_point_positions");
}


void GPU_Implementation2::cuda_transfer_from_device(std::vector<icy::Point> &points)
{
    int n = points.size();
    cudaError_t err;

    real* tmp2 = (real*)tmp_transfer_buffer + n;
    real* tmp3 = (real*)tmp_transfer_buffer + n*2;

    cudaStreamWaitEvent(streamTransfer, eventCycleComplete);    // wait until streamCompute has the data ready
    err = cudaMemcpyAsync(tmp_transfer_buffer, pts_arrays[0], sizeof(real)*n, cudaMemcpyDeviceToHost, streamTransfer);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");
    err = cudaMemcpyAsync((void*) tmp2, pts_arrays[1], sizeof(real)*n, cudaMemcpyDeviceToHost, streamTransfer);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");
    err = cudaMemcpyAsync((void*) tmp3, pts_arrays[12], sizeof(real)*n, cudaMemcpyDeviceToHost, streamTransfer);
    if(err != cudaSuccess) throw std::runtime_error("cuda_transfer_from_device");


    int error_code = 0;
    err = cudaMemcpyFromSymbolAsync(&error_code, gpu_error_indicator, sizeof(int), 0, cudaMemcpyDeviceToHost, streamTransfer);
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

void GPU_Implementation2::transfer_ponts_to_host_finalize(std::vector<icy::Point> &points)
{
    int n = points.size();
    real* tmp1 = (real*)tmp_transfer_buffer;
    real* tmp2 = (real*)tmp_transfer_buffer + n;
    real* tmp3 = (real*)tmp_transfer_buffer + n*2;
    for(int k=0;k<n;k++) points[k].pos[0] = tmp1[k];
    for(int k=0;k<n;k++) points[k].pos[1] = tmp2[k];
    for(int k=0;k<n;k++) points[k].NACC_alpha_p = tmp3[k];

}


void GPU_Implementation2::cuda_reset_grid(size_t nGridNodes)
{
    for(int k=0;k<3;k++)
    {
        cudaError_t err = cudaMemsetAsync(grid_arrays[k], 0, sizeof(real)*nGridNodes, streamCompute);
        if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid error");
    }
}



void GPU_Implementation2::cuda_p2g(const int nPoints)
{
    cudaError_t err;

    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    v2_kernel_p2g<<<blocksPerGrid, threadsPerBlock, 0, streamCompute>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_p2g error executing kernel_p2g " << err << std::endl;
        throw std::runtime_error("cuda_p2g");
    }
}


void GPU_Implementation2::cuda_g2p(const int nPoints)
{
    cudaError_t err;
    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    v2_kernel_g2p<<<blocksPerGrid, threadsPerBlock, 0, streamCompute>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_g2p error " << err << '\n';
        throw std::runtime_error("cuda_g2p");
    }
}


void GPU_Implementation2::cuda_update_nodes(const int nGridNodes,real indenter_x, real indenter_y)
{
    cudaError_t err;
    int blocksPerGrid = (nGridNodes + threadsPerBlock - 1) / threadsPerBlock;
    v2_kernel_update_nodes<<<blocksPerGrid, threadsPerBlock, 0, streamCompute>>>(nGridNodes, indenter_x, indenter_y);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_update_nodes\n";
        throw std::runtime_error("cuda_update_nodes");
    }
}






// ==============================  kernels  ====================================


__global__ void v2_kernel_p2g(const int nPoints)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    p.pos[0] = device_pts_arrays[0][pt_idx];
    p.pos[1] = device_pts_arrays[1][pt_idx];
    p.velocity[0] = device_pts_arrays[2][pt_idx];
    p.velocity[1] = device_pts_arrays[3][pt_idx];

    p.Bp(0,0) = device_pts_arrays[4][pt_idx];
    p.Bp(0,1) = device_pts_arrays[5][pt_idx];
    p.Bp(1,0) = device_pts_arrays[6][pt_idx];
    p.Bp(1,1) = device_pts_arrays[7][pt_idx];
    p.Fe(0,0) = device_pts_arrays[8][pt_idx];
    p.Fe(0,1) = device_pts_arrays[9][pt_idx];
    p.Fe(1,0) = device_pts_arrays[10][pt_idx];
    p.Fe(1,1) = device_pts_arrays[11][pt_idx];
//    p.NACC_alpha_p = device_pts_arrays[12][pt_idx];

    Matrix2r Re = polar_decomp_R(p.Fe);
    real Je = p.Fe.determinant();
    Matrix2r dFe = 2.*mu*(p.Fe - Re)* p.Fe.transpose() +
            lambda * (Je - 1.) * Je * Matrix2r::Identity();
    Matrix2r stress = - (dt * vol) * (Dinv * dFe);
    Matrix2r affine = stress + particle_mass * p.Bp;

    constexpr real offset = 0.5;  // 0 for cubic; 0.5 for quadratic
    const int i0 = (int)(p.pos[0]*h_inv - offset);
    const int j0 = (int)(p.pos[1]*h_inv - offset);

    Vector2r base_coord(i0,j0);
    Vector2r fx = p.pos*h_inv - base_coord;

    Vector2r v0(1.5-fx[0],1.5-fx[1]);
    Vector2r v1(fx[0]-1.,fx[1]-1.);
    Vector2r v2(fx[0]-.5,fx[1]-.5);

    Vector2r w[3];
    w[0] << .5*v0[0]*v0[0],  .5*v0[1]*v0[1];
    w[1] << .75-v1[0]*v1[0], .75-v1[1]*v1[1];
    w[2] << .5*v2[0]*v2[0],  .5*v2[1]*v2[1];

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            real Wip = w[i][0]*w[j][1];
            Vector2r dpos((i-fx[0])*h, (j-fx[1])*h);
            Vector2r incV = Wip*(p.velocity*particle_mass+affine*dpos);
            real incM = Wip*particle_mass;

            int idx_gridnode = (i+i0) + (j+j0)*gridX;
            if((i+i0) < 0 || (j+j0) < 0 || (i+i0) >=gridX || (j+j0)>=gridY || idx_gridnode < 0)
                gpu_error_indicator = 1;

            // Udpate mass, velocity and force
            atomicAdd(&device_grid_arrays[0][idx_gridnode], incM);
            atomicAdd(&device_grid_arrays[1][idx_gridnode], incV[0]);
            atomicAdd(&device_grid_arrays[2][idx_gridnode], incV[1]);
        }
}

__global__ void v2_kernel_update_nodes(const int nGridNodes, real indenter_x, real indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nGridNodes) return;

    icy::GridNode gn;
    gn.mass = device_grid_arrays[0][idx];
    if(gn.mass == 0) return;

    gn.velocity[0] = device_grid_arrays[1][idx];
    gn.velocity[1] = device_grid_arrays[2][idx];

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

    gn.velocity /= gn.mass;
    gn.velocity[1] -= dt*gravity;
    if(gn.velocity.norm() > cellsize/dt) gn.velocity = gn.velocity.normalized()*cellsize/dt;

    int idx_x = idx % gridX;
    int idx_y = idx / gridX;

    // indenter
    Vector2r gnpos(idx_x*cellsize, idx_y*cellsize);
    Vector2r n = gnpos - indCenter;
    if(n.squaredNorm() < indRsq)
    {
        // grid node is inside the indenter
        Vector2r vrel = gn.velocity - vco;
        n.normalize();
        real vn = vrel.dot(n);   // normal component of the velocity
        if(vn < 0)
        {
            Vector2r vt = vrel - n*vn;   // tangential portion of relative velocity
            gn.velocity = vco + vt + ice_friction_coeff*vn*vt.normalized();
        }
    }

    // attached bottom layer
    if(idx_y <= 3) gn.velocity.setZero();
    else if(idx_y >= gridY-4 && gn.velocity[1]>0) gn.velocity[1] = 0;
    if(idx_x <= 3 && gn.velocity.x()<0) gn.velocity[0] = 0;
    else if(idx_x >= gridX-5) gn.velocity[0] = 0;

    // write the updated grid velocity back to memory
    device_grid_arrays[1][idx] = gn.velocity[0];
    device_grid_arrays[2][idx] = gn.velocity[1];
}

__global__ void v2_kernel_g2p(const int nPoints)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPoints) return;

    icy::Point p;
    p.pos[0] = device_pts_arrays[0][pt_idx];
    p.pos[1] = device_pts_arrays[1][pt_idx];

    p.Fe(0,0) = device_pts_arrays[8][pt_idx];
    p.Fe(0,1) = device_pts_arrays[9][pt_idx];
    p.Fe(1,0) = device_pts_arrays[10][pt_idx];
    p.Fe(1,1) = device_pts_arrays[11][pt_idx];
    p.NACC_alpha_p = device_pts_arrays[12][pt_idx];

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

    Vector2r w[3];
    w[0] << .5*v0[0]*v0[0],  .5*v0[1]*v0[1];
    w[1] << .75-v1[0]*v1[0], .75-v1[1]*v1[1];
    w[2] << .5*v2[0]*v2[0],  .5*v2[1]*v2[1];

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            Vector2r dpos = Vector2r(i, j) - fx;
            real weight = w[i][0]*w[j][1];

            int idx_gridnode = i+i0 + (j+j0)*gridX;
            icy::GridNode node;
            node.mass = device_grid_arrays[0][idx_gridnode];
            node.velocity[0] = device_grid_arrays[1][idx_gridnode];
            node.velocity[1] = device_grid_arrays[2][idx_gridnode];

            const Vector2r &grid_v = node.velocity;
            p.velocity += weight * grid_v;
            p.Bp += (4.*h_inv)*weight *(grid_v*dpos.transpose());
        }

    // Advection
    p.pos += dt * p.velocity;

    NACCUpdateDeformationGradient(p);

    device_pts_arrays[0][pt_idx] = p.pos[0];
    device_pts_arrays[1][pt_idx] = p.pos[1];
    device_pts_arrays[2][pt_idx] = p.velocity[0];
    device_pts_arrays[3][pt_idx] = p.velocity[1];

    device_pts_arrays[4][pt_idx] = p.Bp(0,0);
    device_pts_arrays[5][pt_idx] = p.Bp(0,1);
    device_pts_arrays[6][pt_idx] = p.Bp(1,0);
    device_pts_arrays[7][pt_idx] = p.Bp(1,1);

    device_pts_arrays[8][pt_idx] = p.Fe(0,0);
    device_pts_arrays[9][pt_idx] = p.Fe(0,1);
    device_pts_arrays[10][pt_idx] = p.Fe(1,0);
    device_pts_arrays[11][pt_idx] = p.Fe(1,1);

    device_pts_arrays[12][pt_idx] = p.NACC_alpha_p;
}







//===========================================================================
/**
\brief 2x2 SVD (singular value decomposition) a=USV'
\param[in] a Input matrix.
\param[out] u Robustly a rotation matrix.
\param[out] sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
\param[out] v Robustly a rotation matrix.
*/
__device__ void svd(const real a[4], real u[4], real sigma[2], real v[4])
{
    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(a, gu, sigma, gv);
    gu.template fill<2, real>(u);
    gv.template fill<2, real>(v);
}


__device__ void svd2x2(const Matrix2r &mA,
                       Matrix2r &mU,
                       Matrix2r &mS,
                       Matrix2r &mV)
{
    real U[4], V[4], S[2];
    real a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);

    mU << U[0],U[1],U[2],U[3];
    mS << S[0],0,0,S[1];
    mV << V[0],V[1],V[2],V[3];
}


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
    real p0 = kappa * (magic_epsilon + sinh(xi * fmaxf(-alpha, 0.)));

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




__device__ Matrix2r polar_decomp_R(const Matrix2r &val)
{
    // polar decomposition
    // http://www.cs.cornell.edu/courses/cs4620/2014fa/lectures/polarnotes.pdf
    real th = atan2(val(1,0) - val(0,1), val(0,0) + val(1,1));
    Matrix2r result;
    result << cosf(th), -sinf(th), sinf(th), cosf(th);
    return result;
}

