#include "gpu_implementation0.h"
#include "parameters_sim.h"
#include "point.h"
#include "gridnode.h"
#include <stdio.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>



__constant__ icy::SimParams gprms;
__device__ icy::Point *gpu_points;
__device__ icy::GridNode *gpu_nodes;
__device__ int gpu_error_indicator;


GPU_Implementation0::GPU_Implementation0()
{
    test_cuda();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}


void GPU_Implementation0::start_timing()
{
    cudaEventRecord(start);
}

float GPU_Implementation0::end_timing()
{
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}



void GPU_Implementation0::test_cuda()
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

//    std::cout << "A=\n" << A << '\n';
//    std::cout << "U=\n" << U << '\n';
//    std::cout << "S=\n" << S << '\n';
//    std::cout << "V=\n" << V << '\n';
//    std::cout << "USV^T=\n" << U*S*V.transpose() << '\n';
    cudaFree(results);

}





void GPU_Implementation0::cuda_update_constants(const icy::SimParams &prms)
{
    cudaError_t err;
    int error_code = 0;
    err = cudaMemcpyToSymbol(gpu_error_indicator, &error_code, sizeof(int));
    if(err != cudaSuccess) throw std::runtime_error("gpu_error_indicator initialization");

    err = cudaMemcpyToSymbol(gprms, &prms, sizeof(icy::SimParams));
    if(err!=cudaSuccess) throw std::runtime_error("cuda_update_constants: gprms");

    std::cout << "CUDA constants copied to device\n";
}

void GPU_Implementation0::cuda_allocate_arrays(size_t nGridNodes, size_t nPoints)
{
    cudaFree((void*)gpu_points_);
    cudaFree((void*)gpu_nodes_);

    cudaError_t err;

    err = cudaMalloc(&gpu_points_, sizeof(icy::Point)*nPoints);
    if(err != cudaSuccess)
    {
        std::cout << "cuda_allocate_arrays can't allocate\n";
        throw std::runtime_error("cuda_allocate_arrays");
    }

    err = cudaMemcpyToSymbol(gpu_points, &gpu_points_, sizeof(gpu_points_));
    if(err != cudaSuccess)
    {
        std::cout << "cuda_allocate_arrays cudaMemcpyToSymbol error\n";
        throw std::runtime_error("cuda_allocate_arrays");
    }

    err = cudaMalloc(&gpu_nodes_, sizeof(icy::GridNode)*nGridNodes);
    if(err != cudaSuccess)
    {
        std::cout << "cuda_allocate_arrays can't allocate\n";
        throw std::runtime_error("cuda_allocate_arrays");
    }
    err = cudaMemcpyToSymbol(gpu_nodes, &gpu_nodes_, sizeof(gpu_nodes_));
    if(err != cudaSuccess)
    {
        std::cout << "cuda_allocate_arrays cudaMemcpyToSymbol error\n";
        throw std::runtime_error("cuda_allocate_arrays");
    }
    std::cout << "cuda_allocate_arrays done\n";

}

void GPU_Implementation0::transfer_ponts_to_device(size_t nPoints, void* hostSource)
{
    cudaError_t err;
    err = cudaMemcpy(gpu_points_, hostSource, nPoints*sizeof(icy::Point), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        std::cout << "transfer_ponts_to_device failed with code \n";
        throw std::runtime_error("transfer_ponts_to_device");
    }
}

void GPU_Implementation0::cuda_transfer_from_device(size_t nPoints, void *hostArray)
{
    cudaError_t err;
    err = cudaMemcpy(hostArray, gpu_points_, nPoints*sizeof(icy::Point), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        std::cout << "cuda_transfer_from_device failed\n";
        throw std::runtime_error("cuda_transfer_from_device");
    }

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

void GPU_Implementation0::cuda_device_synchronize()
{
    cudaError_t err;
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_device_synchronize failed\n";
        throw std::runtime_error("cuda_device_synchronize");
    }
}

void GPU_Implementation0::cuda_reset_grid(size_t nGridNodes)
{
    cudaError_t err = cudaMemsetAsync(gpu_nodes_, 0, sizeof(icy::GridNode)*nGridNodes);
    if(err != cudaSuccess) throw std::runtime_error("cuda_reset_grid memset error");
}



void GPU_Implementation0::cuda_p2g(const int nPoints)
{
    cudaError_t err;

    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    kernel_p2g<<<blocksPerGrid, threadsPerBlock>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_p2g error executing kernel_p2g\n";
        throw std::runtime_error("cuda_p2g");
    }
}


void GPU_Implementation0::cuda_g2p(const int nPoints)
{
    cudaError_t err;
    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    kernel_g2p<<<blocksPerGrid, threadsPerBlock>>>(nPoints);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_g2p error\n";
        throw std::runtime_error("cuda_g2p");
    }
}


void GPU_Implementation0::cuda_update_nodes(const int nGridNodes,float indenter_x, float indenter_y)
{
    cudaError_t err;
    int blocksPerGrid = (nGridNodes + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_nodes<<<blocksPerGrid, threadsPerBlock>>>(nGridNodes, indenter_x, indenter_y);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "cuda_update_nodes\n";
        throw std::runtime_error("cuda_update_nodes");
    }
}






// ==============================  kernels  ====================================

__global__ void cuda_hello(Eigen::Matrix2f A, Eigen::Matrix2f *result)
{
    Eigen::Matrix2f &U = result[0];
    Eigen::Matrix2f &Sigma = result[1];
    Eigen::Matrix2f &V = result[2];
    svd2x2(A, U, Sigma, V);
    printf("Hello World from GPU!\n\n");
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

    const float &particle_volume = gprms.ParticleVolume;
    const float &cellsize = gprms.cellsize;
    const float &Dp_inv = gprms.Dp_inv;
    const float &lambda = gprms.lambda;
    const float &mu = gprms.mu;
    const int &gridX = gprms.GridX;
    const int &gridY = gprms.GridY;
    const float &particle_mass = gprms.ParticleMass;

    icy::Point &p = gpu_points[pt_idx];

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

__global__ void kernel_update_nodes(const int nGridNodes, float indenter_x, float indenter_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nGridNodes) return;

    icy::GridNode &gn = gpu_nodes[idx];
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
}

__global__ void kernel_g2p(const int nPoints)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPoints) return;

    icy::Point &p = gpu_points[pt_idx];

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


__device__ void NACCUpdateDeformationGradient(icy::Point &p, Eigen::Matrix2f &FModifier)
{
    constexpr float magic_epsilon = 1e-5f;
    constexpr int d = 2; // dimensions
    float &alpha = p.NACC_alpha_p;
    const float &mu = gprms.mu;
    const float &kappa = gprms.kappa;
    const float &beta = gprms.NACC_beta;
    const float &M_sq = gprms.NACC_M_sq;
    const float &xi = gprms.NACC_xi;
    const bool &hardening = gprms.NACC_hardening;
    const float &dt = gprms.InitialTimeStep;

    Eigen::Matrix2f FeTr = (Eigen::Matrix2f::Identity() + dt * FModifier) * p.Fe;

    Eigen::Matrix2f U, V, Sigma;

    svd2x2(FeTr, U, Sigma, V);

    // line 4
    float p0 = kappa * (magic_epsilon + sinhf(xi * fmaxf(-alpha, 0.f)));

    // line 5
    float Je_tr = Sigma(0,0)*Sigma(1,1);    // this is for 2D

    // line 6
    Eigen::Matrix2f SigmaSquared = Sigma*Sigma;
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
    //p.visualized_value = alpha;
}



/**
\brief 2x2 SVD (singular value decomposition) a=USV'
\param[in] a Input matrix.
\param[out] u Robustly a rotation matrix.
\param[out] sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
\param[out] v Robustly a rotation matrix.
*/
__device__ void svd(const float a[4], float u[4], float sigma[2], float v[4])
{
    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(a, gu, sigma, gv);
    gu.template fill<2, float>(u);
    gv.template fill<2, float>(v);
}


__device__ void svd2x2(const Eigen::Matrix2f &mA,
                       Eigen::Matrix2f &mU,
                       Eigen::Matrix2f &mS,
                       Eigen::Matrix2f &mV)
{
    float U[4], V[4], S[2];
    float a[4] = {mA(0,0), mA(0,1), mA(1,0), mA(1,1)};
    svd(a, U, S, V);

    mU << U[0],U[1],U[2],U[3];
    mS << S[0],0,0,S[1];
    mV << V[0],V[1],V[2],V[3];
}

