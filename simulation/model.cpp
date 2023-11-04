#include "model.h"
#include <spdlog/spdlog.h>


bool icy::Model::Step()
{
    real simulation_time = prms.SimulationTime;
    std::cout << '\n';
    spdlog::info("step {} ({}) started; sim_time {}", prms.SimulationStep, prms.SimulationStep/prms.UpdateEveryNthStep, simulation_time);

    int count_unupdated_steps = 0;
    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) == 0) cudaEventRecord(gpu.eventCycleStart);
    do
    {
        indenter_x = indenter_x_initial + simulation_time*prms.IndVelocity;
        gpu.cuda_reset_grid();
        gpu.cuda_p2g();
        gpu.cuda_update_nodes(indenter_x, indenter_y);
        gpu.cuda_g2p();
        count_unupdated_steps++;
        simulation_time += prms.InitialTimeStep;
    } while((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep != 0);

    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) == 0) cudaEventRecord(gpu.eventCycleStop);   // we want to time the computation steps excluding data transfer
    if(prms.SimulationStep % (prms.UpdateEveryNthStep*2) != 0)
    {
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, gpu.eventCycleStart, gpu.eventCycleStop);
        compute_time_per_cycle = milliseconds/prms.UpdateEveryNthStep;
        spdlog::info("cycle time {} ms", compute_time_per_cycle);
    }

    processing_current_cycle_data.lock();   // if locked, previous results are not yet processed by the host
    prms.SimulationTime = simulation_time;
    prms.SimulationStep += count_unupdated_steps;

    gpu.cuda_transfer_from_device();

    if(prms.SimulationTime >= prms.SimulationEndTime || gpu.error_code) return false;
    return true;
}


void icy::Model::FinalizeDataTransfer()
{
    hostside_data_update_mutex.lock();
    gpu.transfer_ponts_to_host_finalize(points);
    hostside_data_update_mutex.unlock();
}

void icy::Model::UnlockCycleMutex()
{
    // current data was handled by host - allow next cycle to proceed
    processing_current_cycle_data.unlock();
}


void icy::Model::Reset()
{
    // this should be called after prms are set as desired (either via GUI or CLI)
    spdlog::info("icy::Model::Reset()");

    prms.SimulationStep = 0;
    prms.SimulationTime = 0;
    compute_time_per_cycle = 0;

    const real &block_length = prms.BlockLength;
    const real &block_height = prms.BlockHeight;
    const real &h = prms.cellsize;

    const real kRadius = sqrt(block_length*block_height/(prms.PointsWanted*(0.5*SimParams::pi)*100./97.));
    const std::array<real, 2>kXMin{5.0*h, 2.0*h};
    const std::array<real, 2>kXMax{5.0*h+block_length, 2.0*h+block_height};
    spdlog::info("starting thinks::PoissonDiskSampling");
    std::vector<std::array<real, 2>> prresult = thinks::PoissonDiskSampling(kRadius, kXMin, kXMax);
    spdlog::info("finished thinks::PoissonDiskSampling; {} ", prms.nPts);
    prms.nPts = prresult.size();
    points.resize(prms.nPts);

    prms.ParticleVolume = block_length*block_height/prms.nPts;
    prms.ParticleMass = prms.ParticleVolume*prms.Density;
    for(int k = 0; k<prms.nPts; k++)
    {
        Point &p = points[k];
        p.pos[0] = prresult[k][0];
        p.pos[1] = prresult[k][1];
        p.Fe.setIdentity();
        p.velocity.setZero();
//        p.velocity.x() = 1.f + (p.pos.y()-1.5)/2;
//        p.velocity.y() = 2.f + (-p.pos.x()-1.5)/2;
        //p.Fp.setIdentity();
        p.Bp.setZero();
        p.NACC_alpha_p = prms.NACC_alpha;
        p.q = 0;
    }
    indenter_y = block_height + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    indenter_x = indenter_x_initial = 5*h - prms.IndDiameter/2 - h;

    gpu.cuda_allocate_arrays(prms.GridX*prms.GridY, prms.nPts);
    gpu.transfer_ponts_to_device(points);
    Prepare();
    spdlog::info("icy::Model::Reset() done");
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    abortRequested = false;
    gpu.cuda_update_constants();
}

