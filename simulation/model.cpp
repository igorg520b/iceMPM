#include "model.h"
#include <spdlog/spdlog.h>

icy::Model::Model()
{
    prms.Reset();
}

bool icy::Model::Step()
{
    spdlog::info("step {} started", prms.SimulationStep);
    real simulation_time = prms.SimulationTime;


    int count_unupdated_steps = 0;
    cudaEventRecord(gpu.eventTimingStart);
    do
    {
        indenter_x = indenter_x_initial + simulation_time*prms.IndVelocity;
        gpu.cuda_reset_grid(grid.size());
        gpu.cuda_p2g(points.size());
        gpu.cuda_update_nodes(grid.size(),indenter_x, indenter_y);
        gpu.cuda_g2p(points.size());

        count_unupdated_steps++;

        simulation_time += prms.InitialTimeStep;
    } while((prms.SimulationStep+count_unupdated_steps) % prms.UpdateEveryNthStep != 0);

    cudaEventRecord(gpu.eventTimingStop);   // we want to time the computation steps excluding data transfer
    processing_current_cycle_data.lock();   // if locked, previous results are not yet processed by the host
    gpu.backup_point_positions(points.size());  // make a copy of nodal positions on the device
    cudaEventRecord(gpu.eventCycleComplete);

    gpu.cuda_transfer_from_device(points);
    cudaEventRecord(gpu.eventDataCopiedToHost);

//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, gpu.eventTimingStart, gpu.eventTimingStop);
//    compute_time_per_cycle = milliseconds/count_unupdated_steps;

    prms.SimulationTime = simulation_time;
    prms.SimulationStep += count_unupdated_steps;

    FinalizeDataTransfer();

    if(prms.SimulationTime >= prms.SimulationEndTime) return false;
    return true;
}


void icy::Model::FinalizeDataTransfer()
{
    cudaEventSynchronize(gpu.eventDataCopiedToHost);

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
    const size_t nPoints = prresult.size();
    points.resize(nPoints);
    prms.PointCountActual = nPoints;
    spdlog::info("finished thinks::PoissonDiskSampling; {} ", nPoints);

    prms.ParticleVolume = block_length*block_height/nPoints;
    prms.ParticleMass = prms.ParticleVolume*prms.Density;
    for(int k = 0; k<nPoints; k++)
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
    }
    grid.resize(prms.GridX*prms.GridY);
    indenter_y = block_height + 2*h + prms.IndDiameter/2 - prms.IndDepth;
    indenter_x = indenter_x_initial = 5*h - prms.IndDiameter/2 - h;

    prms.MemAllocGrid = (real)sizeof(GridNode)*grid.size()/(1024*1024);
    prms.MemAllocPoints = (real)sizeof(Point)*points.size()/(1024*1024);
    prms.MemAllocTotal = prms.MemAllocGrid + prms.MemAllocPoints;
    spdlog::info("icy::Model::Reset(); grid {:03.2f} Mb; points {:03.2f} Mb ; total {:03.2f} Mb",
                 prms.MemAllocGrid, prms.MemAllocPoints, prms.MemAllocTotal);

    gpu.cuda_allocate_arrays(grid.size(), points.size());
    gpu.transfer_ponts_to_device(points);
    spdlog::info("icy::Model::Reset() done");
}

void icy::Model::Prepare()
{
    spdlog::info("icy::Model::Prepare()");
    gpu.cuda_update_constants(prms);
    abortRequested = false;
}








