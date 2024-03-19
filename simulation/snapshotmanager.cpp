#include "snapshotmanager.h"
#include "model.h"

#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>

void icy::SnapshotManager::SaveSnapshot(std::string outputDirectory)
{
    std::filesystem::path odp(outputDirectory);
    if(!std::filesystem::is_directory(odp) || !std::filesystem::exists(odp)) std::filesystem::create_directory(odp);

    const int current_frame_number = model->prms.AnimationFrameNumber();
    char fileName[20];
    snprintf(fileName, sizeof(fileName), "d%05d.h5", current_frame_number);
    std::string filePath = outputDirectory + "/" + fileName;
    spdlog::info("saving NC frame {} to file {}", current_frame_number, filePath);

    H5::H5File file(filePath, H5F_ACC_TRUNC);

    hsize_t dims_indenter = model->prms.n_indenter_subdivisions*2;
    H5::DataSpace dataspace_indneter(1, &dims_indenter);
    H5::DataSet dataset_indneter = file.createDataSet("Indenter_2D", H5::PredType::NATIVE_DOUBLE, dataspace_indneter);
    dataset_indneter.write(model->gpu.host_side_indenter_force_accumulator, H5::PredType::NATIVE_DOUBLE);

    hsize_t dims_params = sizeof(icy::SimParams);
    H5::DataSpace dataspace_params(1, &dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    SaveParametersAsAttributes(dataset_params);

    hsize_t dims_points = model->prms.nPtsPitch*icy::SimParams::nPtsArrays;
    H5::DataSpace dataspace_points(1, &dims_points);

    hsize_t chunk_dims = (hsize_t)std::min(1024*256, model->prms.nPts);
    H5::DSetCreatPropList proplist;
    proplist.setChunk(1, &chunk_dims);
    proplist.setDeflate(6);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points, proplist);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

    file.close();



//    hsize_t att_dim = 1;
//    H5::DataSpace att_dspace(1, &att_dim);
//    H5::Attribute att = dataset_points.createAttribute("full_data", H5::PredType::NATIVE_INT,att_dspace);
//    att.write(H5::PredType::NATIVE_INT, &full_data);
}

void icy::SnapshotManager::SaveParametersAsAttributes(H5::DataSet &dataset)
{
    hsize_t att_dim = 1;
    H5::DataSpace att_dspace(1, &att_dim);
    H5::Attribute att_indenter_x = dataset.createAttribute("indenter_x", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_indenter_y = dataset.createAttribute("indenter_y", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_SimulationTime = dataset.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_GridX = dataset.createAttribute("GridX", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_GridY = dataset.createAttribute("GridY", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_nPts = dataset.createAttribute("nPts", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_nPtsPitch = dataset.createAttribute("nPtsPitch", H5::PredType::NATIVE_INT64, att_dspace);

    H5::Attribute att_UpdateEveryNthStep = dataset.createAttribute("UpdateEveryNthStep", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_n_indenter_subdivisions = dataset.createAttribute("n_indenter_subdivisions", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_cellsize = dataset.createAttribute("cellsize", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_IndDiameter = dataset.createAttribute("IndDiameter", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_InitialTimeStep = dataset.createAttribute("InitialTimeStep", H5::PredType::NATIVE_DOUBLE, att_dspace);
    H5::Attribute att_SetupType = dataset.createAttribute("SetupType", H5::PredType::NATIVE_INT, att_dspace);
    H5::Attribute att_Volume = dataset.createAttribute("Volume", H5::PredType::NATIVE_DOUBLE, att_dspace);

    att_indenter_x.write(H5::PredType::NATIVE_DOUBLE, &model->prms.indenter_x);
    att_indenter_y.write(H5::PredType::NATIVE_DOUBLE, &model->prms.indenter_y);
    att_SimulationTime.write(H5::PredType::NATIVE_DOUBLE, &model->prms.SimulationTime);
    att_GridX.write(H5::PredType::NATIVE_INT, &model->prms.GridX);
    att_GridY.write(H5::PredType::NATIVE_INT, &model->prms.GridY);
    att_nPts.write(H5::PredType::NATIVE_INT, &model->prms.nPts);
    att_nPtsPitch.write(H5::PredType::NATIVE_INT64, &model->prms.nPtsPitch);

    att_UpdateEveryNthStep.write(H5::PredType::NATIVE_INT, &model->prms.UpdateEveryNthStep);
    att_n_indenter_subdivisions.write(H5::PredType::NATIVE_INT, &model->prms.n_indenter_subdivisions);
    att_cellsize.write(H5::PredType::NATIVE_DOUBLE, &model->prms.cellsize);
    att_IndDiameter.write(H5::PredType::NATIVE_DOUBLE, &model->prms.IndDiameter);
    att_InitialTimeStep.write(H5::PredType::NATIVE_DOUBLE, &model->prms.InitialTimeStep);
    att_SetupType.write(H5::PredType::NATIVE_INT, &model->prms.SetupType);
    att_Volume.write(H5::PredType::NATIVE_DOUBLE, &model->prms.Volume);
}




void icy::SnapshotManager::ReadSnapshot(std::string fileName)
{
    /*
    if(!std::filesystem::exists(fileName)) return -1;

    std::string numbers = fileName.substr(fileName.length()-8,5);
    int idx = std::stoi(numbers);
    spdlog::info("reading snapshot {}", idx);

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    // read and process SimParams
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams)) throw std::runtime_error("ReadSnapshot: SimParams size mismatch");

    icy::SimParams tmp_params;
    dataset_params.read(&tmp_params, H5::PredType::NATIVE_B8);

    if(tmp_params.nGridPitch != model->prms.nGridPitch || tmp_params.nPtsPitch != model->prms.nPtsPitch)
        model->gpu.cuda_allocate_arrays(tmp_params.nGridPitch,tmp_params.nPtsPitch);
    double ParticleViewSize = model->prms.ParticleViewSize;
    model->prms = tmp_params;
    model->prms.ParticleViewSize = ParticleViewSize;

    // read point data
    H5::DataSet dataset_points = file.openDataSet("Points");
//    H5::Attribute att = dataset_points.openAttribute("full_data");
//    int full_data;
//    att.read(H5::PredType::NATIVE_INT, &full_data);

    dataset_points.read(model->gpu.tmp_transfer_buffer,H5::PredType::NATIVE_DOUBLE);

    model->gpu.transfer_ponts_to_host_finalize(model->points);
    file.close();
    return idx;*/
}

void icy::SnapshotManager::LoadRawPoints(std::string fileName)
{
    spdlog::info("ReadRawPoints {}",fileName);
    if(!std::filesystem::exists(fileName)) throw std::runtime_error("error reading raw points file - no file");;

    spdlog::info("reading raw points file {}",fileName);
    H5::H5File file(fileName, H5F_ACC_RDONLY);

    H5::DataSet dataset = file.openDataSet("Points_Raw_2D");
    hsize_t dims[2] = {};
    dataset.getSpace().getSimpleExtentDims(dims, NULL);
    int nPoints = dims[0];
    if(dims[1]!=icy::SimParams::dim) throw std::runtime_error("error reading raw points file - dimensions mismatch");
    spdlog::info("dims[0] {}, dims[1] {}", dims[0], dims[1]);
    model->prms.nPts = nPoints;

    std::vector<std::array<float, 2>> buffer;
    buffer.resize(nPoints);
    dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

    std::vector<short> grainIDs(nPoints);
    H5::DataSet dataset_grains = file.openDataSet("GrainIDs");
    dataset_grains.read(grainIDs.data(), H5::PredType::NATIVE_INT16);

    H5::Attribute att_volume = dataset_grains.openAttribute("volume");
    float volume;
    att_volume.read(H5::PredType::NATIVE_FLOAT, &volume);
    model->prms.Volume = (double)volume;
    file.close();

    auto result = std::minmax_element(buffer.begin(),buffer.end(), [](std::array<float, 2> &p1, std::array<float, 2> &p2){
        return p1[0]<p2[0];});
    model->prms.xmin = (*result.first)[0];
    model->prms.xmax = (*result.second)[0];
    const float length = model->prms.xmax - model->prms.xmin;

    result = std::minmax_element(buffer.begin(),buffer.end(), [](std::array<float, 2> &p1, std::array<float, 2> &p2){
        return p1[1]<p2[1];});
    model->prms.ymin = (*result.first)[1];
    model->prms.ymax = (*result.second)[1];

    model->prms.ComputeIntegerBlockCoords();

    const double &h = model->prms.cellsize;
    const double box_x = model->prms.GridX*h;

    const double x_offset = (box_x - length)/2;
    const double y_offset = 2*h;

    const double block_left = x_offset;
    const double block_top = model->prms.ymax + y_offset;

    const double r = model->prms.IndDiameter/2;
    const double ht = r - model->prms.IndDepth;
    const double x_ind_offset = sqrt(r*r - ht*ht);

    // set initial indenter position
    model->prms.indenter_x = floor((block_left-x_ind_offset)/h)*h;
    if(model->prms.SetupType == 0)
        model->prms.indenter_y = block_top + ht;
    else if(model->prms.SetupType == 1)
        model->prms.indenter_y = ceil(block_top/h)*h;

    model->prms.indenter_x_initial = model->prms.indenter_x;
    model->prms.indenter_y_initial = model->prms.indenter_y;

    model->prms.ParticleVolume = model->prms.Volume/nPoints;
    model->prms.ParticleMass = model->prms.ParticleVolume * model->prms.Density;

    model->gpu.cuda_allocate_arrays(model->prms.GridTotal, nPoints);
    for(int k=0; k<nPoints; k++)
    {
        Point p;
        p.Reset();
        buffer[k][0] += x_offset;
        buffer[k][1] += y_offset;
        for(int i=0;i<icy::SimParams::dim;i++) p.pos[i] = buffer[k][i];
        p.grain = grainIDs[k];
        p.TransferToBuffer(model->gpu.tmp_transfer_buffer, model->prms.nPtsPitch, k);
    }
    spdlog::info("raw points loaded");

    model->gpu.transfer_ponts_to_device();
    model->Reset();
    model->Prepare();
}
