#include "snapshotmanager.h"
#include "model.h"
#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <filesystem>
#include <string>


void icy::SnapshotManager::SaveSnapshot(std::string fileName)
{
    spdlog::info("writing snapshot {}",fileName);

    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_params = sizeof(icy::SimParams);
    H5::DataSpace dataspace_params(1,&dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    int fullDataArrays = icy::SimParams::nPtsArrays;
    hsize_t nPtsPitched = model->prms.nPtsPitch/sizeof(real);
    hsize_t dims_points = nPtsPitched*fullDataArrays;

    H5::DataSpace dataspace_points(1, &dims_points);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

//    hsize_t att_dim = 1;
//    H5::DataSpace att_dspace(1, &att_dim);
//    H5::Attribute att = dataset_points.createAttribute("full_data", H5::PredType::NATIVE_INT,att_dspace);
//    att.write(H5::PredType::NATIVE_INT, &full_data);


    file.close();
    spdlog::info("SaveSnapshot done {}", fileName);
}

int icy::SnapshotManager::ReadSnapshot(std::string fileName)
{
    if(!std::filesystem::exists(fileName)) return -1;
    std::string numbers = fileName.substr(fileName.length()-8,5);
    int idx = std::stoi(numbers);

    H5::H5File file(fileName, H5F_ACC_RDONLY);

    // read and process SimParams
    H5::DataSet dataset_params = file.openDataSet("Params");
    hsize_t dims_params = 0;
    dataset_params.getSpace().getSimpleExtentDims(&dims_params, NULL);
    if(dims_params != sizeof(icy::SimParams)) throw std::runtime_error("ReadSnapshot: SimParams size mismatch");

    icy::SimParams tmp_params;
    dataset_params.read(&tmp_params, H5::PredType::NATIVE_B8);

    if(tmp_params.nGridPitch != model->prms.nGridPitch || tmp_params.nPtsPitch != model->prms.nPtsPitch)
        model->gpu.cuda_allocate_arrays(tmp_params.nGridPitch/sizeof(real),tmp_params.nPtsPitch/sizeof(real));
    model->prms = tmp_params;


    // read point data
    H5::DataSet dataset_points = file.openDataSet("Points");
//    H5::Attribute att = dataset_points.openAttribute("full_data");
//    int full_data;
//    att.read(H5::PredType::NATIVE_INT, &full_data);

    dataset_points.read(model->gpu.tmp_transfer_buffer,H5::PredType::NATIVE_DOUBLE);

    model->gpu.transfer_ponts_to_host_finalize(model->points);
    file.close();
    return idx;
}

void icy::SnapshotManager::ReadDirectory(std::string directoryPath)
{
    path = directoryPath;
    // set last_file_index
    for (const auto & entry : std::filesystem::directory_iterator(directoryPath))
    {
        std::string fileName = entry.path();
        std::string extension = fileName.substr(fileName.length()-3,3);
        if(extension != ".h5") continue;
        std::string numbers = fileName.substr(fileName.length()-8,5);
        int idx = std::stoi(numbers);
        if(idx > last_file_index) last_file_index = idx;

//        std::cout << fileName << ", " << extension << ", " << numbers << std::endl;
    }
    spdlog::info("directory scanned; last_file_index is {}", last_file_index);

}

