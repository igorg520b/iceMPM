#include "snapshotwriter.h"
#include "model.h"
#include <spdlog/spdlog.h>
#include <H5Cpp.h>



void icy::SnapshotWriter::SaveSnapshot(std::string fileName, bool fullData)
{
    spdlog::info("writing snapshot {}",fileName);

    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_params = sizeof(icy::SimParams);
    H5::DataSpace dataspace_params(1,&dims_params);
    H5::DataSet dataset_params = file.createDataSet("Params", H5::PredType::NATIVE_B8, dataspace_params);
    dataset_params.write(&model->prms, H5::PredType::NATIVE_B8);

    constexpr int fullDataArrays = 10;
    constexpr int abridgedDataArrays = 3;
    hsize_t nPtsPitched = model->prms.nPtsPitch/sizeof(real);
    hsize_t dims_points = fullData ? nPtsPitched*fullDataArrays : nPtsPitched*abridgedDataArrays;
    int full_data = fullData ? 1 : 0;

    H5::DataSpace dataspace_points(1, &dims_points);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE, dataspace_points);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);

    hsize_t att_dim = 1;
    H5::DataSpace att_dspace(1, &att_dim);
    H5::Attribute att = dataset_points.createAttribute("full_data", H5::PredType::NATIVE_INT,att_dspace);
    att.write(H5::PredType::NATIVE_INT, &full_data);


    file.close();
    spdlog::info("SaveSnapshot done {}", fileName);
}

void icy::SnapshotWriter::ReadSnapshot(std::string fileName)
{
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
    H5::Attribute att = dataset_points.openAttribute("full_data");
    int full_data;
    //H5::DataSpace att_dataspace = att.getSpace();
    att.read(H5::PredType::NATIVE_INT, &full_data);

//    hsize_t dims_nodes;
//    dataset_nodes.getSpace().getSimpleExtentDims(&dims_nodes, NULL);
    dataset_points.read(model->gpu.tmp_transfer_buffer,H5::PredType::NATIVE_DOUBLE);

    model->gpu.transfer_ponts_to_host_finalize(model->points);
    file.close();

    const real &h = model->prms.cellsize;

    model->indenter_y = model->prms.BlockHeight + 2*h + model->prms.IndDiameter/2 - model->prms.IndDepth;
    double indenter_x_initial = 5*h - model->prms.IndDiameter/2 - h;
    model->indenter_x = indenter_x_initial + model->prms.SimulationTime*model->prms.IndVelocity;
}
