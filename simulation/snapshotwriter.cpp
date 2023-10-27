#include "snapshotwriter.h"
#include "model.h"
#include <spdlog/spdlog.h>
#include <H5Cpp.h>


icy::SnapshotWriter::SnapshotWriter()
{

}

void icy::SnapshotWriter::SaveSnapshot(std::string fileName)
{
    spdlog::info("writing snapshot {}",fileName);

//    H5::IntType datatype_int(H5::PredType::NATIVE_INT);
//    H5::FloatType datatype_double(H5::PredType::NATIVE_DOUBLE);

    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims_points[2] = {model->points.size()*3,1};
    H5::DataSpace dataspace_points(2, dims_points);

    H5::DSetCreatPropList cparms_points;
    hsize_t chunk_dims[2] = {10000, 1};
    cparms_points.setChunk(2, chunk_dims);
    cparms_points.setDeflate(9);
    H5::DataSet dataset_points = file.createDataSet("Points", H5::PredType::NATIVE_DOUBLE,
                                                    dataspace_points, cparms_points);
    dataset_points.write(model->gpu.tmp_transfer_buffer, H5::PredType::NATIVE_DOUBLE);
    file.close();

    spdlog::info("SaveSnapshot done");
}
